### Two-machine inference with llama-cpp-rs (no RPC)

This hands-on guide shows what you can realistically do today to leverage two local machines for inference using Rust and `llama-cpp-rs`, without using llama.cpp RPC.

It covers:
- What is and isn’t supported for true model-parallel across machines
- A practical, working "state handoff" demo using `llama-cpp-rs` (single model, two machines, sequential baton pass)
- An outline for distributed speculative decoding (two machines, two models) for real speedups
- Notes on single-host multi-GPU splitting (for context)

---

## 0) Capabilities and constraints

- llama.cpp supports splitting across multiple GPUs on a single host (e.g., `tensor_split`, `n_gpu_layers`) and many backends (CUDA, Metal, Vulkan, SYCL…).
- llama.cpp does not natively support splitting a single model across multiple machines for one-step token evaluation (layer/tensor parallel across nodes) unless you use its RPC toolchain. This README avoids RPC by request.
- Without RPC, you can still:
  - Transfer context state (KV cache, RNG, etc.) between machines and continue generation on a different host (sequential baton pass)
  - Use two machines cooperatively via speculative decoding (small draft model on one node, target model on the other) to get real speedups

Why true multi-node model parallel is hard: you’d need to stream per-layer activations across the network every step (shape roughly n_tokens×hidden_size each time), which adds high latency and bandwidth requirements and needs deep integration with ggml backends.

---

## 1) Hands-on: State handoff (single model, two machines, sequential)

This pattern lets you prefill on Machine A and then continue generation on Machine B by copying the llama.cpp context state (including KV cache) over the network. It’s not parallel, but it’s a concrete, supported way to “split” a session across machines without RPC.

Prereqs
- Both machines have Rust installed and can build/run `llama-cpp-rs`.
- Both machines have the exact same GGUF model file, with identical `LlamaContextParams` (e.g., `n_ctx`, embedding flags) and backend configuration. Quantization and model must match exactly.
- Network connectivity between machines.

Key API used (from `llama-cpp-rs` wrappers over llama.cpp):
- `LlamaContext::get_state_size()`
- `LlamaContext::copy_state_data(dest_ptr)`
- `LlamaContext::set_state_data(&bytes)`

Flow
1. Machine A: load model, build context, tokenize and decode the prompt (prefill).
2. Machine A: allocate `Vec<u8>` of size `ctx.get_state_size()`, then `copy_state_data` into it.
3. Send bytes over the network to Machine B (TCP/file/pipe—your choice).
4. Machine B: load the same model and create a context with identical params. Call `set_state_data(&bytes)` and continue generation.

Example snippets (trimmed to essentials)

Sender (Machine A):
```rust
use anyhow::Result;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaModel, AddBos, Special};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;

fn main() -> Result<()> {
    // init
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(&backend, "PATH/TO/MODEL.gguf", &LlamaModelParams::default())?;
    let ctx_params = LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // prefill prompt
    let prompt = "Hello, my name is";
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1);
    let last = (tokens.len() as i32) - 1;
    for (i, t) in (0_i32..).zip(tokens.iter().copied()) {
        batch.add(t, i, &[0], i == last)?; // logits for last only
    }
    ctx.decode(&mut batch)?;

    // serialize state
    let sz = ctx.get_state_size();
    let mut buf = vec![0u8; sz];
    unsafe { ctx.copy_state_data(buf.as_mut_ptr()); }

    // send buf over the network (TCP/UDP/file/etc.). Here you’d write buf to a TcpStream.
    // std::net::TcpStream::connect("MACHINE_B:PORT")?.write_all(&buf)?;
    Ok(())
}
```

Receiver (Machine B):
```rust
use anyhow::Result;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaModel, Special};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;

fn main() -> Result<()> {
    // read buf from the network; here assume you have it in `buf: Vec<u8>`
    let buf: Vec<u8> = receive_bytes_somehow()?;

    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(&backend, "PATH/TO/MODEL.gguf", &LlamaModelParams::default())?;
    let ctx_params = LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // restore state
    unsafe { ctx.set_state_data(&buf); }

    // continue generation
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);
    let mut batch = LlamaBatch::new(512, 1);
    let mut n_cur = 0i32;
    loop {
        let token = sampler.sample(&ctx, ctx.token_data_array().last_selected_index());
        sampler.accept(token);
        if model.is_eog_token(token) { break; }
        let piece = model.token_to_bytes(token, Special::Tokenize)?;
        print!("{}", String::from_utf8_lossy(&piece));
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }
    Ok(())
}

fn receive_bytes_somehow() -> Result<Vec<u8>> {
    // e.g., read from a TcpListener and read_exact into a Vec
    unimplemented!()
}
```

Notes
- The model path, quantization, and context params must match exactly across machines.
- If you use GPU backends, configure both sides consistently.
- This method transfers the entire state once; you can repeat the cycle (generate N tokens, handoff again) but each handoff incurs network latency.

---

## 2) Outline: Distributed speculative decoding (two machines, real speedups)

This uses two models:
- Machine A runs a small “draft” model to propose K tokens quickly.
- Machine B runs the target model to verify/accept as many proposed tokens as possible in one step.

Why it helps: A can batch-propose several tokens at once; B verifies in fewer forward passes, often accepting multiple tokens per verification. Over the network, you exchange only token IDs and (optionally) logits or acceptance windows.

Sketch
1. Coordinator process (can run on A or standalone) orchestrates message flow.
2. A (draft): given current prefix, proposes a sequence of candidate tokens T_draft.
3. B (target): verifies T_draft in one or few forwards, returns how many tokens were accepted `n_accept` and the last accepted token/logits.
4. If `n_accept > 0`, append those tokens to the shared prefix, repeat.
5. If `n_accept == 0`, fallback to generating 1 token on the target, update prefix, continue.

Implementation tips with `llama-cpp-rs`
- Both A and B each load their own `LlamaModel` and `LlamaContext`.
- A uses a small GGUF (same family as target for better alignment) to propose tokens.
- The coordinator exchanges JSON messages over TCP (e.g., tokio + serde_json): `{ tokens: [..] }`, `{ accepted: n }`.
- On B, pack candidate tokens into a `LlamaBatch` and evaluate; compute acceptance by checking whether target’s greedy path matches draft tokens step-by-step.

This pattern is battle-tested in llama.cpp’s examples (C++), and porting the control flow to Rust is straightforward once you have the per-step decode and sampling in place.

---

## 3) Single-host multi-GPU (context only)

If your goal is to split a single model across multiple GPUs, llama.cpp supports this on a single host via:
- Offloading layers to GPU (`n_gpu_layers`)
- Row/layer split modes and `tensor_split` to distribute tensors across available devices

This README avoids CLI specifics to prevent drift; refer to the upstream llama.cpp docs for the current flags and examples.

---

## References
- llama.cpp README (features/backends, speculative decoding, server): see upstream repository
- `llama-cpp-rs` crate docs: docs.rs for `llama-cpp-2`
- Relevant APIs in this repo:
  - `llama-cpp-2/src/context/session.rs` for state save/restore
  - `llama-cpp-2/src/llama_batch.rs` and `llama-cpp-2/src/context.rs` for batched decode



