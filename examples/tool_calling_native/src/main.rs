use clap::Parser;
use llama_cpp_2::{
    chat::{
        chat_format_name, parse_chat_response, ChatMessage, ChatSyntax, ChatTemplateInputs,
        ChatTemplates, ChatTool, ChatToolChoice, ReasoningFormat,
    },
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde_json::json;
use std::{collections::HashSet, io::Write, num::NonZeroU32, path::PathBuf, time::Instant};
use uuid::Uuid;

#[derive(Parser)]
#[command(about = "Tool calling example with native token classification")]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(
        short,
        long,
        default_value = "What's the weather like in San Francisco?"
    )]
    prompt: String,

    #[arg(short = 'n', long, default_value = "2048")]
    max_tokens: u32,

    #[arg(short, long, default_value = "0.7")]
    temperature: f32,

    #[arg(short = 'k', long, default_value = "0.95")]
    top_p: f32,

    #[arg(short = 'c', long, default_value = "false")]
    use_cpu: bool,

    #[arg(short = 'd', long)]
    debug: bool,

    #[arg(long, help = "Enable OpenAI-compatible streaming")]
    oai_stream: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🦙 Llama.cpp Tool Calling Example");
    println!("Model: {}", args.model.display());
    println!("Prompt: {}", args.prompt);
    println!("Max tokens: {}", args.max_tokens);
    println!("Temperature: {}", args.temperature);
    println!("Top-p: {}", args.top_p);
    println!("Use CPU: {}", args.use_cpu);
    println!("Debug: {}", args.debug);
    if args.oai_stream {
        println!("OpenAI Stream: enabled");
    }
    println!();

    // Initialize backend
    let backend = LlamaBackend::init()?;

    // Load model
    let model_params = if cfg!(feature = "metal") || cfg!(feature = "cuda") && !args.use_cpu {
        LlamaModelParams::default().with_n_gpu_layers(1000)
    } else {
        LlamaModelParams::default()
    };

    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)?;

    // Initialize chat templates
    let chat_templates = ChatTemplates::new(&model, None, None, None)?;

    println!("✅ Model loaded successfully");
    println!(
        "📝 Chat template was {}",
        if chat_templates.was_explicit() {
            "explicit"
        } else {
            "auto-detected"
        }
    );

    // Define tools
    let tools = vec![
        ChatTool {
            name: "get_weather".to_string(),
            description: "Get current weather for a location".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            })
            .to_string(),
        },
        ChatTool {
            name: "calculate".to_string(),
            description: "Perform mathematical calculations".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            })
            .to_string(),
        },
    ];

    // Create messages
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant that can get weather information and perform calculations. Use the available tools when needed.".to_string(),
            content_parts: vec![],
            tool_calls: vec![],
            reasoning_content: None,
            tool_name: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: args.prompt.clone(),
            content_parts: vec![],
            tool_calls: vec![],
            reasoning_content: None,
            tool_name: None,
            tool_call_id: None,
        },
    ];

    // Create template inputs
    let inputs = ChatTemplateInputs {
        messages,
        grammar: None,
        json_schema: None,
        add_generation_prompt: true,
        use_jinja: true,
        tools,
        tool_choice: ChatToolChoice::Auto,
        parallel_tool_calls: false,
        reasoning_format: ReasoningFormat::DeepSeek,
        enable_thinking: true,
    };

    // Apply chat template
    let chat_params = chat_templates.apply(&inputs)?;

    println!("🔧 Chat template applied successfully");
    println!("📋 Format: {}", chat_format_name(chat_params.format));
    println!("📏 Prompt length: {} characters", chat_params.prompt.len());
    println!();

    // Initialize context
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(4096));
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // Tokenize prompt
    let tokens = model.str_to_token(&chat_params.prompt, AddBos::Always)?;

    println!("🔤 Tokenized prompt: {} tokens", tokens.len());

    // Collect preserved token IDs from chat template, like in server.cpp
    let preserved_token_ids: HashSet<LlamaToken> = chat_params
        .preserved_tokens
        .iter()
        .flat_map(|s| model.str_to_token(s, AddBos::Never).unwrap_or_default())
        .collect();

    if args.debug && !preserved_token_ids.is_empty() {
        println!("[DEBUG] Preserved token IDs: {:?}", preserved_token_ids);
    }

    // Process prompt
    let mut batch = LlamaBatch::new(512, 1);
    let t_start = Instant::now();

    // Add tokens to batch
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(token, i as i32, &[0], is_last)?;

        if batch.n_tokens() == 512 || is_last {
            ctx.decode(&mut batch)?;
            batch.clear();
        }
    }

    let prompt_time = t_start.elapsed();
    println!("⏱️  Prompt processed in {:?}", prompt_time);

    // Initialize sampler
    let mut sampler = LlamaSampler::chain_simple(
        [
            Some(LlamaSampler::penalties(64, 1.1, 0.0, 0.0)),
            Some(LlamaSampler::top_p(args.top_p, 1)),
            Some(LlamaSampler::temp(args.temperature)),
            Some(LlamaSampler::dist(42)),
            Some(LlamaSampler::greedy()),
        ]
        .into_iter()
        .flatten(),
    );

    // Generation loop following server.cpp approach
    let mut generated_text = String::new();
    let mut n_cur = tokens.len() as i32;
    let t_gen_start = Instant::now();
    let mut last_parsed_message = ChatMessage::default();
    let completion_id = format!("cmpl-{}", Uuid::new_v4());
    let mut is_first_chunk = true;
    let mut end_of_generation_token_reached = false;

    // Create chat syntax for parsing (when needed)
    let chat_syntax = ChatSyntax {
        format: chat_params.format,
        reasoning_format: ReasoningFormat::DeepSeek,
        reasoning_in_content: false,
        thinking_forced_open: false,
        parse_tool_calls: true,
    };

    println!("🤖 Starting generation...\n");

    // Helper function to determine if a token should be treated as special (following server.cpp)
    let accept_special_token = |token| -> bool {
        // In server.cpp this checks: params_base.special || slot.params.sampling.preserved_tokens.find(token) != end()
        // We will use the preserved tokens from the chat template.
        preserved_token_ids.contains(&token)
    };

    for _step in 0..args.max_tokens {
        // Sample next token
        let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(new_token);

        // Check if end of generation
        if model.is_eog_token(new_token) {
            println!("\n🏁 End of generation token reached");
            end_of_generation_token_reached = true;
            break;
        }

        // Convert token to text following server.cpp approach
        let token_str = if accept_special_token(new_token) {
            // Handle special tokens by getting their raw representation
            let token_bytes = model.token_to_bytes(new_token, Special::Plaintext)?;
            String::from_utf8_lossy(&token_bytes).to_string()
        } else {
            // Handle regular tokens
            let token_bytes = model.token_to_bytes(new_token, Special::Tokenize)?;
            String::from_utf8_lossy(&token_bytes).to_string()
        };

        // Accumulate generated text (following server.cpp approach)
        generated_text.push_str(&token_str);

        if args.oai_stream {
            if let Ok(new_message) = parse_chat_response(&generated_text, true, &chat_syntax) {
                if is_first_chunk {
                    let chunk = json!({
                        "id": &completion_id,
                        "object": "chat.completion.chunk",
                        "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                        "model": args.model.to_string_lossy(),
                        "choices": [{
                            "index": 0,
                            "delta": { "role": "assistant", "content": null },
                            "finish_reason": null
                        }]
                    });
                    println!("data: {}", chunk.to_string());
                    println!();
                    std::io::stdout().flush().unwrap();
                    is_first_chunk = false;
                }

                // --- simple Rust-side diff (avoid FFI seg-faults) ---
                let mut delta_obj = json!({});

                // Reasoning delta
                if let (Some(prev), Some(new)) = (
                    &last_parsed_message.reasoning_content,
                    &new_message.reasoning_content,
                ) {
                    if new.len() > prev.len() {
                        delta_obj["reasoning"] = json!(new[prev.len()..].to_string());
                    }
                } else if let Some(new) = &new_message.reasoning_content {
                    if !new.is_empty() && last_parsed_message.reasoning_content.is_none() {
                        delta_obj["reasoning"] = json!(new);
                    }
                }

                // Content delta (simple suffix diff)
                if new_message.content.len() > last_parsed_message.content.len() {
                    delta_obj["content"] =
                        json!(new_message.content[last_parsed_message.content.len()..].to_string());
                }

                // Tool-call delta (any newly added calls)
                if new_message.tool_calls.len() > last_parsed_message.tool_calls.len() {
                    let idx = last_parsed_message.tool_calls.len();
                    for (i, tc) in new_message.tool_calls[idx..].iter().enumerate() {
                        delta_obj["tool_calls"] = json!([{
                            "index": idx + i,
                            "id": tc.id,
                            "type": "function",
                            "function": { "name": tc.name, "arguments": tc.arguments }
                        }]);
                    }
                }

                if delta_obj.as_object().map_or(false, |m| !m.is_empty()) {
                    let chunk = json!({
                        "id": &completion_id,
                        "object": "chat.completion.chunk",
                        "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                        "model": args.model.to_string_lossy(),
                        "choices": [{
                            "index": 0,
                            "delta": delta_obj,
                            "finish_reason": null
                        }]
                    });
                    println!("data: {}", chunk.to_string());
                    println!();
                    std::io::stdout().flush().unwrap();
                }
                last_parsed_message = new_message;
            }
        } else {
            // Print token immediately for streaming effect
            print!("{}", token_str);
            std::io::stdout().flush().unwrap();
        }

        // Prepare for next iteration
        batch.clear();
        batch.add(new_token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch)?;
    }

    if args.oai_stream {
        // Send final chunk with finish_reason
        let final_msg =
            parse_chat_response(&generated_text, false, &chat_syntax).unwrap_or_default();
        let finish_reason = if !final_msg.tool_calls.is_empty() {
            "tool_calls"
        } else if end_of_generation_token_reached {
            "stop"
        } else {
            "length"
        };
        let chunk = json!({
            "id": &completion_id,
            "object": "chat.completion.chunk",
            "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            "model": args.model.to_string_lossy(),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        println!("data: {}", chunk.to_string());
        println!();
        std::io::stdout().flush().unwrap();

        println!("data: [DONE]");
        println!();
        std::io::stdout().flush().unwrap();
    }

    let gen_time = t_gen_start.elapsed();

    if !args.oai_stream {
        println!("\n\n📊 Generation Statistics:");
        println!("⏱️  Total time: {:?}", gen_time);
        println!("🔤 Tokens generated: {}", n_cur - tokens.len() as i32);
        println!(
            "🚀 Tokens/sec: {:.2}",
            (n_cur - tokens.len() as i32) as f64 / gen_time.as_secs_f64()
        );

        // Final structured parsing (following server.cpp approach)
        println!("\n🎯 Final Parse Results:");
        if let Ok(final_msg) = parse_chat_response(&generated_text, false, &chat_syntax) {
            if let Some(reasoning) = &final_msg.reasoning_content {
                if !reasoning.is_empty() {
                    println!("💭 [REASONING] {}", reasoning);
                }
            }
            if !final_msg.content.is_empty() {
                println!("💬 [CONTENT] {}", final_msg.content);
            }
            for (i, tool_call) in final_msg.tool_calls.iter().enumerate() {
                println!(
                    "🔧 [TOOL CALL {}] Name: {}, Args: {}",
                    i, tool_call.name, tool_call.arguments
                );
            }
        }
    }

    Ok(())
}
