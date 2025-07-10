# Tool Calling Native Example

This example demonstrates how to implement tool calling with native token classification using the llama-cpp-rs bindings.

## Key Features

- **Structured Message Parsing**: Uses the same mechanism as `server.cpp` - accumulates tokens and parses them into structured message components
- **Token Classification**: Classifies tokens by their semantic meaning (thinking, content, tool calls) using `ChatMessage::compute_diffs()`
- **Real-time Streaming**: Provides classified token streaming with appropriate labels
- **Tool Support**: Includes example tools for weather and calculations

## How It Works

Instead of trying to classify individual tokens, this example:

1. **Accumulates tokens** into a complete text buffer during generation
2. **Periodically parses** the buffer into structured `ChatMessage` objects
3. **Uses `compute_diffs()`** to determine what content changed and what type it is
4. **Streams classified content** with appropriate labels (💭 THINKING, 💬 CONTENT, 🔧 TOOL CALL)

This matches the `server.cpp` approach where token classification comes from message structure analysis rather than individual token inspection.

## Usage

```bash
# Build the example
cargo build --release

# Run with a model
cargo run --release -- --model /path/to/your/model.gguf

# Customize the prompt
cargo run --release -- --model /path/to/your/model.gguf --prompt "What's 2+2 and what's the weather in Tokyo?"

# Use CPU only
cargo run --release -- --model /path/to/your/model.gguf --use-cpu

# Adjust generation parameters
cargo run --release -- --model /path/to/your/model.gguf --temperature 0.3 --top-p 0.9 --max-tokens 1000

# Using short options
cargo run --release -- -m /path/to/your/model.gguf -p "What's 2+2?" -n 500 -t 0.5 -k 0.9

# Enable debug mode to see parsing attempts
cargo run --release -- -m /path/to/your/model.gguf -p "What's the weather?" --debug
```

## Command Line Options

- `--model` / `-m`: Path to the GGUF model file (required)
- `--prompt` / `-p`: Input prompt (default: "What's the weather like in San Francisco?")
- `--max-tokens` / `-n`: Maximum tokens to generate (default: 2048)
- `--temperature` / `-t`: Sampling temperature (default: 0.7)
- `--top-p` / `-k`: Top-p sampling (default: 0.95)
- `--use-cpu` / `-c`: Force CPU usage instead of GPU
- `--debug` / `-d`: Enable debug output to show parsing attempts

## Example Output

```
🦙 Llama.cpp Tool Calling Example
Model: /path/to/model.gguf
Prompt: What's the weather like in San Francisco?
...

🤖 Starting generation with structured parsing...

💭 [THINKING] I need to get weather information for San Francisco. I'll use the get_weather tool.

💬 [CONTENT] I'll help you get the current weather for San Francisco.

🔧 [TOOL CALL 0] Name: get_weather, Args: {"location": "San Francisco, CA", "unit": "fahrenheit"}

🎯 Final Parsed Message:
Role: assistant
💭 Reasoning: I need to get weather information for San Francisco. I'll use the get_weather tool.
💬 Content: I'll help you get the current weather for San Francisco.
🔧 Tool Calls:
  [0] get_weather: {"location": "San Francisco, CA", "unit": "fahrenheit"}
```

## Model Requirements

This example works best with models that support:

- Function calling (e.g., Hermes, Functionary, Command-R)
- Reasoning/thinking (e.g., DeepSeek-R1, QwQ)
- Chat templates with tool support

## Architecture

The example demonstrates the key insight from `server.cpp`: **token classification comes from message structure analysis, not individual token inspection**. This approach is more robust and provides better semantic understanding of the generated content.
