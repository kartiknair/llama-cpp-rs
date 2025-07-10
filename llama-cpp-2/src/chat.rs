use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::ptr;

use crate::model::LlamaModel;
use llama_cpp_sys_2::*;

/// Tool choice for chat completion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatToolChoice {
    Auto,
    Required,
    None,
}

impl From<ChatToolChoice> for c_chat_tool_choice {
    fn from(choice: ChatToolChoice) -> Self {
        match choice {
            ChatToolChoice::Auto => C_CHAT_TOOL_CHOICE_AUTO,
            ChatToolChoice::Required => C_CHAT_TOOL_CHOICE_REQUIRED,
            ChatToolChoice::None => C_CHAT_TOOL_CHOICE_NONE,
        }
    }
}

/// Reasoning format for handling thinking/reasoning content
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFormat {
    None,
    DeepSeek,
    DeepSeekLegacy,
}

impl From<ReasoningFormat> for c_reasoning_format {
    fn from(format: ReasoningFormat) -> Self {
        match format {
            ReasoningFormat::None => C_REASONING_FORMAT_NONE,
            ReasoningFormat::DeepSeek => C_REASONING_FORMAT_DEEPSEEK,
            ReasoningFormat::DeepSeekLegacy => C_REASONING_FORMAT_DEEPSEEK_LEGACY,
        }
    }
}

/// Chat syntax options for parsing
#[derive(Debug, Clone)]
pub struct ChatSyntax {
    pub format: ChatFormat,
    pub reasoning_format: ReasoningFormat,
    pub reasoning_in_content: bool,
    pub thinking_forced_open: bool,
    pub parse_tool_calls: bool,
}

impl Default for ChatSyntax {
    fn default() -> Self {
        Self {
            format: ChatFormat::ContentOnly,
            reasoning_format: ReasoningFormat::None,
            reasoning_in_content: false,
            thinking_forced_open: false,
            parse_tool_calls: true,
        }
    }
}

/// Chat format detected by the template system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    ContentOnly,
    Generic,
    MistralNemo,
    Llama3X,
    Llama3XWithBuiltinTools,
    DeepSeekR1,
    FirefunctionV2,
    FunctionaryV32,
    FunctionaryV31Llama31,
    Hermes2Pro,
    CommandR7B,
}

impl From<c_chat_format> for ChatFormat {
    fn from(format: c_chat_format) -> Self {
        match format {
            C_CHAT_FORMAT_CONTENT_ONLY => ChatFormat::ContentOnly,
            C_CHAT_FORMAT_GENERIC => ChatFormat::Generic,
            C_CHAT_FORMAT_MISTRAL_NEMO => ChatFormat::MistralNemo,
            C_CHAT_FORMAT_LLAMA_3_X => ChatFormat::Llama3X,
            C_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS => ChatFormat::Llama3XWithBuiltinTools,
            C_CHAT_FORMAT_DEEPSEEK_R1 => ChatFormat::DeepSeekR1,
            C_CHAT_FORMAT_FIREFUNCTION_V2 => ChatFormat::FirefunctionV2,
            C_CHAT_FORMAT_FUNCTIONARY_V3_2 => ChatFormat::FunctionaryV32,
            C_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1 => ChatFormat::FunctionaryV31Llama31,
            C_CHAT_FORMAT_HERMES_2_PRO => ChatFormat::Hermes2Pro,
            C_CHAT_FORMAT_COMMAND_R7B => ChatFormat::CommandR7B,
            _ => ChatFormat::ContentOnly,
        }
    }
}

/// A tool that can be called by the model
#[derive(Debug, Clone)]
pub struct ChatTool {
    pub name: String,
    pub description: String,
    pub parameters: String,
}

/// A tool call made by the model
#[derive(Debug, Clone)]
pub struct ChatToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

/// A content part within a chat message
#[derive(Debug, Clone)]
pub struct ChatMessageContentPart {
    pub content_type: String,
    pub text: String,
}

/// A chat message
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub content_parts: Vec<ChatMessageContentPart>,
    pub tool_calls: Vec<ChatToolCall>,
    pub reasoning_content: Option<String>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
}

/// A difference between two chat messages
#[derive(Debug, Clone)]
pub struct ChatMessageDiff {
    pub reasoning_content_delta: Option<String>,
    pub content_delta: Option<String>,
    pub tool_call_index: Option<usize>,
    pub tool_call_delta: Option<ChatToolCall>,
}

/// Collection of chat message differences
#[derive(Debug, Clone)]
pub struct ChatMessageDiffs {
    pub diffs: Vec<ChatMessageDiff>,
}

/// Input for chat template application
#[derive(Debug, Clone)]
pub struct ChatTemplateInputs {
    pub messages: Vec<ChatMessage>,
    pub grammar: Option<String>,
    pub json_schema: Option<String>,
    pub add_generation_prompt: bool,
    pub use_jinja: bool,
    pub tools: Vec<ChatTool>,
    pub tool_choice: ChatToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning_format: ReasoningFormat,
    pub enable_thinking: bool,
}

impl Default for ChatTemplateInputs {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            grammar: None,
            json_schema: None,
            add_generation_prompt: true,
            use_jinja: true,
            tools: Vec::new(),
            tool_choice: ChatToolChoice::Auto,
            parallel_tool_calls: false,
            reasoning_format: ReasoningFormat::None,
            enable_thinking: true,
        }
    }
}

/// Result of applying a chat template
#[derive(Debug, Clone)]
pub struct ChatParams {
    pub format: ChatFormat,
    pub prompt: String,
    pub grammar: Option<String>,
    pub grammar_lazy: bool,
    pub preserved_tokens: Vec<String>,
    pub additional_stops: Vec<String>,
}

/// Chat template handler
pub struct ChatTemplates {
    handle: c_chat_templates_handle,
}

impl ChatTemplates {
    /// Initialize chat templates for a model
    pub fn new(
        model: &LlamaModel,
        chat_template_override: Option<&str>,
        bos_token_override: Option<&str>,
        eos_token_override: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let template_c = chat_template_override
            .map(|s| CString::new(s))
            .transpose()?;
        let bos_c = bos_token_override.map(|s| CString::new(s)).transpose()?;
        let eos_c = eos_token_override.map(|s| CString::new(s)).transpose()?;

        let handle = unsafe {
            c_chat_templates_init(
                model.model.as_ptr(),
                template_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                bos_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                eos_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            )
        };

        if handle.is_null() {
            return Err("Failed to initialize chat templates".into());
        }

        Ok(Self { handle })
    }

    /// Check if the template was explicitly set
    pub fn was_explicit(&self) -> bool {
        unsafe { c_chat_templates_was_explicit(self.handle) }
    }

    /// Get the source of the template
    pub fn source(&self, variant: Option<&str>) -> Option<String> {
        let variant_c = variant.map(|s| CString::new(s).ok()).flatten();

        let source_ptr = unsafe {
            c_chat_templates_source(
                self.handle,
                variant_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            )
        };

        if source_ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(source_ptr) }
                .to_str()
                .ok()
                .map(|s| s.to_string())
        }
    }

    /// Apply chat template to messages
    pub fn apply(
        &self,
        inputs: &ChatTemplateInputs,
    ) -> Result<ChatParams, Box<dyn std::error::Error>> {
        // Convert Rust structures to C structures
        let mut c_messages = Vec::new();
        let mut c_content_parts_data = Vec::new();
        let mut c_tool_calls_data = Vec::new();
        let mut c_strings = Vec::new();

        for msg in &inputs.messages {
            let role = CString::new(msg.role.clone())?;
            let content = CString::new(msg.content.clone())?;
            let reasoning = msg
                .reasoning_content
                .as_ref()
                .map(|s| CString::new(s.clone()))
                .transpose()?;
            let tool_name = msg
                .tool_name
                .as_ref()
                .map(|s| CString::new(s.clone()))
                .transpose()?;
            let tool_call_id = msg
                .tool_call_id
                .as_ref()
                .map(|s| CString::new(s.clone()))
                .transpose()?;

            // Convert content parts
            let mut c_content_parts = Vec::new();
            for part in &msg.content_parts {
                let part_type = CString::new(part.content_type.clone())?;
                let part_text = CString::new(part.text.clone())?;
                c_content_parts.push(c_chat_msg_content_part {
                    type_: part_type.as_ptr() as *mut i8,
                    text: part_text.as_ptr() as *mut i8,
                });
                c_strings.push(part_type);
                c_strings.push(part_text);
            }

            // Convert tool calls
            let mut c_tool_calls = Vec::new();
            for call in &msg.tool_calls {
                let call_name = CString::new(call.name.clone())?;
                let call_args = CString::new(call.arguments.clone())?;
                let call_id = CString::new(call.id.clone())?;
                c_tool_calls.push(c_chat_tool_call {
                    name: call_name.as_ptr() as *mut i8,
                    arguments: call_args.as_ptr() as *mut i8,
                    id: call_id.as_ptr() as *mut i8,
                });
                c_strings.push(call_name);
                c_strings.push(call_args);
                c_strings.push(call_id);
            }

            c_content_parts_data.push(c_content_parts);
            c_tool_calls_data.push(c_tool_calls);

            let c_msg = c_chat_msg {
                role: role.as_ptr() as *mut i8,
                content: content.as_ptr() as *mut i8,
                content_parts: if c_content_parts_data.last().unwrap().is_empty() {
                    ptr::null_mut()
                } else {
                    c_content_parts_data.last_mut().unwrap().as_mut_ptr()
                },
                n_content_parts: c_content_parts_data.last().unwrap().len(),
                tool_calls: if c_tool_calls_data.last().unwrap().is_empty() {
                    ptr::null_mut()
                } else {
                    c_tool_calls_data.last_mut().unwrap().as_mut_ptr()
                },
                n_tool_calls: c_tool_calls_data.last().unwrap().len(),
                reasoning_content: reasoning
                    .as_ref()
                    .map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i8),
                tool_name: tool_name
                    .as_ref()
                    .map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i8),
                tool_call_id: tool_call_id
                    .as_ref()
                    .map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i8),
            };

            c_messages.push(c_msg);
            c_strings.push(role);
            c_strings.push(content);
            if let Some(s) = reasoning {
                c_strings.push(s);
            }
            if let Some(s) = tool_name {
                c_strings.push(s);
            }
            if let Some(s) = tool_call_id {
                c_strings.push(s);
            }
        }

        // Convert tools
        let mut c_tools = Vec::new();
        for tool in &inputs.tools {
            let name = CString::new(tool.name.clone())?;
            let description = CString::new(tool.description.clone())?;
            let parameters = CString::new(tool.parameters.clone())?;
            c_tools.push(c_chat_tool {
                name: name.as_ptr() as *mut i8,
                description: description.as_ptr() as *mut i8,
                parameters: parameters.as_ptr() as *mut i8,
            });
            c_strings.push(name);
            c_strings.push(description);
            c_strings.push(parameters);
        }

        let grammar = inputs
            .grammar
            .as_ref()
            .map(|s| CString::new(s.clone()))
            .transpose()?;
        let json_schema = inputs
            .json_schema
            .as_ref()
            .map(|s| CString::new(s.clone()))
            .transpose()?;

        let c_inputs = c_chat_templates_inputs {
            messages: c_messages.as_ptr() as *mut c_chat_msg,
            n_messages: c_messages.len(),
            grammar: grammar
                .as_ref()
                .map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i8),
            json_schema: json_schema
                .as_ref()
                .map_or(ptr::null_mut(), |s| s.as_ptr() as *mut i8),
            add_generation_prompt: inputs.add_generation_prompt,
            use_jinja: inputs.use_jinja,
            tools: c_tools.as_ptr() as *mut c_chat_tool,
            n_tools: c_tools.len(),
            tool_choice: inputs.tool_choice.into(),
            parallel_tool_calls: inputs.parallel_tool_calls,
            reasoning_format: inputs.reasoning_format.into(),
            enable_thinking: inputs.enable_thinking,
        };

        let c_params = unsafe { c_chat_templates_apply(self.handle, &c_inputs) };

        // Convert result back to Rust
        let format = ChatFormat::from(c_params.format);
        let prompt = if c_params.prompt.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(c_params.prompt) }
                .to_str()
                .unwrap_or("")
                .to_string()
        };

        let grammar = if c_params.grammar.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(c_params.grammar) }
                .to_str()
                .ok()
                .map(|s| s.to_string())
        };

        let mut preserved_tokens = Vec::new();
        if !c_params.preserved_tokens.is_null() {
            for i in 0..c_params.n_preserved_tokens {
                let token_ptr = unsafe { *c_params.preserved_tokens.add(i) };
                if !token_ptr.is_null() {
                    if let Ok(token) = unsafe { CStr::from_ptr(token_ptr) }.to_str() {
                        preserved_tokens.push(token.to_string());
                    }
                }
            }
        }

        let mut additional_stops = Vec::new();
        if !c_params.additional_stops.is_null() {
            for i in 0..c_params.n_additional_stops {
                let stop_ptr = unsafe { *c_params.additional_stops.add(i) };
                if !stop_ptr.is_null() {
                    if let Ok(stop) = unsafe { CStr::from_ptr(stop_ptr) }.to_str() {
                        additional_stops.push(stop.to_string());
                    }
                }
            }
        }

        // Clean up C memory
        let mut c_params_copy = c_params;
        unsafe { c_chat_params_free(&mut c_params_copy) };

        Ok(ChatParams {
            format,
            prompt,
            grammar,
            grammar_lazy: c_params.grammar_lazy,
            preserved_tokens,
            additional_stops,
        })
    }
}

impl Drop for ChatTemplates {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { c_chat_templates_free(self.handle) };
        }
    }
}

/// RAII wrapper that owns a `c_chat_msg` allocated in Rust and frees it with
/// the matching C helper when dropped.  This guarantees the message outlives
/// any downstream pointers (e.g. the diff array) while it is still in scope.
#[derive(Debug)]
struct RawCMsg {
    inner: c_chat_msg,
}

impl RawCMsg {
    fn as_ptr(&self) -> *const c_chat_msg {
        &self.inner as *const c_chat_msg
    }
}

impl Deref for RawCMsg {
    type Target = c_chat_msg;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for RawCMsg {
    fn drop(&mut self) {
        unsafe { c_chat_msg_free(&mut self.inner) };
    }
}

impl ChatMessage {
    /// Compute differences between two chat messages
    pub fn compute_diffs(
        previous: &ChatMessage,
        new: &ChatMessage,
    ) -> Result<ChatMessageDiffs, Box<dyn std::error::Error>> {
        // Convert to C messages and keep them alive for the lifetime of the diff array
        let previous_c = Self::to_c_message(previous)?;
        let new_c = Self::to_c_message(new)?;

        // Call C function using the raw pointers
        let c_diffs = unsafe { c_chat_msg_compute_diffs(previous_c.as_ptr(), new_c.as_ptr()) };

        // Convert back to Rust
        let mut diffs = Vec::new();

        if !c_diffs.diffs.is_null() {
            for i in 0..c_diffs.n_diffs {
                let diff = unsafe { &*c_diffs.diffs.add(i) };

                let reasoning_content_delta = if diff.reasoning_content_delta.is_null() {
                    None
                } else {
                    unsafe { CStr::from_ptr(diff.reasoning_content_delta) }
                        .to_str()
                        .ok()
                        .map(|s| s.to_string())
                };

                let content_delta = if diff.content_delta.is_null() {
                    None
                } else {
                    unsafe { CStr::from_ptr(diff.content_delta) }
                        .to_str()
                        .ok()
                        .map(|s| s.to_string())
                };

                let tool_call_index = if diff.tool_call_index == usize::MAX {
                    None
                } else {
                    Some(diff.tool_call_index)
                };

                let tool_call_delta = if diff.tool_call_delta.name.is_null() {
                    None
                } else {
                    Some(ChatToolCall {
                        name: unsafe { CStr::from_ptr(diff.tool_call_delta.name) }
                            .to_str()
                            .unwrap_or("")
                            .to_string(),
                        arguments: unsafe { CStr::from_ptr(diff.tool_call_delta.arguments) }
                            .to_str()
                            .unwrap_or("")
                            .to_string(),
                        id: unsafe { CStr::from_ptr(diff.tool_call_delta.id) }
                            .to_str()
                            .unwrap_or("")
                            .to_string(),
                    })
                };

                diffs.push(ChatMessageDiff {
                    reasoning_content_delta,
                    content_delta,
                    tool_call_index,
                    tool_call_delta,
                });
            }
        }

        // Drop the temporary C messages *before* freeing the diff array to
        // avoid double-free when the diff array re-uses pointers from those
        // messages.
        drop(previous_c);
        drop(new_c);

        // Clean up C memory for the diff array after the messages are gone.
        let mut c_diffs_copy = c_diffs;
        unsafe { c_chat_msg_diff_array_free(&mut c_diffs_copy) };

        Ok(ChatMessageDiffs { diffs })
    }

    /// Convert Rust ChatMessage to C representation
    fn to_c_message(msg: &ChatMessage) -> Result<RawCMsg, Box<dyn std::error::Error>> {
        // Top-level fields
        let role = CString::new(msg.role.clone())?;
        let content = CString::new(msg.content.clone())?;
        let reasoning = msg
            .reasoning_content
            .as_ref()
            .map(|s| CString::new(s.clone()))
            .transpose()?;
        let tool_name = msg
            .tool_name
            .as_ref()
            .map(|s| CString::new(s.clone()))
            .transpose()?;
        let tool_call_id = msg
            .tool_call_id
            .as_ref()
            .map(|s| CString::new(s.clone()))
            .transpose()?;

        // Content parts
        let c_content_parts = if msg.content_parts.is_empty() {
            ptr::null_mut()
        } else {
            let parts_ptr = unsafe {
                libc::malloc(
                    std::mem::size_of::<c_chat_msg_content_part>() * msg.content_parts.len(),
                ) as *mut c_chat_msg_content_part
            };
            if parts_ptr.is_null() {
                return Err("malloc failed for content parts".into());
            }
            for (i, part) in msg.content_parts.iter().enumerate() {
                unsafe {
                    let c_part = parts_ptr.add(i);
                    (*c_part).type_ = CString::new(part.content_type.clone())?.into_raw();
                    (*c_part).text = CString::new(part.text.clone())?.into_raw();
                }
            }
            parts_ptr
        };

        // Tool calls
        let c_tool_calls = if msg.tool_calls.is_empty() {
            ptr::null_mut()
        } else {
            let calls_ptr = unsafe {
                libc::malloc(std::mem::size_of::<c_chat_tool_call>() * msg.tool_calls.len())
                    as *mut c_chat_tool_call
            };
            if calls_ptr.is_null() {
                // Free already allocated memory before returning an error
                let mut temp_msg_to_free = c_chat_msg {
                    role: role.into_raw(),
                    content: content.into_raw(),
                    reasoning_content: reasoning.map_or(ptr::null_mut(), |s| s.into_raw()),
                    tool_name: tool_name.map_or(ptr::null_mut(), |s| s.into_raw()),
                    tool_call_id: tool_call_id.map_or(ptr::null_mut(), |s| s.into_raw()),
                    content_parts: c_content_parts,
                    n_content_parts: msg.content_parts.len(),
                    tool_calls: ptr::null_mut(),
                    n_tool_calls: 0,
                };
                unsafe {
                    c_chat_msg_free(&mut temp_msg_to_free);
                }
                return Err("malloc failed for tool calls".into());
            }
            for (i, call) in msg.tool_calls.iter().enumerate() {
                unsafe {
                    let c_call = calls_ptr.add(i);
                    (*c_call).name = CString::new(call.name.clone())?.into_raw();
                    (*c_call).arguments = CString::new(call.arguments.clone())?.into_raw();
                    (*c_call).id = CString::new(call.id.clone())?.into_raw();
                }
            }
            calls_ptr
        };

        let c_msg = c_chat_msg {
            role: role.into_raw(),
            content: content.into_raw(),
            content_parts: c_content_parts,
            n_content_parts: msg.content_parts.len(),
            tool_calls: c_tool_calls,
            n_tool_calls: msg.tool_calls.len(),
            reasoning_content: reasoning.map_or(ptr::null_mut(), |s| s.into_raw()),
            tool_name: tool_name.map_or(ptr::null_mut(), |s| s.into_raw()),
            tool_call_id: tool_call_id.map_or(ptr::null_mut(), |s| s.into_raw()),
        };

        Ok(RawCMsg { inner: c_msg })
    }
}

impl Default for ChatMessage {
    fn default() -> Self {
        Self {
            role: String::new(),
            content: String::new(),
            content_parts: Vec::new(),
            tool_calls: Vec::new(),
            reasoning_content: None,
            tool_name: None,
            tool_call_id: None,
        }
    }
}

/// Parse a chat response from the model
pub fn parse_chat_response(
    input: &str,
    is_partial: bool,
    syntax: &ChatSyntax,
) -> Result<ChatMessage, Box<dyn std::error::Error>> {
    let input_c = CString::new(input)?;

    let c_syntax = c_chat_syntax {
        format: match syntax.format {
            ChatFormat::ContentOnly => C_CHAT_FORMAT_CONTENT_ONLY,
            ChatFormat::Generic => C_CHAT_FORMAT_GENERIC,
            ChatFormat::MistralNemo => C_CHAT_FORMAT_MISTRAL_NEMO,
            ChatFormat::Llama3X => C_CHAT_FORMAT_LLAMA_3_X,
            ChatFormat::Llama3XWithBuiltinTools => C_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
            ChatFormat::DeepSeekR1 => C_CHAT_FORMAT_DEEPSEEK_R1,
            ChatFormat::FirefunctionV2 => C_CHAT_FORMAT_FIREFUNCTION_V2,
            ChatFormat::FunctionaryV32 => C_CHAT_FORMAT_FUNCTIONARY_V3_2,
            ChatFormat::FunctionaryV31Llama31 => C_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
            ChatFormat::Hermes2Pro => C_CHAT_FORMAT_HERMES_2_PRO,
            ChatFormat::CommandR7B => C_CHAT_FORMAT_COMMAND_R7B,
        },
        reasoning_format: syntax.reasoning_format.into(),
        reasoning_in_content: syntax.reasoning_in_content,
        thinking_forced_open: syntax.thinking_forced_open,
        parse_tool_calls: syntax.parse_tool_calls,
    };

    let c_msg = unsafe { c_chat_parse(input_c.as_ptr(), is_partial, &c_syntax) };

    // Convert C message to Rust
    let role = if c_msg.role.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(c_msg.role) }
            .to_str()
            .unwrap_or("")
            .to_string()
    };

    let content = if c_msg.content.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(c_msg.content) }
            .to_str()
            .unwrap_or("")
            .to_string()
    };

    let reasoning_content = if c_msg.reasoning_content.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(c_msg.reasoning_content) }
            .to_str()
            .ok()
            .map(|s| s.to_string())
    };

    let tool_name = if c_msg.tool_name.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(c_msg.tool_name) }
            .to_str()
            .ok()
            .map(|s| s.to_string())
    };

    let tool_call_id = if c_msg.tool_call_id.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(c_msg.tool_call_id) }
            .to_str()
            .ok()
            .map(|s| s.to_string())
    };

    // Convert content parts
    let mut content_parts = Vec::new();
    if !c_msg.content_parts.is_null() {
        for i in 0..c_msg.n_content_parts {
            let part = unsafe { c_msg.content_parts.add(i) };
            let part_type = if unsafe { (*part).type_ }.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr((*part).type_) }
                    .to_str()
                    .unwrap_or("")
                    .to_string()
            };
            let text = if unsafe { (*part).text }.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr((*part).text) }
                    .to_str()
                    .unwrap_or("")
                    .to_string()
            };
            content_parts.push(ChatMessageContentPart {
                content_type: part_type,
                text,
            });
        }
    }

    // Convert tool calls
    let mut tool_calls = Vec::new();
    if !c_msg.tool_calls.is_null() {
        for i in 0..c_msg.n_tool_calls {
            let call = unsafe { c_msg.tool_calls.add(i) };
            let name = if unsafe { (*call).name }.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr((*call).name) }
                    .to_str()
                    .unwrap_or("")
                    .to_string()
            };
            let arguments = if unsafe { (*call).arguments }.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr((*call).arguments) }
                    .to_str()
                    .unwrap_or("")
                    .to_string()
            };
            let id = if unsafe { (*call).id }.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr((*call).id) }
                    .to_str()
                    .unwrap_or("")
                    .to_string()
            };
            tool_calls.push(ChatToolCall {
                name,
                arguments,
                id,
            });
        }
    }

    // Clean up C memory
    let mut c_msg_copy = c_msg;
    unsafe { c_chat_msg_free(&mut c_msg_copy) };

    Ok(ChatMessage {
        role,
        content,
        content_parts,
        tool_calls,
        reasoning_content,
        tool_name,
        tool_call_id,
    })
}

/// Get the name of a chat format
pub fn chat_format_name(format: ChatFormat) -> String {
    let c_format = match format {
        ChatFormat::ContentOnly => C_CHAT_FORMAT_CONTENT_ONLY,
        ChatFormat::Generic => C_CHAT_FORMAT_GENERIC,
        ChatFormat::MistralNemo => C_CHAT_FORMAT_MISTRAL_NEMO,
        ChatFormat::Llama3X => C_CHAT_FORMAT_LLAMA_3_X,
        ChatFormat::Llama3XWithBuiltinTools => C_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
        ChatFormat::DeepSeekR1 => C_CHAT_FORMAT_DEEPSEEK_R1,
        ChatFormat::FirefunctionV2 => C_CHAT_FORMAT_FIREFUNCTION_V2,
        ChatFormat::FunctionaryV32 => C_CHAT_FORMAT_FUNCTIONARY_V3_2,
        ChatFormat::FunctionaryV31Llama31 => C_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
        ChatFormat::Hermes2Pro => C_CHAT_FORMAT_HERMES_2_PRO,
        ChatFormat::CommandR7B => C_CHAT_FORMAT_COMMAND_R7B,
    };

    let name_ptr = unsafe { c_chat_format_name(c_format) };
    if name_ptr.is_null() {
        "Unknown".to_string()
    } else {
        unsafe { CStr::from_ptr(name_ptr) }
            .to_str()
            .unwrap_or("Unknown")
            .to_string()
    }
}
