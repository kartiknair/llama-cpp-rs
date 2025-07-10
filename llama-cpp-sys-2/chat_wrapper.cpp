#include "chat_wrapper.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/chat.h"
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// Note: common_chat_msg_diff is now defined in chat.h

// Helper functions for string conversion
static char *strdup_safe(const std::string &str)
{
    if (str.empty())
        return nullptr;
    char *result = (char *)malloc(str.length() + 1);
    if (result)
    {
        strcpy(result, str.c_str());
    }
    return result;
}

static char **string_vector_to_c_array(const std::vector<std::string> &vec, size_t *out_size)
{
    *out_size = vec.size();
    if (vec.empty())
        return nullptr;

    char **result = (char **)malloc(vec.size() * sizeof(char *));
    if (!result)
        return nullptr;

    for (size_t i = 0; i < vec.size(); ++i)
    {
        result[i] = strdup_safe(vec[i]);
    }
    return result;
}

// Convert C structures to C++ structures
static common_chat_tool_call cpp_tool_call_from_c(const c_chat_tool_call &c_call)
{
    common_chat_tool_call cpp_call;
    if (c_call.name)
        cpp_call.name = c_call.name;
    if (c_call.arguments)
        cpp_call.arguments = c_call.arguments;
    if (c_call.id)
        cpp_call.id = c_call.id;
    return cpp_call;
}

static common_chat_msg_content_part cpp_content_part_from_c(const c_chat_msg_content_part &c_part)
{
    common_chat_msg_content_part cpp_part;
    if (c_part.type)
        cpp_part.type = c_part.type;
    if (c_part.text)
        cpp_part.text = c_part.text;
    return cpp_part;
}

static common_chat_msg cpp_msg_from_c(const c_chat_msg &c_msg)
{
    common_chat_msg cpp_msg;
    if (c_msg.role)
        cpp_msg.role = c_msg.role;
    if (c_msg.content)
        cpp_msg.content = c_msg.content;
    if (c_msg.reasoning_content)
        cpp_msg.reasoning_content = c_msg.reasoning_content;
    if (c_msg.tool_name)
        cpp_msg.tool_name = c_msg.tool_name;
    if (c_msg.tool_call_id)
        cpp_msg.tool_call_id = c_msg.tool_call_id;

    // Convert content parts
    for (size_t i = 0; i < c_msg.n_content_parts; ++i)
    {
        cpp_msg.content_parts.push_back(cpp_content_part_from_c(c_msg.content_parts[i]));
    }

    // Convert tool calls
    for (size_t i = 0; i < c_msg.n_tool_calls; ++i)
    {
        cpp_msg.tool_calls.push_back(cpp_tool_call_from_c(c_msg.tool_calls[i]));
    }

    return cpp_msg;
}

static common_chat_tool cpp_tool_from_c(const c_chat_tool &c_tool)
{
    common_chat_tool cpp_tool;
    if (c_tool.name)
        cpp_tool.name = c_tool.name;
    if (c_tool.description)
        cpp_tool.description = c_tool.description;
    if (c_tool.parameters)
        cpp_tool.parameters = c_tool.parameters;
    return cpp_tool;
}

static common_chat_tool_choice cpp_tool_choice_from_c(c_chat_tool_choice c_choice)
{
    switch (c_choice)
    {
    case C_CHAT_TOOL_CHOICE_AUTO:
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    case C_CHAT_TOOL_CHOICE_REQUIRED:
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    case C_CHAT_TOOL_CHOICE_NONE:
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    default:
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
}

static common_chat_format cpp_format_from_c(c_chat_format c_format)
{
    return static_cast<common_chat_format>(c_format);
}

static common_reasoning_format cpp_reasoning_format_from_c(c_reasoning_format c_format)
{
    switch (c_format)
    {
    case C_REASONING_FORMAT_NONE:
        return COMMON_REASONING_FORMAT_NONE;
    case C_REASONING_FORMAT_DEEPSEEK:
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    case C_REASONING_FORMAT_DEEPSEEK_LEGACY:
        return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    default:
        return COMMON_REASONING_FORMAT_NONE;
    }
}

static common_chat_syntax cpp_syntax_from_c(const c_chat_syntax &c_syntax)
{
    common_chat_syntax cpp_syntax;
    cpp_syntax.format = cpp_format_from_c(c_syntax.format);
    cpp_syntax.reasoning_format = cpp_reasoning_format_from_c(c_syntax.reasoning_format);
    cpp_syntax.reasoning_in_content = c_syntax.reasoning_in_content;
    cpp_syntax.thinking_forced_open = c_syntax.thinking_forced_open;
    cpp_syntax.parse_tool_calls = c_syntax.parse_tool_calls;
    return cpp_syntax;
}

// Convert C++ structures to C structures
static c_chat_tool_call c_tool_call_from_cpp(const common_chat_tool_call &cpp_call)
{
    c_chat_tool_call c_call;
    c_call.name = strdup_safe(cpp_call.name);
    c_call.arguments = strdup_safe(cpp_call.arguments);
    c_call.id = strdup_safe(cpp_call.id);
    return c_call;
}

static c_chat_msg_content_part c_content_part_from_cpp(const common_chat_msg_content_part &cpp_part)
{
    c_chat_msg_content_part c_part;
    c_part.type = strdup_safe(cpp_part.type);
    c_part.text = strdup_safe(cpp_part.text);
    return c_part;
}

static c_chat_msg c_msg_from_cpp(const common_chat_msg &cpp_msg)
{
    c_chat_msg c_msg = {};
    c_msg.role = strdup_safe(cpp_msg.role);
    c_msg.content = strdup_safe(cpp_msg.content);
    c_msg.reasoning_content = strdup_safe(cpp_msg.reasoning_content);
    c_msg.tool_name = strdup_safe(cpp_msg.tool_name);
    c_msg.tool_call_id = strdup_safe(cpp_msg.tool_call_id);

    // Convert content parts
    c_msg.n_content_parts = cpp_msg.content_parts.size();
    if (c_msg.n_content_parts > 0)
    {
        c_msg.content_parts = (c_chat_msg_content_part *)malloc(c_msg.n_content_parts * sizeof(c_chat_msg_content_part));
        for (size_t i = 0; i < c_msg.n_content_parts; ++i)
        {
            c_msg.content_parts[i] = c_content_part_from_cpp(cpp_msg.content_parts[i]);
        }
    }

    // Convert tool calls
    c_msg.n_tool_calls = cpp_msg.tool_calls.size();
    if (c_msg.n_tool_calls > 0)
    {
        c_msg.tool_calls = (c_chat_tool_call *)malloc(c_msg.n_tool_calls * sizeof(c_chat_tool_call));
        for (size_t i = 0; i < c_msg.n_tool_calls; ++i)
        {
            c_msg.tool_calls[i] = c_tool_call_from_cpp(cpp_msg.tool_calls[i]);
        }
    }

    return c_msg;
}

static c_chat_format c_format_from_cpp(common_chat_format cpp_format)
{
    return static_cast<c_chat_format>(cpp_format);
}

// Wrapper function implementations
extern "C"
{

    c_chat_templates_handle c_chat_templates_init(
        const struct llama_model *model,
        const char *chat_template_override,
        const char *bos_token_override,
        const char *eos_token_override)
    {
        std::string template_str = chat_template_override ? chat_template_override : "";
        std::string bos_str = bos_token_override ? bos_token_override : "";
        std::string eos_str = eos_token_override ? eos_token_override : "";

        auto templates = common_chat_templates_init(model, template_str, bos_str, eos_str);
        return templates.release();
    }

    void c_chat_templates_free(c_chat_templates_handle templates)
    {
        if (templates)
        {
            common_chat_templates_free(static_cast<common_chat_templates *>(templates));
        }
    }

    bool c_chat_templates_was_explicit(c_chat_templates_handle templates)
    {
        if (!templates)
            return false;
        return common_chat_templates_was_explicit(static_cast<const common_chat_templates *>(templates));
    }

    const char *c_chat_templates_source(c_chat_templates_handle templates, const char *variant)
    {
        if (!templates)
            return nullptr;
        static std::string result; // Keep result alive
        result = common_chat_templates_source(static_cast<const common_chat_templates *>(templates), variant);
        return result.c_str();
    }

    c_chat_params c_chat_templates_apply(
        c_chat_templates_handle templates,
        const c_chat_templates_inputs *inputs)
    {
        c_chat_params result = {};

        if (!templates || !inputs)
        {
            return result;
        }

        // Convert C inputs to C++ inputs
        common_chat_templates_inputs cpp_inputs;

        // Convert messages
        for (size_t i = 0; i < inputs->n_messages; ++i)
        {
            cpp_inputs.messages.push_back(cpp_msg_from_c(inputs->messages[i]));
        }

        // Convert tools
        for (size_t i = 0; i < inputs->n_tools; ++i)
        {
            cpp_inputs.tools.push_back(cpp_tool_from_c(inputs->tools[i]));
        }

        // Set other fields
        if (inputs->grammar)
            cpp_inputs.grammar = inputs->grammar;
        if (inputs->json_schema)
            cpp_inputs.json_schema = inputs->json_schema;
        cpp_inputs.add_generation_prompt = inputs->add_generation_prompt;
        cpp_inputs.use_jinja = inputs->use_jinja;
        cpp_inputs.tool_choice = cpp_tool_choice_from_c(inputs->tool_choice);
        cpp_inputs.parallel_tool_calls = inputs->parallel_tool_calls;
        cpp_inputs.reasoning_format = cpp_reasoning_format_from_c(inputs->reasoning_format);
        cpp_inputs.enable_thinking = inputs->enable_thinking;

        // Apply templates
        auto cpp_result = common_chat_templates_apply(
            static_cast<const common_chat_templates *>(templates),
            cpp_inputs);

        // Convert result back to C
        result.format = c_format_from_cpp(cpp_result.format);
        result.prompt = strdup_safe(cpp_result.prompt);
        result.grammar = strdup_safe(cpp_result.grammar);
        result.grammar_lazy = cpp_result.grammar_lazy;
        result.preserved_tokens = string_vector_to_c_array(cpp_result.preserved_tokens, &result.n_preserved_tokens);
        result.additional_stops = string_vector_to_c_array(cpp_result.additional_stops, &result.n_additional_stops);

        return result;
    }

    const char *c_chat_format_name(c_chat_format format)
    {
        static std::string result; // Keep result alive
        result = common_chat_format_name(cpp_format_from_c(format));
        return result.c_str();
    }

    c_chat_msg c_chat_parse(const char *input, bool is_partial, const c_chat_syntax *syntax)
    {
        c_chat_msg result = {};

        if (!input || !syntax)
            return result;

        auto cpp_syntax = cpp_syntax_from_c(*syntax);
        auto cpp_result = common_chat_parse(input, is_partial, cpp_syntax);
        return c_msg_from_cpp(cpp_result);
    }

    bool c_chat_verify_template(const char *tmpl, bool use_jinja)
    {
        if (!tmpl)
            return false;
        return common_chat_verify_template(tmpl, use_jinja);
    }

    // Memory management helpers
    void c_chat_params_free(c_chat_params *params)
    {
        if (!params)
            return;

        free(params->prompt);
        free(params->grammar);

        if (params->preserved_tokens)
        {
            for (size_t i = 0; i < params->n_preserved_tokens; ++i)
            {
                free(params->preserved_tokens[i]);
            }
            free(params->preserved_tokens);
        }

        if (params->additional_stops)
        {
            for (size_t i = 0; i < params->n_additional_stops; ++i)
            {
                free(params->additional_stops[i]);
            }
            free(params->additional_stops);
        }

        memset(params, 0, sizeof(*params));
    }

    void c_chat_msg_free(c_chat_msg *msg)
    {
        if (!msg)
            return;

        free(msg->role);
        free(msg->content);
        free(msg->reasoning_content);
        free(msg->tool_name);
        free(msg->tool_call_id);

        if (msg->content_parts)
        {
            for (size_t i = 0; i < msg->n_content_parts; ++i)
            {
                c_chat_msg_content_part_free(&msg->content_parts[i]);
            }
            free(msg->content_parts);
        }

        if (msg->tool_calls)
        {
            for (size_t i = 0; i < msg->n_tool_calls; ++i)
            {
                c_chat_tool_call_free(&msg->tool_calls[i]);
            }
            free(msg->tool_calls);
        }

        memset(msg, 0, sizeof(*msg));
    }

    void c_chat_msg_content_part_free(c_chat_msg_content_part *part)
    {
        if (!part)
            return;
        free(part->type);
        free(part->text);
        memset(part, 0, sizeof(*part));
    }

    void c_chat_tool_call_free(c_chat_tool_call *tool_call)
    {
        if (!tool_call)
            return;
        free(tool_call->name);
        free(tool_call->arguments);
        free(tool_call->id);
        memset(tool_call, 0, sizeof(*tool_call));
    }

    // Diff-related implementations

    // Helper function to convert common_chat_msg_diff to C struct
    static c_chat_msg_diff c_diff_from_cpp(const common_chat_msg_diff &cpp_diff)
    {
        c_chat_msg_diff c_diff = {};

        // Copy reasoning content delta
        c_diff.reasoning_content_delta = strdup_safe(cpp_diff.reasoning_content_delta);

        // Copy content delta
        c_diff.content_delta = strdup_safe(cpp_diff.content_delta);

        // Copy tool call index
        c_diff.tool_call_index = cpp_diff.tool_call_index;

        // Copy tool call delta
        c_diff.tool_call_delta = c_tool_call_from_cpp(cpp_diff.tool_call_delta);

        return c_diff;
    }

    c_chat_msg_diff_array c_chat_msg_compute_diffs(
        const c_chat_msg *previous_msg,
        const c_chat_msg *new_msg)
    {
        c_chat_msg_diff_array result = {};

        if (!previous_msg || !new_msg)
        {
            return result;
        }

        try
        {
            // Convert C structs to C++ objects
            common_chat_msg cpp_previous = cpp_msg_from_c(*previous_msg);
            common_chat_msg cpp_new = cpp_msg_from_c(*new_msg);

            // Compute diffs
            auto cpp_diffs = common_chat_msg_diff::compute_diffs(cpp_previous, cpp_new);

            // Convert results to C
            result.n_diffs = cpp_diffs.size();
            if (result.n_diffs > 0)
            {
                result.diffs = (c_chat_msg_diff *)malloc(sizeof(c_chat_msg_diff) * result.n_diffs);
                if (result.diffs)
                {
                    for (size_t i = 0; i < result.n_diffs; i++)
                    {
                        result.diffs[i] = c_diff_from_cpp(cpp_diffs[i]);
                    }
                }
                else
                {
                    result.n_diffs = 0;
                }
            }
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "Error computing diffs: %s\n", e.what());
            result.n_diffs = 0;
            result.diffs = nullptr;
        }

        return result;
    }

    void c_chat_msg_diff_free(c_chat_msg_diff *diff)
    {
        if (!diff)
            return;
        free(diff->reasoning_content_delta);
        free(diff->content_delta);
        c_chat_tool_call_free(&diff->tool_call_delta);
        memset(diff, 0, sizeof(*diff));
    }

    void c_chat_msg_diff_array_free(c_chat_msg_diff_array *diffs)
    {
        if (diffs && diffs->diffs)
        {
            for (size_t i = 0; i < diffs->n_diffs; i++)
            {
                c_chat_msg_diff_free(&diffs->diffs[i]);
            }
            free(diffs->diffs);
            diffs->diffs = nullptr;
            diffs->n_diffs = 0;
        }
    }

} // extern "C"