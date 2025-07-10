#ifndef CHAT_WRAPPER_H
#define CHAT_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Forward declarations
    struct llama_model;

    // C-compatible structures
    typedef struct
    {
        char *name;
        char *arguments;
        char *id;
    } c_chat_tool_call;

    typedef struct
    {
        char *type;
        char *text;
    } c_chat_msg_content_part;

    typedef struct
    {
        char *role;
        char *content;
        c_chat_msg_content_part *content_parts;
        size_t n_content_parts;
        c_chat_tool_call *tool_calls;
        size_t n_tool_calls;
        char *reasoning_content;
        char *tool_name;
        char *tool_call_id;
    } c_chat_msg;

    typedef struct
    {
        char *name;
        char *description;
        char *parameters;
    } c_chat_tool;

    typedef enum
    {
        C_CHAT_TOOL_CHOICE_AUTO = 0,
        C_CHAT_TOOL_CHOICE_REQUIRED = 1,
        C_CHAT_TOOL_CHOICE_NONE = 2,
    } c_chat_tool_choice;

    typedef enum
    {
        C_CHAT_FORMAT_CONTENT_ONLY = 0,
        C_CHAT_FORMAT_GENERIC = 1,
        C_CHAT_FORMAT_MISTRAL_NEMO = 2,
        C_CHAT_FORMAT_LLAMA_3_X = 3,
        C_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS = 4,
        C_CHAT_FORMAT_DEEPSEEK_R1 = 5,
        C_CHAT_FORMAT_FIREFUNCTION_V2 = 6,
        C_CHAT_FORMAT_FUNCTIONARY_V3_2 = 7,
        C_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1 = 8,
        C_CHAT_FORMAT_HERMES_2_PRO = 9,
        C_CHAT_FORMAT_COMMAND_R7B = 10,
    } c_chat_format;

    typedef enum
    {
        C_REASONING_FORMAT_NONE = 0,
        C_REASONING_FORMAT_DEEPSEEK = 1,
        C_REASONING_FORMAT_DEEPSEEK_LEGACY = 2,
    } c_reasoning_format;

    typedef struct
    {
        c_chat_format format;
        c_reasoning_format reasoning_format;
        bool reasoning_in_content;
        bool thinking_forced_open;
        bool parse_tool_calls;
    } c_chat_syntax;

    typedef struct
    {
        c_chat_msg *messages;
        size_t n_messages;
        char *grammar;
        char *json_schema;
        bool add_generation_prompt;
        bool use_jinja;
        c_chat_tool *tools;
        size_t n_tools;
        c_chat_tool_choice tool_choice;
        bool parallel_tool_calls;
        c_reasoning_format reasoning_format;
        bool enable_thinking;
    } c_chat_templates_inputs;

    typedef struct
    {
        c_chat_format format;
        char *prompt;
        char *grammar;
        bool grammar_lazy;
        char **preserved_tokens;
        size_t n_preserved_tokens;
        char **additional_stops;
        size_t n_additional_stops;
    } c_chat_params;

    // Opaque handle for chat templates
    typedef void *c_chat_templates_handle;

    // Function declarations
    c_chat_templates_handle c_chat_templates_init(
        const struct llama_model *model,
        const char *chat_template_override,
        const char *bos_token_override,
        const char *eos_token_override);

    void c_chat_templates_free(c_chat_templates_handle templates);

    bool c_chat_templates_was_explicit(c_chat_templates_handle templates);

    const char *c_chat_templates_source(c_chat_templates_handle templates, const char *variant);

    c_chat_params c_chat_templates_apply(
        c_chat_templates_handle templates,
        const c_chat_templates_inputs *inputs);

    const char *c_chat_format_name(c_chat_format format);

    c_chat_msg c_chat_parse(const char *input, bool is_partial, const c_chat_syntax *syntax);

    bool c_chat_verify_template(const char *tmpl, bool use_jinja);

    // Memory management helpers
    void c_chat_params_free(c_chat_params *params);
    void c_chat_msg_free(c_chat_msg *msg);
    void c_chat_msg_content_part_free(c_chat_msg_content_part *part);
    void c_chat_tool_call_free(c_chat_tool_call *tool_call);

    // Diff-related structures
    typedef struct
    {
        char *reasoning_content_delta;
        char *content_delta;
        size_t tool_call_index; // SIZE_MAX if not a tool call diff
        c_chat_tool_call tool_call_delta;
    } c_chat_msg_diff;

    typedef struct
    {
        c_chat_msg_diff *diffs;
        size_t n_diffs;
    } c_chat_msg_diff_array;

    // Diff-related functions
    c_chat_msg_diff_array c_chat_msg_compute_diffs(
        const c_chat_msg *previous_msg,
        const c_chat_msg *new_msg);

    void c_chat_msg_diff_array_free(c_chat_msg_diff_array *diffs);
    void c_chat_msg_diff_free(c_chat_msg_diff *diff);

#ifdef __cplusplus
}
#endif

#endif // CHAT_WRAPPER_H