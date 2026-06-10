"""FastChat conversation templates for modern instruct models.

llm-attacks uses FastChat templates for prompt slicing during GCG. We register
custom templates that mirror each model family's chat format, and route them
through the llama-2 incremental slice path (see vendor patch).
"""

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

_REGISTERED = False


def register_attack_templates() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    # Llama 3.1 / 3.2 instruct format
    register_conv_template(
        Conversation(
            name="llama-3",
            system_message="You are a helpful assistant.",
            system_template=(
                "<|begin_of_text|><|start_header_id|>system"
                "<|end_header_id|>\n\n{system_message}<|eot_id|>"
            ),
            roles=(
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="<|eot_id|>",
            sep2="<|eot_id|>",
            stop_token_ids=[128001, 128009],
        )
    )

    # Qwen ChatML (Qwen2/3 instruct)
    register_conv_template(
        Conversation(
            name="qwen-chatml",
            system_template="<|im_start|>system\n{system_message}\n",
            roles=("user", "assistant"),
            sep_style=SeparatorStyle.CHATML,
            sep="\n",
            stop_str="",
        )
    )

    # Phi-3 / Phi-4 mini instruct
    register_conv_template(
        Conversation(
            name="phi-chat",
            system_template="<|system|>\n{system_message}<|end|>\n",
            roles=("user", "assistant"),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="<|end|>\n",
            sep2="<|end|>\n",
            stop_token_ids=[32007],
        )
    )

    # Gemma 3 instruct (text-only path)
    gemma_sep = getattr(SeparatorStyle, "GEMMA", SeparatorStyle.NO_COLON_SINGLE)
    register_conv_template(
        Conversation(
            name="gemma-chat",
            system_message="You are a helpful assistant.",
            roles=("user", "model"),
            sep_style=gemma_sep,
            sep="<start_of_turn>",
            sep2="<end_of_turn>\n",
        )
    )

    _REGISTERED = True
