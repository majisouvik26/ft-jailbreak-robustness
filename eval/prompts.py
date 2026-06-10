"""Chat prompt formatting for inference (post-attack generation check)."""


def get_model_family(model_key: str) -> str:
    key = model_key.lower()
    if "llama-3.1" in key or "llama3.1" in key:
        return "llama3.1"
    if "llama-3.2" in key or "llama3.2" in key:
        return "llama3.2"
    if "qwen" in key:
        return "qwen3"
    if "phi" in key:
        return "phi4"
    if "gemma" in key:
        return "gemma3"
    return "llama3.1"


def format_prompt(user_message: str, model_family: str) -> str:
    """Format a user message for vLLM/HF generation (matches eval_modal_phi)."""
    system_prompt = "You are a helpful assistant."

    if model_family == "qwen3":
        return (
            f"<|im_start|>system\n{system_prompt}\n"
            f"<|im_start|>user\n{user_message}\n"
            f"<|im_start|>assistant\n"
        )
    if model_family in ("llama3.1", "llama3.2", "llama3"):
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if model_family == "phi4":
        return (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_message}<|end|>\n"
            f"<|assistant|>\n"
        )
    if model_family == "gemma3":
        return (
            f"<start_of_turn>user\n{user_message}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    return (
        f"<|im_start|>system\n{system_prompt}\n"
        f"<|im_start|>user\n{user_message}\n"
        f"<|im_start|>assistant\n"
    )


def build_attacked_user_message(goal: str, control: str) -> str:
    """User message with GCG-optimized adversarial suffix appended."""
    separator = " " if goal else ""
    return f"{goal}{separator}{control}"
