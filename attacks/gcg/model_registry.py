"""Target model registry for GCG attacks."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    conv_template: str
    family: str
    description: str


MODELS: Dict[str, ModelSpec] = {
    "llama-3.1-8b-it": ModelSpec(
        key="llama-3.1-8b-it",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        conv_template="llama-3",
        family="llama3.1",
        description="Meta Llama 3.1 8B Instruct",
    ),
    "llama-3.2-3b-it": ModelSpec(
        key="llama-3.2-3b-it",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        conv_template="llama-3",
        family="llama3.2",
        description="Meta Llama 3.2 3B Instruct",
    ),
    "qwen3-8b-it": ModelSpec(
        key="qwen3-8b-it",
        model_id="Qwen/Qwen3-8B",
        conv_template="qwen-chatml",
        family="qwen3",
        description="Qwen3 8B instruct (chat template, non-thinking)",
    ),
    "qwen3-4b-it": ModelSpec(
        key="qwen3-4b-it",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        conv_template="qwen-chatml",
        family="qwen3",
        description="Qwen3 4B Instruct 2507",
    ),
    "phi-4-mini-it": ModelSpec(
        key="phi-4-mini-it",
        model_id="microsoft/Phi-4-mini-instruct",
        conv_template="phi-chat",
        family="phi4",
        description="Microsoft Phi-4 mini instruct",
    ),
    "gemma-3-4b-it": ModelSpec(
        key="gemma-3-4b-it",
        model_id="google/gemma-3-4b-it",
        conv_template="gemma-chat",
        family="gemma3",
        description="Google Gemma 3 4B instruct",
    ),
}


def list_model_keys() -> List[str]:
    return list(MODELS.keys())


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(
            f"Unknown model key '{key}'. Available: {', '.join(list_model_keys())}"
        )
    return MODELS[key]


def resolve_model_keys(keys: Optional[List[str]] = None) -> List[str]:
    if not keys:
        return list_model_keys()
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}")
    return keys
