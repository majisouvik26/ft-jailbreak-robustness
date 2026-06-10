"""GCG hyperparameters aligned with llm-attacks defaults."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GCGConfig:
    """Hyperparameters for llm-attacks GCG (see experiments/configs/template.py)."""

    n_steps: int = 500
    batch_size: int = 512
    topk: int = 256
    temp: float = 1.0
    target_weight: float = 1.0
    control_weight: float = 0.0
    control_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    allow_non_ascii: bool = False
    filter_cand: bool = True
    stop_on_success: bool = True
    test_steps: int = 50
    verbose: bool = True
    anneal: bool = True
    incr_control: bool = True
    device: str = "cuda:0"
    max_samples: int = 100
    data_offset: int = 0
    output_dir: str = "results/gcg"
    model_path_override: Optional[str] = field(default=None)
