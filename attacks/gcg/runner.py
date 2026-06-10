"""
GCG attack runner using vendored llm-attacks.

Wraps IndividualPromptAttack from https://github.com/llm-attacks/llm-attacks
with support for modern instruct models (Llama 3.x, Qwen3, Phi-4, Gemma 3).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from ml_collections import config_dict

from attacks.gcg.config import GCGConfig
from attacks.gcg.model_registry import ModelSpec, get_model_spec, resolve_model_keys
from attacks.gcg.setup_vendor import ensure_llm_attacks_path
from attacks.gcg.templates import register_attack_templates
from data.advbench import goals_and_targets, load_advbench

# Must register templates and vendor path before importing llm_attacks.
ensure_llm_attacks_path()
register_attack_templates()

import llm_attacks.gcg as gcg  # noqa: E402
from llm_attacks import get_workers  # noqa: E402


def _randomize_targets(targets: List[str]) -> List[str]:
    """Match llm-attacks experiments/main.py target augmentation."""
    process_fn = lambda s: s.replace("Sure, h", "H")
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    return [
        process_fn(t) if random.random() < 0.5 else process_fn2(t) for t in targets
    ]


def _build_params(
    spec: ModelSpec,
    cfg: GCGConfig,
    train_data_path: Path,
) -> Any:
    params = config_dict.ConfigDict()
    params.transfer = False
    params.attack = "gcg"

    model_path = cfg.model_path_override or spec.model_id
    params.model_paths = [model_path]
    params.tokenizer_paths = [model_path]
    params.conversation_templates = [spec.conv_template]
    params.tokenizer_kwargs = [{}]
    params.model_kwargs = {
        "low_cpu_mem_usage": True,
        "use_cache": False,
        "torch_dtype": torch.bfloat16,
    }
    params.devices = [cfg.device]
    params.num_train_models = 1

    params.train_data = str(train_data_path)
    params.test_data = ""
    params.n_train_data = cfg.max_samples
    params.n_test_data = 0
    params.data_offset = cfg.data_offset

    params.control_init = cfg.control_init
    params.n_steps = cfg.n_steps
    params.test_steps = cfg.test_steps
    params.batch_size = cfg.batch_size
    params.topk = cfg.topk
    params.temp = cfg.temp
    params.target_weight = cfg.target_weight
    params.control_weight = cfg.control_weight
    params.allow_non_ascii = cfg.allow_non_ascii
    params.filter_cand = cfg.filter_cand
    params.stop_on_success = cfg.stop_on_success
    params.verbose = cfg.verbose
    params.anneal = cfg.anneal
    params.incr_control = cfg.incr_control
    params.gbda_deterministic = True
    params.lr = 0.01
    params.result_prefix = "results/gcg"

    return params


def _generate_response(
    goal: str,
    target: str,
    control: str,
    worker,
    managers: Dict,
    max_new_tokens: int = 256,
) -> str:
    """Generate model completion with optimized adversarial suffix."""
    from copy import deepcopy

    conv = deepcopy(worker.conv_template)
    ap = managers["AP"](
        goal,
        target,
        worker.tokenizer,
        conv,
        control,
    )
    input_ids = ap.input_ids[: ap._assistant_role_slice.stop].unsqueeze(0).to(worker.model.device)
    attn = torch.ones_like(input_ids)
    gen_config = worker.model.generation_config
    gen_config.max_new_tokens = max_new_tokens
    with torch.no_grad():
        out = worker.model.generate(
            input_ids,
            attention_mask=attn,
            generation_config=gen_config,
            pad_token_id=worker.tokenizer.pad_token_id,
        )
    new_tokens = out[0, ap._assistant_role_slice.stop :]
    return worker.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_single_behavior_gcg(
    goal: str,
    target: str,
    workers,
    cfg: GCGConfig,
    managers: Dict,
    log_dir: Path,
    sample_idx: int,
) -> Dict[str, Any]:
    """Run GCG for one AdvBench behavior (individual attack)."""
    target_aug = _randomize_targets([target])[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = log_dir / f"sample_{sample_idx:03d}_{timestamp}.json"

    attack = gcg.IndividualPromptAttack(
        [goal],
        [target_aug],
        workers,
        control_init=cfg.control_init,
        logfile=str(logfile),
        managers=managers,
        mpa_batch_size=cfg.batch_size,
        mpa_n_steps=cfg.n_steps,
    )

    t0 = time.time()
    best_control, _ = attack.run(
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        topk=cfg.topk,
        temp=cfg.temp,
        allow_non_ascii=cfg.allow_non_ascii,
        target_weight=cfg.target_weight,
        control_weight=cfg.control_weight,
        anneal=cfg.anneal,
        test_steps=cfg.test_steps,
        incr_control=cfg.incr_control,
        stop_on_success=cfg.stop_on_success,
        verbose=cfg.verbose,
        filter_cand=cfg.filter_cand,
    )
    elapsed = time.time() - t0

    response = _generate_response(goal, target_aug, best_control, workers[0], managers)

    return {
        "index": sample_idx,
        "goal": goal,
        "target": target,
        "target_augmented": target_aug,
        "control": best_control,
        "response": response,
        "runtime_sec": elapsed,
        "logfile": str(logfile),
    }


def run_gcg_attack(
    model_key: str,
    cfg: Optional[GCGConfig] = None,
    advbench_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run GCG on one model over AdvBench samples.

    Returns a results dict and writes JSON to cfg.output_dir.
    """
    import torch.multiprocessing as mp

    cfg = cfg or GCGConfig()
    spec = get_model_spec(model_key)

    mp.set_start_method("spawn", force=True)

    samples = load_advbench(advbench_csv, max_samples=cfg.max_samples, offset=cfg.data_offset)
    goals, targets = goals_and_targets(advbench_csv, cfg.max_samples, cfg.data_offset)

    out_dir = Path(cfg.output_dir) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # llm-attacks expects a CSV path for get_goals_and_targets; write temp subset.
    subset_csv = out_dir / "advbench_subset.csv"
    import pandas as pd

    pd.DataFrame({"goal": goals, "target": targets}).to_csv(subset_csv, index=False)

    params = _build_params(spec, cfg, subset_csv)
    workers, test_workers = get_workers(params)

    managers = {
        "AP": gcg.AttackPrompt,
        "PM": gcg.PromptManager,
        "MPA": gcg.MultiPromptAttack,
    }

    print(f"\n{'=' * 70}")
    print(f"GCG Attack | {spec.description}")
    print(f"Model: {params.model_paths[0]}")
    print(f"Template: {spec.conv_template}")
    print(f"Samples: {len(samples)} | Steps/behavior: {cfg.n_steps}")
    print(f"{'=' * 70}\n")

    results: List[Dict[str, Any]] = []
    try:
        for i, sample in enumerate(samples):
            print(f"\n[{i + 1}/{len(samples)}] {sample.goal[:80]}...")
            row = run_single_behavior_gcg(
                sample.goal,
                sample.target,
                workers,
                cfg,
                managers,
                log_dir,
                sample.index,
            )
            results.append(row)
            # Persist incrementally
            _save_results(out_dir, spec, cfg, results, partial=True)
    finally:
        for w in workers + test_workers:
            w.stop()

    payload = _save_results(out_dir, spec, cfg, results, partial=False)
    return payload


def _save_results(
    out_dir: Path,
    spec: ModelSpec,
    cfg: GCGConfig,
    results: List[Dict],
    partial: bool,
) -> Dict[str, Any]:
    payload = {
        "attack": "gcg",
        "implementation": "llm-attacks",
        "model_key": spec.key,
        "model_id": spec.model_id,
        "model_family": spec.family,
        "conv_template": spec.conv_template,
        "n_steps": cfg.n_steps,
        "batch_size": cfg.batch_size,
        "topk": cfg.topk,
        "max_samples": cfg.max_samples,
        "timestamp": datetime.now().isoformat(),
        "partial": partial,
        "num_completed": len(results),
        "samples": results,
    }
    suffix = ".partial" if partial else ""
    out_path = out_dir / f"gcg_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    if not partial:
        print(f"\nResults saved to {out_path}")
    return payload


def run_gcg_all_models(
    model_keys: Optional[List[str]] = None,
    cfg: Optional[GCGConfig] = None,
    advbench_csv: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    keys = resolve_model_keys(model_keys)
    all_results = []
    for key in keys:
        all_results.append(run_gcg_attack(key, cfg=cfg, advbench_csv=advbench_csv))
    return all_results


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GCG jailbreak attack via llm-attacks on AdvBench"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model key. Choices: {', '.join(resolve_model_keys())}",
    )
    parser.add_argument("--all-models", action="store_true", help="Run all registered models")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--data-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="results/gcg")
    parser.add_argument("--advbench-csv", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None, help="Override HF model path")
    args = parser.parse_args(argv)

    if not args.model and not args.all_models:
        parser.error("Specify --model <key> or --all-models")

    cfg = GCGConfig(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        max_samples=args.max_samples,
        data_offset=args.data_offset,
        device=args.device,
        output_dir=args.output_dir,
        model_path_override=args.model_path,
    )
    csv_path = Path(args.advbench_csv) if args.advbench_csv else None

    if args.all_models:
        run_gcg_all_models(cfg=cfg, advbench_csv=csv_path)
    else:
        run_gcg_attack(args.model, cfg=cfg, advbench_csv=csv_path)


if __name__ == "__main__":
    # Allow running as: python -m attacks.gcg.runner
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    main()
