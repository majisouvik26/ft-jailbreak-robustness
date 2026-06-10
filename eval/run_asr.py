"""
Evaluate GCG attack results with GPT-4.1 mini judge (ASR).

Usage:
    python -m eval.run_asr --results results/gcg/llama-3.1-8b-it/gcg_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.judge import JudgeConfig, compute_asr_metrics, judge_responses


def load_attack_results(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def run_asr_eval(
    results_path: Path,
    output_path: Optional[Path] = None,
    judge_cfg: Optional[JudgeConfig] = None,
) -> Dict[str, Any]:
    data = load_attack_results(results_path)
    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in {results_path}")

    instructions = [s["goal"] for s in samples]
    responses = [s.get("response", "") for s in samples]

    scores, raw_judge = judge_responses(instructions, responses, cfg=judge_cfg)
    metrics = compute_asr_metrics(scores)

    evaluated = []
    for sample, score, raw in zip(samples, scores, raw_judge):
        evaluated.append({**sample, "judge_score": score, "judge_raw": raw})

    payload = {
        "evaluation": "asr",
        "judge_model": (judge_cfg or JudgeConfig()).model,
        "attack_results_path": str(results_path),
        "model_key": data.get("model_key"),
        "model_id": data.get("model_id"),
        "attack": data.get("attack", "gcg"),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "samples": evaluated,
    }

    out = output_path or results_path.with_name("asr_eval.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"ASR Evaluation | {data.get('model_key', 'unknown')}")
    print(f"  ASR (score=5):     {metrics['asr']:.2%}")
    print(f"  High harm (4-5):   {metrics['high_harm_rate']:.2%}")
    print(f"  Safety (1-2):      {metrics['safety_rate']:.2%}")
    print(f"  Avg score:         {metrics['avg_score']:.2f}")
    print(f"  Valid/Total:       {metrics['num_valid']}/{len(samples)}")
    print(f"Saved to: {out}")
    print(f"{'=' * 50}")

    return payload


def run_asr_batch(
    results_dir: Path,
    pattern: str = "gcg_results.json",
) -> List[Dict[str, Any]]:
    paths = sorted(results_dir.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} under {results_dir}")
    return [run_asr_eval(p) for p in paths]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate GCG ASR with GPT-4.1 mini")
    parser.add_argument("--results", type=str, help="Path to gcg_results.json")
    parser.add_argument("--results-dir", type=str, help="Evaluate all gcg_results.json under dir")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args(argv)

    if not args.results and not args.results_dir:
        parser.error("Provide --results or --results-dir")

    cfg = JudgeConfig(model=args.judge_model)

    if args.results_dir:
        run_asr_batch(Path(args.results_dir))
    else:
        out = Path(args.output) if args.output else None
        run_asr_eval(Path(args.results), output_path=out, judge_cfg=cfg)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    main()
