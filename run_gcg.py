#!/usr/bin/env python3
"""
CLI entry point for GCG attacks via llm-attacks.

Examples:
    python run_gcg.py --model llama-3.1-8b-it
    python run_gcg.py --model llama-3.2-3b-it --max-samples 2 --n-steps 20
    python run_gcg.py --all-models
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from attacks.gcg.runner import main

if __name__ == "__main__":
    main()
