"""GCG attack package (llm-attacks wrapper)."""

__all__ = ["run_gcg_attack"]


def __getattr__(name: str):
    if name == "run_gcg_attack":
        from .runner import run_gcg_attack
        return run_gcg_attack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
