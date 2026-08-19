from __future__ import annotations


def get_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
