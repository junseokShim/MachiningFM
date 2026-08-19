"""Shared utilities."""

from .config import load_config, save_config
from .paths import project_root

__all__ = ["load_config", "save_config", "project_root"]
