"""PyTorch model components."""

from .graph_tokenized_machiningfm import GraphTokenizedStemGNNDecoderOnlyMachiningFM

try:
    from .machiningfm import MachiningFM
except ImportError:
    MachiningFM = None  # type: ignore[assignment,misc]

__all__ = ["MachiningFM", "GraphTokenizedStemGNNDecoderOnlyMachiningFM"]
