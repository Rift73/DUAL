"""DUAL checkpoints to TensorRT: the TensorRT-only ONNX and the engine."""

import os
from pathlib import Path

__version__ = "1.0.0"
# Triton's compiled kernels, beside the tool (so removing its folder removes them),
# unless TRITON_CACHE_DIR names another place: set on the package's import, before any
# of its modules can import Triton.
os.environ.setdefault(
    "TRITON_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "cache" / "triton")
)


class UserError(Exception):
    """A mistake in the input or the options: the command shows it as one line."""
