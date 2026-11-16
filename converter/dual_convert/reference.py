"""The self-check's reference: the checkpoint's model in PyTorch, FP32 on the CPU, on a
seeded input."""

from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .checkpoint import PRESETS, read_state
from .sizes import ALIGNMENT

# The reference's peak memory in GiB: a base, plus TRUNK_GIB[factory] per million trunk
# positions, plus OUTPUT_GIB per million output pixels. Measured peaks (random weights,
# 2x at 384x384: Light 1.59, XS 3.33, S 3.64, M 3.74, L 4.52, XL 4.64 GiB; 2x Light at
# 960x540 3.27 and 1920x1080 8.30 GiB; 2x XS at 960x540 6.38 GiB) give each preset's
# trunk factor, rounded up; the wider and deeper presets keep more per position.
BASE_GIB = 1.6
TRUNK_GIB = {
    "dual_light": 8.0,
    "dual_xs": 11.0,
    "dual_s": 13.0,
    "dual_m": 14.0,
    "dual_l": 19.0,
    "dual_xl": 20.5,
}
OUTPUT_GIB = 0.31
# The reference's time per trunk position at 16 layers of 128 channels (in proportion to
# the layers and to the channels squared), on a 16-core processor.
TRUNK_SECONDS = 1.0e-4
# Correct engines' relative L2 differences ||engine - reference|| / ||reference|| on
# check_input grow with its size (the engine's body runs in BF16). Measured on one model
# family (2x Light: 0.0025 at 256x256, 0.00507 at 640x360, 0.00564 at 812x456, 0.00709 at
# 1280x720) and fitted by least squares as EXPECTED_AT_BASE at BASE_MPX megapixels and
# below, plus EXPECTED_PER_DOUBLING per doubling of the pixels. The check accepts
# LIMIT_FACTOR times that.
BASE_MPX = 0.0655
EXPECTED_AT_BASE = 0.0026
EXPECTED_PER_DOUBLING = 0.0012
LIMIT_FACTOR = 3
# A range engine is checked at a size whose reference takes at most this long.
RANGE_SECONDS = 30


def check_input(size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """A [1, 3, H, W] FP32 image in [0, 1] of four seeded regions: flat dark, vertical
    stripes, texture, and smooth waves. Regions this different make the expert routing
    per position differ from the whole image's mean, so the two routing forms differ
    clearly on it, while the smooth parts keep the engine's own error small."""
    height, width = size
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(0, 1, height, dtype=np.float64),
        np.linspace(0, 1, width, dtype=np.float64),
        indexing="ij",
    )
    channels = []
    for _ in range(3):
        wave = np.zeros_like(x)
        for frequency_y, frequency_x, phase in rng.uniform(0, 4 * np.pi, (4, 3)):
            wave += np.sin(frequency_y * y + frequency_x * x + phase)
        channels.append(0.5 + wave / 8)
    image = np.stack(channels)
    top, left = height // 2, width // 2
    image[:, :top, :left] = 0.1
    image[:, :top, left:] = 0.5 + 0.4 * np.sin(np.arange(left, width) * 0.8)
    image[:, top:, :left] = rng.uniform(0.25, 0.75, (3, height - top, left))
    return image[None].astype(np.float32)


def reference_output(
    checkpoint: Path,
    factory: str,
    scale: int,
    x: np.ndarray,
    *,
    state_key: str | None,
    folded: bool,
    routing_na: bool,
) -> np.ndarray:
    """The model's FP32 output for `x`, built and loaded as the exporter builds it,
    with the expert routing per position when `routing_na`."""
    import torch

    from .vendor.scripts.dual_tensorrt.export import load_arch

    torch.set_num_threads(os.cpu_count() or 1)
    with torch.device("cpu"):
        model = getattr(load_arch(), factory)(
            scale=scale, use_checkpoint=False, context_na=True, routing_na=routing_na
        )
    model.eval().requires_grad_(False)
    if folded:
        model.prepare_for_export()
    model.load_state_dict(read_state(checkpoint, state_key)[0], strict=True)
    model.prepare_for_export()
    with torch.inference_mode():
        return model(torch.from_numpy(x)).numpy()


def cost(factory: str, scale: int, size: tuple[int, int]) -> tuple[float, float]:
    """The reference's estimated peak memory (GiB) and time (seconds) at a (height,
    width) size."""
    channels, depths, unshuffle = next(
        key for key, value in PRESETS.items() if value == factory
    )
    height, width = size
    divisor = 2 if unshuffle else 1
    trunk = (height // divisor) * (width // divisor) / 1e6
    output = height * width * scale * scale / 1e6
    memory = BASE_GIB + TRUNK_GIB[factory] * trunk + OUTPUT_GIB * output
    seconds = TRUNK_SECONDS * 1e6 * trunk * sum(depths) / 16 * (channels / 128) ** 2
    return memory, seconds


def free_gib() -> float:
    """The physical memory free now, in GiB."""

    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong),
            ("available", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("available_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended", ctypes.c_ulonglong),
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.available / 2**30


def fits(memory: float, free: float) -> bool:
    """Whether an estimate leaves a clear margin in the free memory: a quarter of the
    estimate plus 1 GiB."""
    return memory * 1.25 + 1 <= free


def check_size(
    wanted: tuple[int, int],
    smallest: tuple[int, int],
    fit: Callable[[tuple[int, int]], bool],
) -> tuple[int, int] | None:
    """The (height, width) to check at: `wanted` if it fits, else the largest size below
    it in both axes, of its aspect, in multiples of ALIGNMENT and not below `smallest`,
    that fits; None if none does."""
    height, width = wanted
    for candidate in range(height, smallest[0] - 1, -ALIGNMENT):
        across = round(width * candidate / height / ALIGNMENT) * ALIGNMENT
        size = candidate, min(across, width)
        if size[1] >= smallest[1] and fit(size):
            return size
    return None


def limit(size: tuple[int, int]) -> float:
    """The largest relative L2 difference the check accepts at a (height, width) size."""
    height, width = size
    doublings = max(0.0, math.log2(height * width / 1e6 / BASE_MPX))
    return LIMIT_FACTOR * (EXPECTED_AT_BASE + EXPECTED_PER_DOUBLING * doublings)


def relative_l2(output: np.ndarray, reference: np.ndarray) -> float:
    difference = output.astype(np.float64) - reference
    return float(np.linalg.norm(difference) / np.linalg.norm(reference))
