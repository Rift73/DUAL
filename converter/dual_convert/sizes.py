"""Input sizes: parsing WIDTHxHEIGHT, and the rules a fixed size or a size range of a
DUAL model keeps. Sizes are (height, width) inside the package."""

from __future__ import annotations

import argparse

# Every DUAL input size is a multiple of this (as the PyTorch model pads its input).
ALIGNMENT = 4
RANGE_OPTIONS = ("--min", "--opt", "--max")


def parse_size(text: str) -> tuple[int, int]:
    """A WIDTHxHEIGHT size as (height, width), both positive multiples of ALIGNMENT."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a size: write WIDTHxHEIGHT, e.g. 960x540"
        )
    width, height = (int(part) for part in parts)
    for name, value in (("width", width), ("height", height)):
        if not value or value % ALIGNMENT:
            raise argparse.ArgumentTypeError(
                f"{text}: the {name} {value} is not a positive multiple of {ALIGNMENT}"
            )
    return height, width


def size_text(size: tuple[int, int]) -> str:
    height, width = size
    return f"{width}x{height}"


def padded(size: tuple[int, int]) -> tuple[int, int]:
    """A (height, width) rounded up to multiples of ALIGNMENT."""
    height, width = size
    return -(-height // ALIGNMENT) * ALIGNMENT, -(-width // ALIGNMENT) * ALIGNMENT


def check_order(profile: tuple[tuple[int, int], ...]) -> None:
    """Refuse a (minimum, optimal, maximum) range that is not ordered in each axis,
    naming the two values that break the order."""
    for axis, name in ((1, "width"), (0, "height")):
        for (low_option, low), (high_option, high) in zip(
            zip(RANGE_OPTIONS, profile), zip(RANGE_OPTIONS[1:], profile[1:])
        ):
            if low[axis] > high[axis]:
                raise ValueError(
                    f"the {name} of {low_option} {size_text(low)} is larger than the"
                    f" {name} of {high_option} {size_text(high)}: the range needs"
                    " --min <= --opt <= --max in width and in height"
                )


def check_indexing(
    option: str,
    size: tuple[int, int],
    channels: int,
    unshuffle: bool,
    parts: int,
) -> None:
    """Refuse a size whose largest per-position tensor, [1, H, W, parts * C] of the
    trunk (half the input with the unshuffle), passes the kernels' and TensorRT's 32-bit
    element indexing. `parts` is 3 (the qkv), or with the routing per position the
    routed convolution's blocks when there are more."""
    height, width = size
    divisor = 2 if unshuffle else 1
    elements = parts * (height // divisor) * (width // divisor) * channels
    if elements >= 2**31:
        raise ValueError(
            f"{option} {size_text(size)} is too large for this model: its kernels index"
            f" {parts} x {width // divisor} x {height // divisor} x {channels} ="
            f" {elements} elements, which must stay below 2^31 ({2**31}); choose a"
            f" smaller {option}"
        )
