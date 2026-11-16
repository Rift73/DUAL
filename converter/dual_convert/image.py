"""A test picture as the engine's RGB input in [0, 1]."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import UserError

# Pillow's modes of 16-bit (or 32-bit integer) grey pictures
GREY_16 = ("I;16", "I;16B", "I;16L", "I")


def read_image(path: Path) -> tuple[np.ndarray, str | None]:
    """The picture as an FP32 [H, W, 3] array in [0, 1] (16-bit grey at its full
    precision), and a note when its transparency was dropped."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            transparent = "A" in mode or "transparency" in image.info
            if mode in GREY_16:
                grey = np.asarray(image, dtype=np.float64) / 65535
                pixels = np.repeat(np.clip(grey, 0, 1)[..., None], 3, axis=2)
            else:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    except FileNotFoundError as error:
        raise UserError(f"the test picture {path} does not exist.") from error
    except (UnidentifiedImageError, OSError) as error:
        raise UserError(
            f"the test picture {path.name} cannot be read: {error}"
        ) from error
    note = (
        f"{path.name}: its transparency is dropped (the engine takes RGB)."
        if transparent
        else None
    )
    return pixels.astype(np.float32), note
