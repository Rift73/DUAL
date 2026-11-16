"""A DUAL checkpoint's state dict, and the preset, scale and form its weights show."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import UserError

# (embed_dim, depths, pixel unshuffle) -> the exporter's factory
PRESETS = {
    (128, (4, 3, 3, 6), True): "dual_light",
    (128, (4, 3, 3, 6), False): "dual_xs",
    (160, (6,) * 6, False): "dual_s",
    (192, (6,) * 6, False): "dual_m",
    (224, (6,) * 10, False): "dual_l",
    (256, (6,) * 12, False): "dual_xl",
}
# A wrapped .pth's state dicts, in the order one is taken.
STATE_KEYS = ("params_ema", "params")


def read_state(path: Path, state_key: str | None) -> tuple[dict[str, Any], str | None]:
    """The state dict a checkpoint holds and the key it was found under: `state_key`
    when given, else a .pth's first of STATE_KEYS, else the file's own dictionary
    (None). Only tensors under string keys, all finite, are taken."""
    import pickle
    import zipfile

    import torch

    if path.suffix.lower() == ".safetensors":
        if state_key:
            raise UserError(
                f"{path.name} is a .safetensors file, which holds one state dict: omit"
                " --state-key."
            )
        from safetensors import SafetensorError
        from safetensors.torch import load_file

        try:
            state = load_file(str(path), device="cpu")
        except (SafetensorError, OSError) as error:
            raise UserError(
                f"{path.name} cannot be read as safetensors: {error}"
            ) from error
    else:
        try:
            content = torch.load(path, map_location="cpu", weights_only=True)
        except (
            pickle.UnpicklingError, EOFError, zipfile.BadZipFile, RuntimeError
        ) as error:
            first = (str(error).strip() or type(error).__name__).splitlines()[0]
            raise UserError(
                f"{path.name} cannot be read as a .pth checkpoint: {first}"
            ) from error
        if not isinstance(content, dict):
            raise UserError(f"{path.name} does not hold a dictionary of weights.")
        if state_key is None:
            state_key = next(
                (key for key in STATE_KEYS if isinstance(content.get(key), dict)), None
            )
        if state_key is None:
            state = content
        elif isinstance(content.get(state_key), dict):
            state = content[state_key]
        else:
            keys = ", ".join(str(key) for key in content)
            raise UserError(
                f"{path.name} has no state dict under {state_key!r}; its keys: {keys}."
            )
    if not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise UserError(
            f"{path.name} holds no state dict of tensors"
            + (f" under {state_key!r}" if state_key else "")
            + "; --state-key chooses the entry that holds the weights."
        )
    if any(
        v.is_floating_point() and not torch.isfinite(v).all().item()
        for v in state.values()
    ):
        raise UserError(f"{path.name} holds weights that are not finite numbers.")
    return state, state_key


def sequence_length(state: dict[str, Any], prefix: str) -> int:
    """One more than the largest index i of the keys `{prefix}.{i}.*`; 0 for none."""
    start = prefix + "."
    indices = {
        int(key[len(start) :].split(".", 1)[0])
        for key in state
        if key.startswith(start)
    }
    return max(indices) + 1 if indices else 0


def routed_parts(state: dict[str, Any]) -> int:
    """The experts of a state dict's routed convolutions (their channel blocks of C with
    the routing per position; the exporter's spec.routed_parts)."""
    return max(
        value.shape[0] for key, value in state.items() if key.endswith(".experts")
    )


def detect(state: dict[str, Any]) -> tuple[str | None, int | None, bool]:
    """The factory and scale a DUAL state dict's shapes give (None where they match
    none of the six presets or scales 1, 2 and 4) and whether it is folded (one plain
    convolution per reparameterized block, as exported models hold)."""
    folded = "stem.weight" in state
    if folded:
        embed_dim, in_channels = state["stem.weight"].shape[:2]
    elif "stem.direct.expand.weight" in state:
        in_channels = state["stem.direct.expand.weight"].shape[1]
        embed_dim = state["stem.direct.reduce.weight"].shape[0]
    else:
        raise UserError(
            "The file's weights are not a DUAL image model (they have no DUAL stem)."
        )
    unshuffle = in_channels == 12
    # the tail: its first convolution, one per x2 stage (one more with the unshuffle)
    # and its last, with an activation between each two
    stages = (sequence_length(state, "tail") + 1) // 2 - 2 - int(unshuffle)
    depths = tuple(
        sequence_length(state, f"groups.{group}.layers")
        for group in range(sequence_length(state, "groups"))
    )
    scale = 2**stages if stages in (0, 1, 2) else None
    return PRESETS.get((embed_dim, depths, unshuffle)), scale, folded
