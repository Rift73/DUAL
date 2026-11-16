"""DUAL's TensorRT-only ONNX from a checkpoint: the vendored exporter's files for one
input size or for every size, joined into one model with its weights embedded and the
exporter's manifest in its metadata."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from . import UserError
from .sizes import ALIGNMENT

# The exporter's manifest (export.json) in the ONNX model's metadata, under the key
# chaiNNer-C reads too, so its DUAL ONNX files and this tool's build alike.
METADATA_KEY = "chainner_c.dual_tensorrt.export"


def read_manifest(onnx_bytes: bytes, name: str) -> dict[str, Any]:
    """The exporter's manifest of a DUAL ONNX written by export_onnx (the file `name`)."""
    import onnx
    from google.protobuf.message import DecodeError

    try:
        model = onnx.load_from_string(onnx_bytes)
    except DecodeError as error:
        raise UserError(f"{name} cannot be read as ONNX: {error}") from error
    for entry in model.metadata_props:
        if entry.key == METADATA_KEY:
            return json.loads(entry.value)
    raise UserError(f"{name} is not a DUAL ONNX written by dual-convert.")


def export_onnx(
    checkpoint: Path,
    factory: str,
    scale: int,
    size: tuple[int, int] | None,
    *,
    state_key: str | None,
    folded: bool,
    routing_na: bool,
) -> bytes:
    """The ONNX model of a checkpoint for one input size (height, width), or for every
    size (None; vendor/scripts/dual_tensorrt/dynamic.py). routing_na: the expert
    routing per position (the manifest's "routing")."""
    import onnx

    from .vendor.scripts.dual_tensorrt.dynamic import export_dynamic
    from .vendor.scripts.dual_tensorrt.export import export_checkpoint

    with tempfile.TemporaryDirectory(prefix="dual-convert-") as temporary:
        export = Path(temporary) / "export"
        options = {"state_key": state_key, "folded": folded, "routing_na": routing_na}
        try:
            if size is None:
                export_dynamic(factory, scale, ALIGNMENT, checkpoint, export, **options)
            else:
                height, width = size
                export_checkpoint(
                    factory, scale, height, width, checkpoint, export, **options
                )
        except RuntimeError as error:
            # The strict load of the preset's model: the weights are another model's.
            if "Error(s) in loading state_dict" not in str(error):
                raise
            raise UserError(
                f"The weights do not fit DUAL {factory.removeprefix('dual_')} x{scale}:"
                f" {mismatch(str(error))} (check --preset and --scale)"
            ) from error
        manifest = json.loads((export / "export.json").read_text(encoding="utf-8"))
        model = onnx.load(str(export / "model.onnx"), load_external_data=True)
    # The factories' geometry: context windows cannot be read from the weights.
    manifest["geometry"] = "default (C32/C64 context windows assumed)"
    entry = model.metadata_props.add()
    entry.key, entry.value = METADATA_KEY, json.dumps(manifest)
    return model.SerializeToString()

def mismatch(message: str) -> str:
    """A short account of a strict load's error: how many weights are missing, unexpected
    or of another shape, with the first names of each."""
    parts = []
    for label, pattern in (
        ("missing", r"Missing key\(s\) in state_dict: ([^\n]*)"),
        ("unexpected", r"Unexpected key\(s\) in state_dict: ([^\n]*)"),
    ):
        found = re.search(pattern, message)
        if found:
            names = re.findall(r'"([^"]+)"', found.group(1))
            more = ", ..." if len(names) > 3 else ""
            parts.append(f"{len(names)} {label} ({', '.join(names[:3])}{more})")
    shapes = re.findall(r"size mismatch for ([^:]+):", message)
    if shapes:
        more = ", ..." if len(shapes) > 3 else ""
        parts.append(f"{len(shapes)} of another shape ({', '.join(shapes[:3])}{more})")
    return "; ".join(parts) or message.splitlines()[0]
