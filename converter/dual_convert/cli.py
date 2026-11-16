"""The dual-convert command line."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import UserError, __version__
from .checkpoint import PRESETS, detect, read_state, routed_parts
from .engine import OUTPUT, build_engine, gpu_name, gpu_names, input_name
from .image import read_image
from .onnx_export import export_onnx, read_manifest
from .sizes import (
    ALIGNMENT,
    RANGE_OPTIONS,
    check_indexing,
    check_order,
    padded,
    parse_size,
    size_text,
)

# The command line's preset names -> the exporter's factories
FACTORIES = {
    "light": "dual_light",
    "xs": "dual_xs",
    "s": "dual_s",
    "m": "dual_m",
    "l": "dual_l",
    "xl": "dual_xl",
}
PRESET_NAMES = {factory: name for name, factory in FACTORIES.items()}
CHANNELS = {factory: dim for (dim, _, _), factory in PRESETS.items()}
CHECKPOINTS = (".safetensors", ".pth")
FIXED_EXAMPLE = "--size 960x540"
RANGE_EXAMPLE = "--min 256x256 --opt 1280x720 --max 1920x1088"
CHOOSE = (
    f"type --size WxH for one fixed size (e.g. {FIXED_EXAMPLE}) or --min WxH --opt WxH"
    f" --max WxH for a size range (e.g. {RANGE_EXAMPLE})"
)

DESCRIPTION = """\
Convert a DUAL image upscaling model, trained with traiNNer, for NVIDIA TensorRT:
into the TensorRT-only ONNX file (--onnx), into a TensorRT engine with DUAL's GPU
kernels embedded (--engine), or into both (neither option, the default).

The ONNX holds DUAL's TensorRT plugin nodes: only TensorRT, with this tool's engine
build, can run it. Its precision is mixed and fixed: a BF16 body with FP32 input and
output.

<input> is a checkpoint (.safetensors or .pth) or an .onnx file written by this tool;
from an .onnx file only an engine is built."""

EPILOG = f"""\
sizes (required; there is no default):
  Sizes are written WIDTHxHEIGHT in pixels.
  From a checkpoint, give exactly one of:
    --size WxH                       a fixed size: the ONNX and the engine take this
                                     one input size ({FIXED_EXAMPLE})
    --min WxH --opt WxH --max WxH    a size range: a dynamic ONNX, which takes any
                                     size that is a multiple of {ALIGNMENT}, and an engine
                                     for the sizes from --min to --max, tuned for --opt
                                     ({RANGE_EXAMPLE})
    All three range options are needed, also with --onnx alone: the range is checked
    against the model's limits, and with --engine it is the engine's.
  From an .onnx file:
    a fixed ONNX carries its size: give no size (a --size must equal it)
    a dynamic ONNX needs --min WxH --opt WxH --max WxH for its engine
  Rules: width and height are multiples of {ALIGNMENT}; --min <= --opt <= --max in width
  and in height; the largest size keeps the kernels' 32-bit indexing:
  3 x W x H x channels below 2^31, with W and H halved for Light (more than 3 parts
  with --routing-na when the model has more experts). Channels: Light and XS 128,
  S 160, M 192, L 224, XL 256.

model:
  The preset (Light, XS, S, M, L or XL) and the scale (1, 2 or 4) are read from the
  checkpoint's weights; --preset and --scale replace what was read.
  A .pth file that wraps its weights is read from "params_ema", else "params", else
  the file's own dictionary; the one used is printed. --state-key chooses another.
  --routing-na: the checkpoint does not record whether the model was trained with
  routing_na: true. For such a model pass --routing-na, or its engine computes the
  expert routing from the whole image's mean instead of per position.

check:
  After an engine build from a checkpoint, the engine runs once in a process of its own
  (with no plugin loaded) on a seeded test image at the fixed size or at --opt, and its
  output is compared with the PyTorch model of the same options (FP32, on the processor):
  this shows that the engine computes that model on this PC, not that the options (such as
  --routing-na) match the training. The model's memory need and time are estimated first:
  a range engine is checked at --opt, or at the largest smaller size of the same shape (not
  below --min) whose model fits in the free memory and takes at most about 30 s; a fixed
  engine at its own size however long it takes, or, if its model does not fit in the free
  memory, only run once and checked for finite output (the line says so, and this is no
  failure). The accepted difference grows
  with the check size; the line shows it. A failed check ends with an error and leaves the
  engine as <name>.engine.failed. --no-verify skips it; from an .onnx input it cannot run
  (there is no model to compare with).

output:
  Files are written next to the input, or into --out. Their names hold the model's
  name (the input's file name), the size ("dynamic" for a dynamic ONNX, the range for
  its engine), the routing form ("standard" or "routing-na") and, for an engine, the
  TensorRT version. An existing file stops the command before any work unless --overwrite
  is given. The engine is written as <name>.engine.unchecked and renamed when its check
  has an outcome.

examples:
  dual-convert model.safetensors --onnx --size 960x540
      The ONNX for 960x540 input only, as model_960x540_standard.onnx.
  dual-convert net_g_ema.pth --routing-na --min 256x256 --opt 1920x1088 --max 1920x1088
      The dynamic ONNX and an engine for 256x256 to 1920x1088, with the expert routing
      per position, from the "params_ema" weights.
  dual-convert model_dynamic_standard.onnx {RANGE_EXAMPLE}
      An engine for 256x256 to 1920x1088, tuned for 1280x720, from a dynamic ONNX
      made before.
  dual-convert model_960x540_standard.onnx --test photo.png
      An engine for the fixed ONNX's own size (960x540), checked on photo.png."""


def output_name(model: str, shape: str, routing_na: bool, extension: str) -> str:
    """A written file's name: the model's name, its size or range, its routing form and
    `extension` (with an engine's TensorRT version)."""
    form = "routing-na" if routing_na else "standard"
    return f"{model}_{shape}_{form}{extension}"


def model_name(onnx_stem: str) -> str:
    """The model's name in an ONNX file's name written by output_name."""
    return re.sub(r"_(dynamic|\d+x\d+)_(standard|routing-na)$", "", onnx_stem)


def run_child(
    mode: str, engine: Path, args: argparse.Namespace, source: Path, target: Path
) -> None:
    """runner.py's `mode` in a process of its own; its failure ends this one with its
    exit code."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "dual_convert.runner",
            mode,
            str(engine),
            str(args.gpu),
            "1" if args.verbose else "0",
            str(source),
            str(target),
        ],
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if done.returncode:
        sys.exit(done.returncode)


def sidecar_contents(
    manifest: dict[str, Any],
    profile: tuple[tuple[int, int], ...] | None,
    tensorrt_version: str,
    gpu: str,
    check: dict[str, Any] | None,
) -> dict[str, Any]:
    """What an engine's .json file records about it."""
    spec = manifest["specialization"]
    if profile is None:
        sizes = {"size": size_text((spec["height"], spec["width"]))}
    else:
        sizes = {
            name: size_text(size) for name, size in zip(("min", "opt", "max"), profile)
        }
    return {
        "tool": f"dual-convert {__version__}",
        "preset": PRESET_NAMES[spec["factory"]],
        "scale": spec["scale"],
        "routing": "routing-na" if "routing" in manifest else "standard",
        "input": {
            "name": input_name(spec),
            "layout": "NCHW, batch 1, RGB",
            "type": "float32",
            "values": "0 to 1",
            **sizes,
            "size_format": "WIDTHxHEIGHT",
            "multiple_of": spec.get("alignment", 4),
        },
        "output": {
            "name": OUTPUT,
            "layout": "NCHW, batch 1, RGB",
            "type": "float32",
            "values": "about 0 to 1 (clip before converting to 8 bits)",
            "size": f"{spec['scale']} x the input's width and height",
        },
        "tensorrt": tensorrt_version,
        "gpu": gpu,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "self_check": check,
    }


def self_check(
    engine: Path,
    args: argparse.Namespace,
    profile: tuple[tuple[int, int], ...] | None,
    factory: str,
    scale: int,
    state_key: str | None,
    folded: bool,
    routing_na: bool,
) -> dict[str, Any]:
    """The engine run in a fresh process on check_input and compared with the PyTorch
    model of the same options, at the size the free memory and time allow: the sidecar's
    record, with "passed" and, when it did not pass, "failure"."""
    import numpy as np

    from . import reference

    wanted = args.size if profile is None else profile[1]
    free = reference.free_gib()

    def acceptable(candidate: tuple[int, int]) -> bool:
        # A fixed engine is checked at its size however long it takes; a range
        # engine at a size whose reference also takes at most RANGE_SECONDS.
        memory, seconds = reference.cost(factory, scale, candidate)
        return reference.fits(memory, free) and (
            profile is None or seconds <= reference.RANGE_SECONDS
        )

    smallest = wanted if profile is None else profile[0]
    size = reference.check_size(wanted, smallest, acceptable)
    x = reference.check_input(size or wanted)
    with tempfile.TemporaryDirectory(prefix="dual-convert-") as temporary:
        source, result = Path(temporary, "x.npy"), Path(temporary, "y.npy")
        np.save(source, x)
        run_child("check", engine, args, source, result)
        output = np.load(result)
    if size is None:
        memory, seconds = reference.cost(factory, scale, smallest)
        reason = (
            f"the model on the CPU would need about {memory:.1f} GiB"
            + ("" if profile is None else f" and {seconds:.0f} s")
            + f" at {size_text(smallest)}, and {free:.1f} GiB of memory is free"
        )
        finite = bool(np.isfinite(output).all())
        if not finite:
            return {"passed": False, "failure": "the engine's output is not finite."}
        print(
            f"Self-check: comparison skipped ({reason}); the engine loaded in a fresh"
            " process and gave finite output, but it was not compared with the model."
        )
        return {"passed": True, "compared": False, "reason": reason}
    memory, seconds = reference.cost(factory, scale, size)
    if size != wanted:
        print(
            f"Self-check: at {size_text(size)}, the largest size up to --opt whose model"
            f" on the CPU fits in the free memory ({free:.1f} GiB) and takes at most"
            f" about {reference.RANGE_SECONDS} s"
        )
    print(
        f"Self-check: running the model (PyTorch FP32 on the CPU) at {size_text(size)},"
        f" about {memory:.1f} GiB"
        + (f" and about {seconds:.0f} s" if seconds > 20 else "")
        + " ...",
        flush=True,
    )
    expected = reference.reference_output(
        args.input.resolve(),
        factory,
        scale,
        x,
        state_key=state_key,
        folded=folded,
        routing_na=routing_na,
    )
    difference = reference.relative_l2(output, expected)
    allowed = reference.limit(size)
    check = {
        "passed": difference <= allowed,
        "compared": True,
        "relative_l2": difference,
        "limit": allowed,
        "size": size_text(size),
    }
    if not check["passed"]:
        check["failure"] = (
            f"the engine differs from the PyTorch model of the same options by"
            f" {difference:.4f} (relative L2), above the limit {allowed:.4f} at"
            f" {size_text(size)}."
        )
        return check
    print(
        f"Self-check: passed, relative L2 difference {difference:.4f} (limit"
        f" {allowed:.4f} at {size_text(size)})"
    )
    return check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dual-convert",
        usage="dual-convert <input> (--size WxH | --min WxH --opt WxH --max WxH)"
        " [options]",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("input", type=Path, metavar="<input>", help=argparse.SUPPRESS)
    what = parser.add_argument_group("what to write (neither: both)")
    what.add_argument(
        "-onnx", "--onnx", action="store_true", help="write the TensorRT-only ONNX"
    )
    what.add_argument(
        "-engine", "--engine", action="store_true", help="build the TensorRT engine"
    )
    sizes = parser.add_argument_group("size (required, see 'sizes' below)")
    sizes.add_argument(
        "-size", "--size", type=parse_size, metavar="WxH", help="one fixed input size"
    )
    for option, label in zip(RANGE_OPTIONS, ("smallest", "optimal", "largest")):
        sizes.add_argument(
            option[1:],
            option,
            type=parse_size,
            metavar="WxH",
            help=f"the {label} size of a size range",
        )
    model = parser.add_argument_group("model")
    model.add_argument(
        "-routing-na",
        "--routing-na",
        "-high-quality",
        "--high-quality",
        dest="routing_na",
        action="store_true",
        help="the expert routing per position, for models trained with"
        " routing_na: true (not recorded in the checkpoint: pass it yourself)",
    )
    model.add_argument(
        "-preset",
        "--preset",
        choices=FACTORIES,
        help="the preset, instead of the one read from the weights",
    )
    model.add_argument(
        "-scale",
        "--scale",
        type=int,
        choices=(1, 2, 4),
        help="the scale, instead of the one read from the weights",
    )
    model.add_argument(
        "-state-key",
        "--state-key",
        metavar="KEY",
        help="the key of a .pth file's state dict",
    )
    other = parser.add_argument_group("output and checks")
    other.add_argument(
        "-out",
        "--out",
        type=Path,
        metavar="FOLDER",
        help="the folder to write into (default: the input's folder)",
    )
    other.add_argument(
        "-test",
        "--test",
        type=Path,
        metavar="IMAGE",
        help="after the engine build, upscale IMAGE with the engine and write the"
        " result next to the engine",
    )
    other.add_argument(
        "-no-verify",
        "--no-verify",
        action="store_true",
        help="skip the check after the engine build that runs the engine once and"
        " compares its output with the model's",
    )
    other.add_argument(
        "-overwrite",
        "--overwrite",
        action="store_true",
        help="replace this run's existing outputs (ONNX, engine, its .json, the test"
        " picture, and a leftover .unchecked or .failed engine)",
    )
    other.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        default=0,
        metavar="INDEX",
        help="the NVIDIA GPU to build and run on (default 0, the first)",
    )
    other.add_argument(
        "-verbose",
        "--verbose",
        action="store_true",
        help="also show TensorRT's warnings and information (errors always show)",
    )
    other.add_argument(
        "-h", "-help", "--help", action="help", help="print this help and exit"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Lines in order with the child processes' output
    sys.stdout.reconfigure(line_buffering=True)
    try:
        convert(parser, args)
    except UserError as error:
        sys.exit(f"dual-convert: {error}")


def convert(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    suffix = args.input.suffix.lower()
    if suffix not in (*CHECKPOINTS, ".onnx"):
        parser.error(f"{args.input.name}: the input must be .safetensors, .pth or .onnx")
    if not args.input.is_file():
        parser.error(f"{args.input} does not exist")
    from_onnx = suffix == ".onnx"
    if from_onnx and args.onnx:
        parser.error("from an .onnx file only an engine can be built")
    write_engine = args.engine or not args.onnx
    if (args.test or args.no_verify) and not write_engine:
        parser.error("--test and --no-verify apply to an engine build")
    values = (args.min, args.opt, args.max)
    given = [option for option, value in zip(RANGE_OPTIONS, values) if value]
    if args.size and given:
        parser.error(f"--size and {given[0]} exclude each other: {CHOOSE}")
    if given and len(given) < len(RANGE_OPTIONS):
        missing = " and ".join(o for o in RANGE_OPTIONS if o not in given)
        parser.error(
            f"a size range needs all of --min, --opt and --max; {missing} missing"
            f" (e.g. {RANGE_EXAMPLE})"
        )
    profile = values if given else None
    if profile is not None:
        try:
            check_order(profile)
        except ValueError as error:
            parser.error(str(error))
    if from_onnx:
        given = [
            option
            for option, value in (
                ("--routing-na", args.routing_na),
                ("--preset", args.preset),
                ("--scale", args.scale),
                ("--state-key", args.state_key),
            )
            if value
        ]
        if given:
            parser.error(f"{given[0]} does not apply to an .onnx input: it is in the file")
        onnx_bytes = args.input.read_bytes()
        try:
            manifest = read_manifest(onnx_bytes, args.input.name)
        except ValueError as error:
            parser.error(str(error))
        spec = manifest["specialization"]
        if spec.get("dynamic", False):
            if args.size:
                parser.error(
                    "this ONNX is dynamic: its engine takes --min WxH --opt WxH"
                    f" --max WxH, not --size (e.g. {RANGE_EXAMPLE})"
                )
            if profile is None:
                parser.error(
                    "this ONNX is dynamic (it takes any size): its engine needs"
                    f" --min WxH --opt WxH --max WxH (e.g. {RANGE_EXAMPLE})"
                )
        else:
            fixed = spec["height"], spec["width"]
            if profile is not None:
                parser.error(
                    f"this ONNX is fixed at {size_text(fixed)}: give no --min, --opt"
                    " or --max"
                )
            if args.size and args.size != fixed:
                parser.error(
                    f"--size {size_text(args.size)} contradicts this ONNX's fixed size"
                    f" {size_text(fixed)}; omit --size"
                )
            args.size = fixed
        channels, unshuffle = spec["channels"], spec["unshuffle"]
        routing_na = "routing" in manifest
        parts = max(3, manifest["routed_parts"]) if routing_na else 3
        factory, scale = spec["factory"], spec["scale"]
        model = model_name(args.input.stem)
    elif not args.size and profile is None:
        parser.error(f"a size is required: {CHOOSE}")
    else:
        print(f"Reading {args.input.name} ...", flush=True)
        state, state_key = read_state(args.input, args.state_key)
        if suffix == ".pth":
            print(
                "Weights:",
                f"the state dict under {state_key!r}"
                if state_key
                else "the file's own dictionary",
            )
        factory, scale, folded = detect(state)
        if args.preset:
            factory = FACTORIES[args.preset]
        if args.scale:
            scale = args.scale
        if factory is None:
            sizes = ", ".join(sorted({str(dim) for dim, _, _ in PRESETS}))
            raise UserError(
                "the weights match none of the six presets (embed_dim"
                f" {sizes}); pass --preset"
            )
        if scale is None:
            raise UserError("the scale could not be read from the weights; pass --scale")
        channels, unshuffle = CHANNELS[factory], factory == "dual_light"
        routing_na = args.routing_na
        parts = max(3, routed_parts(state)) if routing_na else 3
        del state
        model = args.input.stem
        print(f"Model: DUAL {PRESET_NAMES[factory]} x{scale}{' (folded)' if folded else ''}")
    try:
        if profile is None:
            check_indexing("--size", args.size, channels, unshuffle, parts)
        else:
            check_indexing("--max", profile[2], channels, unshuffle, parts)
    except ValueError as error:
        parser.error(str(error))
    if args.test:
        pixels, _ = read_image(args.test)
        picture = pixels.shape[:2]
        if profile is None and picture != args.size:
            raise UserError(
                f"{args.test.name} is {size_text(picture)}, but a fixed engine for"
                f" {size_text(args.size)} takes only images of that size: resize or crop"
                " the image, or build for its size or a size range."
            )
        if profile is not None and not all(
            low <= side <= high
            for low, side, high in zip(profile[0], padded(picture), profile[2])
        ):
            raise UserError(
                f"{args.test.name} is {size_text(picture)}, outside the range"
                f" {size_text(profile[0])} to {size_text(profile[2])}."
            )
    if write_engine:
        names = gpu_names()
        if not 0 <= args.gpu < len(names):
            found = ", ".join(f"{i}: {name}" for i, name in enumerate(names))
            raise UserError(
                f"GPU {args.gpu} does not exist; "
                + (f"the GPUs are {found}." if names else "CUDA sees no NVIDIA GPU.")
            )
    out = args.out or args.input.parent
    onnx_file = None
    targets = []
    if not from_onnx and (args.onnx or not args.engine):
        shape = "dynamic" if profile is not None else size_text(args.size)
        onnx_file = out / output_name(model, shape, routing_na, ".onnx")
        targets.append(onnx_file)
    if write_engine:
        import tensorrt

        shape = (
            size_text(args.size)
            if profile is None
            else "-".join(size_text(size) for size in profile)
        )
        engine_file = out / output_name(
            model, shape, routing_na, f"_trt{tensorrt.__version__}.engine"
        )
        # The engine stays under the working name until its check has an outcome.
        working = engine_file.with_name(engine_file.name + ".unchecked")
        failed = engine_file.with_name(engine_file.name + ".failed")
        sidecar = engine_file.with_name(engine_file.name + ".json")
        targets += [engine_file, sidecar, working, failed]
        if args.test:
            png = engine_file.with_name(f"{engine_file.stem}_{args.test.stem}.png")
            targets.append(png)
    existing = [target for target in targets if target.exists()]
    if existing and not args.overwrite:
        raise UserError(
            f"{existing[0]} exists: pass --overwrite to replace it, remove it, or choose"
            " --out."
        )
    # With --overwrite every earlier output of this run goes now, after the checks and
    # before the long work, so that none stays next to new results.
    for target in existing:
        try:
            target.unlink()
        except OSError as error:
            raise UserError(
                f"{target} cannot be removed for --overwrite ({error.strerror}); close"
                " the program that holds it and run again."
            ) from error
    if not from_onnx:
        print("Exporting the ONNX ...", flush=True)
        onnx_bytes = export_onnx(
            args.input.resolve(),
            factory,
            scale,
            args.size if profile is None else None,
            state_key=state_key,
            folded=folded,
            routing_na=routing_na,
        )
        manifest = read_manifest(onnx_bytes, "the export")
        if onnx_file is not None:
            out.mkdir(parents=True, exist_ok=True)
            onnx_file.write_bytes(onnx_bytes)
            print(f"ONNX: {onnx_file}")
    if not write_engine:
        return
    print(
        "Building the engine (the plugins' kernels are compiled for this GPU; this"
        " takes a minute or more) ...",
        flush=True,
    )
    try:
        engine_bytes = build_engine(
            onnx_bytes, manifest, args.gpu, profile, verbose=args.verbose
        )
        out.mkdir(parents=True, exist_ok=True)
        working.write_bytes(engine_bytes)
    except Exception:
        if onnx_file is not None:
            print(
                f"The engine build failed; the ONNX {onnx_file} is complete. Build its"
                f" engine with: dual-convert {onnx_file} (with the size options), or run"
                " this command again with --overwrite.",
                file=sys.stderr,
            )
        raise
    check = None
    if from_onnx:
        print(
            "Self-check: not run; it needs the checkpoint (an .onnx input has no model"
            " to compare with)."
        )
    elif args.no_verify:
        print("Self-check: skipped (--no-verify).")
    else:
        check = self_check(
            working, args, profile, factory, scale, state_key, folded, routing_na
        )
        if not check["passed"]:
            working.replace(failed)
            raise UserError(
                f"self-check FAILED: {check['failure']} The engine is left as {failed}"
                " so that nothing loads it. Build it again; if this repeats, report it"
                " with the TensorRT version, the GPU and the model's preset."
            )
    sidecar.write_text(
        json.dumps(
            sidecar_contents(
                manifest, profile, tensorrt.__version__, gpu_name(args.gpu), check
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    working.replace(engine_file)
    print(f"Engine: {engine_file}")
    print(f"Sidecar: {sidecar}")
    if args.test:
        run_child("test", engine_file, args, args.test.resolve(), png)