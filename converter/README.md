# dual-convert

dual-convert turns a DUAL image upscaling model, trained with traiNNer, into files for
NVIDIA TensorRT:

- the **TensorRT-only ONNX**: the model with DUAL's TensorRT plugin nodes, which only this
  tool's engine build reads (other ONNX programs cannot run it);
- a **TensorRT engine** with DUAL's GPU kernels built in, ready to load in any program that
  uses TensorRT, with no plugin library;
- or both in one command.

It runs on Windows with an NVIDIA graphics card. After each build from a checkpoint, the
engine is run once and compared with the PyTorch model built from the same options: this
shows that the engine computes that model on your PC. It cannot show whether the options
(such as `--routing-na`) match how the model was trained.

## What you need

- Windows 10 or 11, 64-bit.
- An NVIDIA graphics card and a current NVIDIA driver (the engines are built for the card
  in your PC).
- 64-bit Python 3.14, 3.13 or 3.12 from https://www.python.org/downloads/windows/ (the
  installer looks for them in this order). Tested: Python 3.14.
- About 3.5 GB of free disk space for the tool's `venv` folder, and an internet connection
  during the installation (about 2.3 GB is downloaded).
- RAM: the check after each build runs the model on the processor. For a 2x DUAL Light it
  needs about 3.5 GB at 960x540 and about 8.5 GB at 1920x1080; larger presets need more.
  The tool estimates this before the check and compares it with the free memory: a range
  engine is checked at `--opt`, or at a smaller size of the same shape whose check fits in
  the free memory and takes at most about 30 seconds (the size is printed); a fixed engine
  is checked at its own size however long it takes, and if it is too large for the free
  memory it is only run once and checked for finite output. When the check will take
  longer than about 20 seconds, the tool says so first.

## Install

1. Download the repository (on GitHub: Code, Download ZIP), unpack it and put its
   `converter` folder where you want to keep it (a path with spaces is fine).
2. Double-click `install venv.bat`. It asks which TensorRT version to use: type 1, 2 or 3,
   or press Enter for 11.4.0.106. It creates the folder `venv` here and installs
   everything into it; nothing is installed into your Python.

To install without the question, give the full version number (not 1, 2 or 3):
`"install venv.bat" 11.2.1.2`. The window then closes by itself when it is done.
Running it again with another version switches the TensorRT version of the existing
`venv`. The installation ends with a check that lists the installed versions and your
graphics card.

### TensorRT versions

| Version | |
|---|---|
| 11.4.0.106 | the default |
| 11.3.0.99 | |
| 11.2.1.2 | |

Choose the version of the program that will load your engines. **An engine only loads in
the TensorRT version that built it and on the GPU model it was built on**: TensorRT picks
and stores kernels for that version and that card when it builds. For another version or
another card, build the engine again (from the same ONNX file if you kept it).

The tool installs TensorRT's CUDA 13 build (`tensorrt-cu13`). The CUDA build of the program
that loads the engine does not have to match: an engine built here also loads and runs in
the CUDA 12 build of the same TensorRT version, and the other way round (tested with
11.4.0.106).

## Use

Double-click `run.bat`. It opens a command prompt in which `dual-convert` works from any
folder. The prompt starts in the tool's folder: change to your model's folder first
(`cd /d D:\models`), or type the model's full path in quotes; dragging a file into the
window pastes its path. Then, for example:

```
dual-convert model.safetensors --size 960x540
dual-convert net_g_ema.pth --min 256x256 --opt 1280x720 --max 1920x1088
dual-convert model_dynamic_standard.onnx --min 256x256 --opt 1920x1088 --max 1920x1088
dual-convert model.safetensors --size 960x540 --test photo.png
dual-convert -help
```

The first line writes `model_960x540_standard.onnx` and
`model_960x540_standard_trt11.4.0.106.engine` with its `.json` file next to the model.
Building an engine takes a minute or more.

### Sizes (always required)

Sizes are written WIDTHxHEIGHT in pixels, both multiples of 4.

- `--size WxH`: a **fixed** engine for this one input size. It is the fastest form; it
  takes only images of exactly this size.
- `--min WxH --opt WxH --max WxH`: an engine for a **size range**. It takes any size from
  `--min` to `--max` and is tuned for `--opt`. From a checkpoint this writes a dynamic ONNX,
  which takes any size; the range is still required, also with `--onnx` alone, because it
  is checked against the model's limits.
- From an `.onnx` file: a fixed ONNX carries its size (give no size, or the same one); a
  dynamic ONNX needs `--min --opt --max`.

Rules: `--min` <= `--opt` <= `--max` in width and in height, and the largest size must fit
the kernels' 32-bit indexing (see Limits).

### Options

| Option | Meaning |
|---|---|
| `<input>` | a checkpoint (`.safetensors` or `.pth`) or an `.onnx` file written by this tool |
| `--onnx` | write the ONNX |
| `--engine` | build the engine (neither option: both; from an `.onnx` file: the engine) |
| `--size WxH` | one fixed input size |
| `--min WxH`, `--opt WxH`, `--max WxH` | a size range: smallest, tuned-for and largest size |
| `--routing-na` (or `--high-quality`) | the expert routing per position, for models trained with `routing_na: true` |
| `--preset light\|xs\|s\|m\|l\|xl` | the preset, if the one read from the weights is wrong |
| `--scale 1\|2\|4` | the scale, if the one read from the weights is wrong |
| `--state-key KEY` | the entry of a `.pth` file that holds the weights |
| `--out FOLDER` | where to write (default: the input's folder) |
| `--overwrite` | replace this run's existing outputs (ONNX, engine, its .json, the test picture, a leftover .unchecked or .failed engine) |
| `--test IMAGE` | after the build, upscale IMAGE with the engine, write a PNG next to the engine and print the engine's time per image |
| `--no-verify` | skip the check after the build |
| `--gpu INDEX` | the graphics card to use (default 0, the first) |
| `--verbose` | also show TensorRT's warnings and information |
| `-h`, `-help`, `--help` | the full help |

Every option can be written with one or two dashes (`-size` or `--size`).

The preset (Light, XS, S, M, L, XL) and the scale (1, 2, 4) are read from the weights. A
`.pth` file is read from its `params_ema` weights, else `params`, else the file itself;
the tool prints which.

### --routing-na

A DUAL model can be trained with `routing_na: true`: its convolutions then choose their
experts for each position from the neighborhood around it, instead of from the whole
image's mean. The checkpoint does not record this, so you have to say it: pass
`--routing-na` for such models. Without it, such a model's engine still runs but routes
from the mean and its output differs from the trained model's. The routed engine is
somewhat slower. The check after the build uses the same option for the engine and the
model, so it cannot tell whether `--routing-na` was right for your model: only you know
how it was trained.

### --test

`--test IMAGE` reads the image as RGB: colour pictures at 8 bits per channel (a 16-bit
colour picture is read at 8 bits), 16-bit grey pictures at full precision; transparency is
dropped, and a line says so. It runs the image through the new engine and writes
`<engine name>_<image name>.png` next to the engine. A fixed engine takes only images of
its own size. A range engine takes any size inside its range; a size that is not a
multiple of 4 is padded by reflection and cropped back. The picture is checked before the
export and the build. The result is an 8-bit RGB PNG. The printed time is the engine's
alone (a CUDA-graph replay); reading and writing the image is not included.

### Output names

`<model>_<size>_<form>.onnx` and `<model>_<size or range>_<form>_trt<version>.engine`,
where `<form>` is `standard` or `routing-na` and a range is written
`<min>-<opt>-<max>`. An existing file is not overwritten unless `--overwrite` is given;
the command stops before any work. With `--overwrite`, this run's existing outputs are
removed after the checks of the options and before the long work, so no earlier file is
left next to new results. An engine is written as `<name>.engine.unchecked`
and gets its final name, next to its `.json`, only when its check has an outcome.

## Using an engine in your own program

The `.json` file next to every engine lists what follows, for that engine.

- **Input**: the tensor named `input_aligned_4`, FP32, layout NCHW, batch 1, RGB, values
  from 0 to 1. Width and height are multiples of 4: pad other sizes (reflection works
  well) and crop the output. A fixed engine takes exactly its size; a range engine any size
  in its range (set the input shape before each run).
- **Output**: the tensor named `output`, FP32, NCHW, RGB, scale times the input's width and
  height. Values are about 0 to 1; clip before converting to 8 bits.
- The engine needs no plugin library: load it with `IRuntime.deserializeCudaEngine` in the
  same TensorRT version.
- **Replay it as a CUDA graph.** The engine runs many small kernels, and every launch costs
  time on the processor. Run it once for a shape, capture one `enqueueV3` /
  `execute_async_v3` on your stream in a CUDA graph (`cudaStreamBeginCapture` ...
  `cudaStreamEndCapture`, `cudaGraphInstantiate`), and from then on launch the graph
  (`cudaGraphLaunch`) for every image of that shape. Capture again when the shape or the
  buffers change.

## Limits

- Only DUAL image models (presets Light, XS, S, M, L, XL; scale 1, 2 or 4). RDUAL (the
  video model) is not supported.
- Models trained with `regional_na` are not supported: the engine has no per-position form
  of the regional channel attention.
- Models must have the standard geometry of the presets (dot attention, the default
  context windows).
- Size bound: an engine's largest size must keep 3 x W x H x channels below 2^31 (with W
  and H halved for Light, and more than 3 parts with `--routing-na` when the model has more
  than 3 experts). Channels: Light and XS 128, S 160, M 192, L 224, XL 256. For example a
  DUAL M takes at most about 1920x1920 per engine; tile larger images.

## Messages

| Message | Meaning |
|---|---|
| `a size is required` / `a size range needs all of --min, --opt and --max` | type `--size WxH` or all three range options |
| `... is not a positive multiple of 4` | width and height must be multiples of 4 |
| `... is larger than ...` | the range is not ordered: `--min` <= `--opt` <= `--max` in both directions |
| `... is too large for this model` | the size breaks the size bound (see Limits); choose a smaller size |
| `The file's weights are not a DUAL image model` | the checkpoint holds another kind of model |
| `the weights match none of the six presets` | a DUAL model of an unusual size; `--preset` can force one |
| `The weights do not fit DUAL ... (check --preset and --scale)` | a forced `--preset` or `--scale` does not match the weights |
| `... cannot be read as safetensors` / `as a .pth checkpoint` / `as ONNX` | the file is damaged, incomplete or of another kind |
| `... has no state dict under 'KEY'; its keys: ...` | choose one of the listed keys with `--state-key` |
| `... holds no state dict of tensors` | the .pth file's weights are under another entry: choose it with `--state-key` |
| `the test picture ... does not exist` / `cannot be read` / `is WxH, but a fixed engine ...` / `outside the range` | the `--test` picture is checked before any work; fix the path or the size |
| `... is a .safetensors file, which holds one state dict: omit --state-key` | `--state-key` applies to .pth files only |
| `... exists: pass --overwrite to replace it` | an output of this run is already there; nothing was written |
| `... cannot be removed for --overwrite` | another program holds that file open; close it and run again |
| `The engine build failed; the ONNX ... is complete` | the ONNX is fine: build its engine from the .onnx file, or run again with `--overwrite` |
| `is not a DUAL ONNX written by dual-convert` | only this tool's ONNX files can be built into engines |
| `this ONNX is fixed at ...` / `this ONNX is dynamic ...` | a fixed ONNX keeps its size; a dynamic one needs the range |
| `Self-check: passed, relative L2 difference ... (limit ... at WxH)` | the engine's output matches the model's (the engine computes in BF16, so a small difference is normal; it grows with the size, and so does the accepted limit) |
| `self-check FAILED ... left as ....engine.failed` | the engine does not compute the PyTorch model of the same options on this PC; it was renamed so that nothing loads it by accident. Build it again; if it repeats, report it with the TensorRT version, the GPU and the preset |
| a file ending in `.engine.unchecked` | a run that stopped before its check ended (for example Ctrl+C); do not use it, run again with `--overwrite` |
| `Self-check: at WxH, the largest size up to --opt whose model on the CPU fits in the free memory ... and takes at most about 30 s` | the range engine is checked at this smaller size, because at `--opt` the check would not fit in the free memory or would take longer |
| `Self-check: comparison skipped (the model on the CPU would need about ... GiB ...)` | a fixed engine too large for the free memory: it was loaded and run once and gave finite values, but not compared with the model; this is not an error. Close other programs and build again for a full check |
| `Self-check: not run; it needs the checkpoint` | from an `.onnx` file there is no model to compare with |
| `... could not be loaded; TensorRT's error is above` | read TensorRT's error: often another TensorRT version or GPU model built the engine (build it again there), or the file is damaged |
| `... is WxH, but this engine takes only ...` / `... outside this engine's range` | the test picture must have the fixed engine's size, or fit the range engine's sizes |
| `GPU N does not exist; the GPUs are ...` | choose an index from the list with `--gpu` (checked before any work) |
| `GPU N cannot be used` | CUDA refused that graphics card; check the NVIDIA driver |
| `--size ... contradicts this ONNX's fixed size ...` | a fixed ONNX keeps its size: omit `--size` |
| `Functionality provided through tensorrt.plugin module is experimental.` | a notice TensorRT always prints when the kernels are built; it can be ignored |
| `TensorRT error: ...` | TensorRT's own error; the lines around it say what failed |

## Files in this folder

- `dual_convert\`: the tool.
- `install venv.bat`, `requirements.txt`: the installation.
- `run.bat`: the command prompt with the tool ready.
- `venv\`: created by the installer.
- `cache\`: created on the first build; it holds the compiled kernels so that later builds
  are faster. It can be deleted at any time.

## License

MIT; see `LICENSE` in the repository's root, which also holds the notices retained for
parts of the model file.
