"""Runs a built engine in a process of its own, with no plugin registered (so a run
proves the engine holds its kernels): TensorRT, cuda.bindings and numpy only.

    python -m dual_convert.runner check ENGINE GPU VERBOSE INPUT.npy OUTPUT.npy
    python -m dual_convert.runner test ENGINE GPU VERBOSE IMAGE OUTPUT.png

check: one FP32 NCHW input in, the engine's output written as .npy.
test: an RGB image upscaled into a PNG, and the engine's time per image (20 warm-ups,
then the median of 50 CUDA-graph replays)."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from .engine import logger
from .image import read_image
from .sizes import padded, size_text


def ok(result: tuple[Any, ...]) -> Any:
    """The value of a cuda.bindings call, which returns (error, *values)."""
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error {result[0]}")
    return result[1] if len(result) == 2 else None


class Session:
    """One engine with its context, stream and device buffers for one input shape."""

    def __init__(self, path: Path, gpu: int, verbose: bool) -> None:
        ok(cudart.cudaSetDevice(gpu))
        self.runtime = trt.Runtime(logger(verbose))
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise SystemExit(
                f"{path.name} could not be loaded; TensorRT's error is above. One"
                " possible cause: an engine loads only in the TensorRT version that"
                f" built it (this is {trt.__version__}) and on the GPU model it was"
                " built on; a damaged or incomplete file is another."
            )
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.output = next(n for n in names if n != self.input)
        self.stream = ok(cudart.cudaStreamCreate())
        self.graph = None

    def bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """The smallest and largest (height, width) the engine takes."""
        shape = tuple(self.engine.get_tensor_shape(self.input))
        if -1 not in shape:
            return shape[2:], shape[2:]
        low, _, high = self.engine.get_tensor_profile_shape(self.input, 0)
        return tuple(low[2:]), tuple(high[2:])

    def run(self, x: np.ndarray) -> np.ndarray:
        """The engine's output for one contiguous FP32 [1, 3, H, W] input."""
        self.context.set_input_shape(self.input, x.shape)
        y = np.empty(tuple(self.context.get_tensor_shape(self.output)), np.float32)
        self.x_device = ok(cudart.cudaMalloc(x.nbytes))
        self.y_device = ok(cudart.cudaMalloc(y.nbytes))
        ok(
            cudart.cudaMemcpy(
                self.x_device,
                x.ctypes.data,
                x.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        self.context.set_tensor_address(self.input, int(self.x_device))
        self.context.set_tensor_address(self.output, int(self.y_device))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT failed to run the engine.")
        ok(cudart.cudaStreamSynchronize(self.stream))
        ok(
            cudart.cudaMemcpy(
                y.ctypes.data,
                self.y_device,
                y.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        return y

    def replay_ms(self) -> float:
        """The median time of 50 replays of the last run's shape, captured in a CUDA
        graph, after 20 warm-ups."""
        mode = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        ok(cudart.cudaStreamBeginCapture(self.stream, mode))
        launched = self.context.execute_async_v3(int(self.stream))
        graph = ok(cudart.cudaStreamEndCapture(self.stream))
        if not launched:
            raise RuntimeError("TensorRT failed to run the engine in a CUDA graph.")
        replay = ok(cudart.cudaGraphInstantiate(graph, 0))
        for _ in range(20):
            ok(cudart.cudaGraphLaunch(replay, self.stream))
        ok(cudart.cudaStreamSynchronize(self.stream))
        start, end = ok(cudart.cudaEventCreate()), ok(cudart.cudaEventCreate())
        times = []
        for _ in range(50):
            ok(cudart.cudaEventRecord(start, self.stream))
            ok(cudart.cudaGraphLaunch(replay, self.stream))
            ok(cudart.cudaEventRecord(end, self.stream))
            ok(cudart.cudaEventSynchronize(end))
            times.append(ok(cudart.cudaEventElapsedTime(start, end)))
        return statistics.median(times)


def test(session: Session, image_path: Path, png: Path) -> None:
    from PIL import Image

    pixels, note = read_image(image_path)
    if note:
        print(note)
    size = pixels.shape[:2]
    low, high = session.bounds()
    if low == high:
        if size != low:
            raise SystemExit(
                f"{image_path.name} is {size_text(size)}, but this engine takes only"
                f" {size_text(low)} images (a fixed engine): resize or crop the image,"
                " or build an engine for its size or a size range."
            )
    elif not all(a <= b <= c for a, b, c in zip(low, padded(size), high, strict=True)):
        raise SystemExit(
            f"{image_path.name} is {size_text(size)} (padded to"
            f" {size_text(padded(size))}), outside this engine's range"
            f" {size_text(low)} to {size_text(high)}."
        )
    height, width = size
    pad_height, pad_width = padded(size)
    x = pixels.transpose(2, 0, 1)[None]
    x = np.pad(
        x, ((0, 0), (0, 0), (0, pad_height - height), (0, pad_width - width)), "reflect"
    )
    y = session.run(np.ascontiguousarray(x))
    scale = y.shape[2] // pad_height
    y = y[0, :, : height * scale, : width * scale]
    out = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(out).save(png)
    print(f"Test image: {png}")
    print(
        f"Engine time: {session.replay_ms():.2f} ms per {size_text((pad_height, pad_width))}"
        " image (median of 50 CUDA-graph replays after 20 warm-ups; reading and writing"
        " the image not included)"
    )


def main(mode: str, engine: str, gpu: str, verbose: str, source: str, target: str) -> None:
    session = Session(Path(engine), int(gpu), verbose == "1")
    if mode == "check":
        x = np.ascontiguousarray(np.load(source), dtype=np.float32)
        np.save(target, session.run(x))
    else:
        test(session, Path(source), Path(target))


if __name__ == "__main__":
    main(*sys.argv[1:])
