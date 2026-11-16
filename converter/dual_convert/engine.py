"""A TensorRT engine of a DUAL ONNX: its plugin nodes lowered to AOT Python plugins
(plugins/dual_aot.py), whose Triton kernels are compiled for the building GPU and
embedded in the engine, so the engine loads without them. The builder policy: strongly
typed, an 8 GiB workspace, optimization level 3, TF32 off, dense weights, no tiling
optimization, no auxiliary streams, detailed profiling verbosity and a fresh timing
cache. A fixed ONNX keeps its size; a dynamic one gets one optimization profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tensorrt as trt

WORKSPACE_GIB = 8
OPTIMIZATION_LEVEL = 3
# The exported graph's output tensor
OUTPUT = "output"


def input_name(spec: dict[str, Any]) -> str:
    """An engine's input name: input_aligned_<A>, A the multiple its sizes keep (a
    fixed ONNX's manifest has no alignment: its size is a multiple of 4)."""
    return f"input_aligned_{spec.get('alignment', 4)}"


def logger(verbose: bool) -> trt.ILogger:
    """TensorRT's messages: errors always, warnings and information with `verbose`."""
    import tensorrt as trt

    shown = trt.ILogger.Severity.INFO if verbose else trt.ILogger.Severity.ERROR

    class Logger(trt.ILogger):
        def __init__(self) -> None:
            trt.ILogger.__init__(self)

        def log(self, severity: trt.ILogger.Severity, message: str) -> None:
            if severity <= shown:
                print(f"TensorRT {severity.name.lower()}: {message}", flush=True)

    return Logger()


def build_engine(
    onnx_bytes: bytes,
    manifest: dict[str, Any],
    gpu: int,
    profile: tuple[tuple[int, int], ...] | None,
    *,
    verbose: bool,
) -> bytes:
    """The serialized engine of a DUAL ONNX on GPU `gpu`. A dynamic ONNX takes its
    (minimum, optimal, maximum) (height, width) profile; a fixed one None. The input is
    named input_aligned_<alignment>, the multiple its sizes keep."""
    import onnx
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    from .plugins import dual_aot

    spec = manifest["specialization"]
    key = spec["plugin_key"]
    dynamic = "_h" not in key
    if dynamic != (profile is not None):
        raise ValueError("Only a dynamic DUAL ONNX takes a size profile.")
    (error,) = cudart.cudaSetDevice(gpu)
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"GPU {gpu} cannot be used: {error}")
    lowered = dual_aot.lower(onnx.load_from_string(onnx_bytes), key)
    if dynamic:
        dual_aot.register_dynamic(int(key[1:]))
    else:
        dual_aot.register(key)
    log = logger(verbose)
    builder = trt.Builder(log)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, log)
    if not parser.parse(lowered.model.SerializeToString()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT could not read the ONNX:\n" + "\n".join(errors))
    dual_aot.attach(network, lowered)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = OPTIMIZATION_LEVEL
    config.tiling_optimization_level = trt.TilingOptimizationLevel.NONE
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GIB << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_timing_cache(config.create_timing_cache(b""), ignore_mismatch=False)
    image = network.get_input(0)
    image.name = input_name(spec)
    if profile is not None:
        optimization = builder.create_optimization_profile()
        optimization.set_shape(image.name, *((1, 3, h, w) for h, w in profile))
        config.add_optimization_profile(optimization)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the engine (see its errors above).")
    return bytes(engine)


def gpu_names() -> list[str]:
    """The names of the NVIDIA GPUs CUDA sees, in index order."""
    from cuda.bindings import runtime as cudart

    error, count = cudart.cudaGetDeviceCount()
    if error != cudart.cudaError_t.cudaSuccess:
        return []
    return [gpu_name(index) for index in range(count)]


def gpu_name(gpu: int) -> str:
    from cuda.bindings import runtime as cudart

    error, properties = cudart.cudaGetDeviceProperties(gpu)
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"GPU {gpu} cannot be read: {error}")
    return properties.name.split(b"\0", 1)[0].decode()
