"""The installed packages' versions and the NVIDIA GPUs, for the installer's last step:

    python -m dual_convert.environment
"""

from __future__ import annotations

from importlib import metadata

PACKAGES = (
    "torch",
    "numpy",
    "onnx",
    "safetensors",
    "pillow",
    "triton-windows",
    "cuda-bindings",
    "tensorrt_cu13_bindings",
    "tensorrt_cu13_libs",
)


def main() -> None:
    import numpy
    import onnx
    import PIL
    import safetensors
    import tensorrt
    import torch
    import triton
    from cuda.bindings import runtime as cudart

    for module in (numpy, onnx, PIL, safetensors, tensorrt, torch, triton):
        print(f"import {module.__name__}: ok")
    for package in PACKAGES:
        print(f"{package} {metadata.version(package)}")
    error, count = cudart.cudaGetDeviceCount()
    if error != cudart.cudaError_t.cudaSuccess or count == 0:
        raise SystemExit(
            f"No NVIDIA GPU is usable ({error}): install or update the NVIDIA driver."
        )
    for index in range(count):
        error, properties = cudart.cudaGetDeviceProperties(index)
        if error != cudart.cudaError_t.cudaSuccess:
            raise SystemExit(f"GPU {index} cannot be read: {error}")
        name = properties.name.split(b"\0", 1)[0].decode()
        print(f"GPU {index}: {name} (compute {properties.major}.{properties.minor})")


if __name__ == "__main__":
    main()
