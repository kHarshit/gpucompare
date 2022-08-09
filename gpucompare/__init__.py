"""Compare GPUs"""

import sys
from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

# params to compare
spec_params: list[str] = [
    "architecture",
    "mem_bandwidth",
    "cuda_cores",
    "fp32_perf",
    "int8_perf",
    "fp16_perf",
]
# architecture speedup in increasing order
arch_list: list[str] = ["volta", "turing", "ampere", "hopper"]
# ampere_w.r.t._volta will have speedup of 0.05+0.05
# volta_w.r.t._ampere will have speedup of -0.05-0.05
arch_speedup: list[float] = [0.05, 0.05, 0.05]
