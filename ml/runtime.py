"""CPU-only runtime settings shared by notebooks, training, and inference."""

import json
import os
import platform
import random

import numpy as np
import psutil
import torch


def configure(seed: int = 42, threads: int = 2) -> dict:
    """Bound CPU parallelism and seed all random generators without probing accelerators."""
    threads = max(1, min(threads, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return {
        "device": "cpu",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "logical_cpus": os.cpu_count(),
        "compute_threads": threads,
        "available_ram_gib": round(psutil.virtual_memory().available / 1024**3, 2),
        "seed": seed,
    }


if __name__ == "__main__":
    print(json.dumps(configure(), indent=2))
