from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(torch.cuda.get_arch_list())
    if arch not in supported_arches:
        print(
            f"CUDA device capability {arch} is not supported by this PyTorch build "
            f"({sorted(supported_arches)}). Falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")
