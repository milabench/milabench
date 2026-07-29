def torchmem_fetcher(device=None):
    """Return a callable that reports allocated/reserved MiB per CUDA/ROCm device.

    Uses ``torch.cuda`` memory APIs (ROCm also exposes itself via ``torch.cuda``).
    Each device entry includes ``allocated``, ``reserved``, ``max_allocated``,
    and ``max_reserved``. Returns ``{}`` when CUDA/ROCm is unavailable.
    """
    try:
        import torch
    except Exception:
        return lambda: {}

    if not hasattr(torch, "cuda"):
        return lambda: {}

    def fetch():
        try:
            if not torch.cuda.is_available():
                return {}

            # CUDA and ROCm both use the torch.cuda namespace
            if not (torch.version.cuda or getattr(torch.version, "hip", None)):
                return {}

            if device is not None:
                devices = [device]
            else:
                devices = range(torch.cuda.device_count())

            result = {}
            for i in devices:
                result[i] = {
                    "allocated": torch.cuda.memory_allocated(i) / (1024**2),
                    "reserved": torch.cuda.memory_reserved(i) / (1024**2),
                    "max_allocated": torch.cuda.max_memory_allocated(i) / (1024**2),
                    "max_reserved": torch.cuda.max_memory_reserved(i) / (1024**2),
                }
            return result
        except Exception:
            return {}

    return fetch
