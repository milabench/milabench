def _device_namespace():
    """Return the unified accelerator namespace via torchcompat.

    ``torchcompat.core.device_module`` is the raw device namespace for the
    available accelerator (``torch.cuda`` on CUDA/ROCm, ``torch.xpu`` on Intel
    GPU), so the allocator API is identical across all of them. ``None`` when
    torchcompat has no accelerator (e.g. CPU only).
    """
    try:
        import torchcompat.core as accelerator
    except Exception:
        return None
    return getattr(accelerator, "device_module", None)


def torchmem_fetcher(device=None):
    """Return a callable that reports allocated/reserved MiB per accelerator.

    CUDA, ROCm and Intel XPU share a single code path through torchcompat. Each
    device entry includes ``allocated``, ``reserved``, ``max_allocated`` and
    ``max_reserved``. Returns ``{}`` when no accelerator exposes memory stats.
    """

    def fetch():
        try:
            namespace = _device_namespace()
            if namespace is None or not hasattr(namespace, "memory_allocated"):
                return {}

            if device is not None:
                devices = [device]
            else:
                devices = range(namespace.device_count())

            result = {}
            for i in devices:
                result[i] = {
                    "allocated": namespace.memory_allocated(i) / (1024**2),
                    "reserved": namespace.memory_reserved(i) / (1024**2),
                    "max_allocated": namespace.max_memory_allocated(i) / (1024**2),
                    "max_reserved": namespace.max_memory_reserved(i) / (1024**2),
                }
            return result
        except Exception:
            return {}

    return fetch
