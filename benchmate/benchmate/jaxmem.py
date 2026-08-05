class JaxGpuRequiredError(RuntimeError):
    """Raised when the voir GPU arch requires a device JAX is not using."""


# Architectures that do not require a JAX accelerator backend.
_NON_GPU_ARCHES = frozenset({"cpu", "mock", ""})

# Substrings used to recognize a JAX device as belonging to a voir GPU arch.
_ARCH_DEVICE_MARKERS = {
    "rocm": ("rocm", "hip", "amd"),
    "cuda": ("cuda", "nvidia"),
    "xpu": ("xpu", "intel"),
    "hpu": ("hpu", "habana"),
}


def _voir_gpu_arch():
    """Return the active voir GPU arch, or ``None`` when unavailable/CPU."""
    try:
        from voir.instruments.gpu import select_backend

        arch = getattr(select_backend(), "arch", None)
    except Exception:
        return None

    if arch is None:
        return None
    arch = str(arch).strip().lower()
    if arch in _NON_GPU_ARCHES:
        return None
    return arch


def _device_identity(device):
    return " ".join(
        str(part).lower()
        for part in (
            type(device).__name__,
            repr(device),
            getattr(device, "platform", ""),
            getattr(device, "device_kind", ""),
        )
    )


def _device_matches_arch(device, arch):
    """Return True if a JAX device looks like it belongs to ``arch``."""
    identity = _device_identity(device)
    markers = _ARCH_DEVICE_MARKERS.get(arch)
    if markers:
        return any(marker in identity for marker in markers)

    # Unknown GPU arch: accept any non-CPU accelerator device.
    platform = getattr(device, "platform", None)
    return platform not in (None, "cpu")


def _ensure_jax_matches_arch(jax, devices, arch):
    """Raise if voir reports a GPU arch but JAX is not using that device type."""
    matching = [d for d in devices if _device_matches_arch(d, arch)]
    if matching:
        return

    try:
        backend = jax.default_backend()
    except Exception:
        backend = None

    raise JaxGpuRequiredError(
        f"voir GPU arch {arch!r} requires a matching JAX device, but JAX is "
        f"not using one (backend={backend!r}, devices={list(devices)!r}). "
        f"Install a GPU-enabled JAX build for this arch "
        f"(e.g. jax-rocm7-plugin / jax[cuda])."
    )


def jaxmem_fetcher(device=None):
    """Return a callable that reports allocated/reserved MiB per JAX device.

    Uses ``device.memory_stats()``. Each device entry includes ``allocated``,
    ``reserved``, ``max_allocated``, and ``max_reserved`` (torchmem-aligned
    names mapped from JAX ``bytes_in_use`` / ``peak_bytes_*`` fields).
    Returns ``{}`` when JAX is unavailable or stats cannot be read.

    When ``voir.instruments.gpu.select_backend().arch`` is a GPU arch, raises
    :class:`JaxGpuRequiredError` if JAX has no device matching that arch.
    """
    gpu_arch = _voir_gpu_arch()

    try:
        import jax
    except Exception as exc:
        if gpu_arch is None:
            return lambda: {}

        import_error = exc

        def fetch_missing_jax():
            raise JaxGpuRequiredError(
                f"voir GPU arch {gpu_arch!r} requires JAX, but importing "
                f"jax failed: {import_error}"
            ) from import_error

        return fetch_missing_jax

    def fetch():
        try:
            devices = jax.devices()
            if gpu_arch is not None:
                _ensure_jax_matches_arch(jax, devices, gpu_arch)

            if not devices:
                return {}

            if device is not None:
                selected = [devices[device]]
                indices = [device]
            else:
                selected = devices
                indices = range(len(devices))

            result = {}
            for i, dev in zip(indices, selected):
                stats = dev.memory_stats()
                if stats is None:
                    continue
                result[i] = {
                    "allocated": stats.get("bytes_in_use", 0) / (1024**2),
                    "reserved": stats.get("bytes_reserved", 0) / (1024**2),
                    "max_allocated": stats.get("peak_bytes_in_use", 0) / (1024**2),
                    "max_reserved": stats.get("peak_bytes_reserved", 0) / (1024**2),
                }
            return result
        except JaxGpuRequiredError:
            raise
        except Exception:
            return {}

    return fetch


def memory_peak_fetcher():
    """Return a callable that reports the max ``peak_bytes_in_use`` across devices (MiB)."""
    fetch = jaxmem_fetcher()

    def fetch_memory_peak():
        data = fetch()
        if not data:
            return -1
        return max(entry["max_allocated"] for entry in data.values())

    return fetch_memory_peak
