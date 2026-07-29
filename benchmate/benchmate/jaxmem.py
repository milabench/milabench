def jaxmem_fetcher(device=None):
    """Return a callable that reports allocated/reserved MiB per JAX device.

    Uses ``device.memory_stats()``. Each device entry includes ``allocated``,
    ``reserved``, ``max_allocated``, and ``max_reserved`` (torchmem-aligned
    names mapped from JAX ``bytes_in_use`` / ``peak_bytes_*`` fields).
    Returns ``{}`` when JAX is unavailable or stats cannot be read.
    """
    try:
        import jax
    except Exception:
        return lambda: {}

    def fetch():
        try:
            devices = jax.devices()
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
