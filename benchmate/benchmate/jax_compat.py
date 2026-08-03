"""JAX API compatibility helpers for milabench benchmarks."""

from __future__ import annotations


def _jax_attr_available(jax_mod, name: str) -> bool:
    """Return True if ``getattr(jax, name)`` works (not deprecated/removed)."""
    try:
        getattr(jax_mod, name)
        return True
    except AttributeError:
        return False


def _device_put_replicated(x, devices):
    """Drop-in for ``jax.device_put_replicated`` (jax migrate_pmap docs)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x
    )


def _device_put_sharded(shards, devices):
    """Drop-in for ``jax.device_put_sharded`` (jax migrate_pmap docs)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards
    )


def ensure_device_put_compat() -> bool:
    """Restore deprecated ``jax.device_put_{replicated,sharded}`` if needed.

    Newer JAX raises ``AttributeError`` on these names. Older brax/flax stacks
    still call them. When missing/deprecated, install the public drop-in
    replacements from the JAX pmap migration guide.

    Returns:
        True if any shim was installed, False if native APIs were already usable.
    """
    import jax

    patched = False
    if not _jax_attr_available(jax, "device_put_replicated"):
        jax.device_put_replicated = _device_put_replicated
        patched = True
    if not _jax_attr_available(jax, "device_put_sharded"):
        jax.device_put_sharded = _device_put_sharded
        patched = True
    return patched
