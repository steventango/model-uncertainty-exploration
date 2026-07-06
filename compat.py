def patch_brax_fluid():
    """Fix brax 0.14.2 / JAX ≥0.5 clip API incompatibility.

    brax.fluid uses jp.clip(array, a_min=...) but JAX renamed the kwargs to
    min=/max=. Patch the module-level binding before the first brax env step.
    """
    import jax.numpy as jnp
    import brax.fluid as _f

    _orig = jnp.clip

    def _clip(a, a_min=None, a_max=None, **kw):
        if a_min is not None:
            kw["min"] = a_min
        if a_max is not None:
            kw["max"] = a_max
        return _orig(a, **kw)

    _f.jp.clip = _clip
