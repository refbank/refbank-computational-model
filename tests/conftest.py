import os
import pathlib

# Point JAX at a project-local disk cache so XLA doesn't recompile on every run.
# JAX reads this env var on first initialization, so it must be set before any
# test module imports jax/numpyro.
_xla_cache = pathlib.Path(__file__).parent.parent / ".xla_cache"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_xla_cache))

collect_ignore_glob = []
