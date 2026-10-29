import os

# Smoke tests are tiny; keep them off a GPU that experiments may be using.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
