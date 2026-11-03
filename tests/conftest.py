import os

# Smoke tests are tiny; keep them off a GPU that experiments may be using.
# The gpu golden runs main.py in a subprocess with its own environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
