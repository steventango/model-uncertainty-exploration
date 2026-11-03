import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

# Smoke tests are tiny; keep them off a GPU that experiments may be using.
# The --gpu golden runs main.py in a subprocess with its own environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# JAX reads JAX_PLATFORMS when imported, so chex (which imports jax) comes after.
import chex

GOLDEN_DIR = Path("/tmp/model-uncertainty-exploration-golden")


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Rewrite the golden JSON files from this run instead of comparing.",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        help="Run the full-scale GPU golden tests (bit-exact, recorded on one GPU).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="needs --gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def golden(request):
    """Compare ``{seed: {tag: [[step, value], ...]}}`` with GOLDEN_DIR/<name>.json."""

    def check(name, actual, *, exact=False):
        path = GOLDEN_DIR / f"{name}.json"
        if request.config.getoption("--update-golden"):
            GOLDEN_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
            return
        if not path.exists():
            warnings.warn(f"no golden for {name}; record one with --update-golden")
            return
        expected = json.loads(path.read_text())
        actual, expected = (
            {
                seed: {tag: np.asarray(points) for tag, points in tags.items()}
                for seed, tags in d.items()
            }
            for d in (actual, expected)
        )
        if exact:
            chex.assert_trees_all_equal(actual, expected)
        else:
            chex.assert_trees_all_close(actual, expected, rtol=1e-5, atol=1e-6)

    return check
