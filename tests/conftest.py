import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

# Smoke tests are tiny; keep them off a GPU that experiments may be using.
# The --gpu golden runs main.py in a subprocess with its own environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

GOLDEN_DIR = Path(__file__).parent / "golden"


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Rewrite tests/golden/*.json from this run instead of comparing.",
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
    """Compare ``{seed: {tag: [[step, value], ...]}}`` with tests/golden/<name>.json."""

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
        assert actual.keys() == expected.keys()
        for seed in expected:
            assert actual[seed].keys() == expected[seed].keys(), seed
            for tag, points in expected[seed].items():
                got = np.asarray(actual[seed][tag])
                want = np.asarray(points)
                assert got.shape == want.shape, f"{seed}/{tag}"
                np.testing.assert_array_equal(got[:, 0], want[:, 0], f"{seed}/{tag}")
                if exact:
                    np.testing.assert_array_equal(
                        got[:, 1], want[:, 1], f"{seed}/{tag}"
                    )
                else:
                    np.testing.assert_allclose(
                        got[:, 1],
                        want[:, 1],
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg=f"{seed}/{tag}",
                    )

    return check
