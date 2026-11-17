import os

import pytest


def pytest_configure(config: pytest.Config):
    if "not gpu" in (config.option.markexpr or ""):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
