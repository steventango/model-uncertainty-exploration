import os


def pytest_configure(config):
    # JAX picks its platform when first imported, so it is chosen from the
    # environment, not in code. The default run (-m 'not gpu') needs CPU.
    if "not gpu" in (config.option.markexpr or ""):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
