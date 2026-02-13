import os
import sys


def setup_uv_env():
    """Set UV_PROJECT_ENVIRONMENT so Ray subprocesses find the venv."""
    if "UV_PROJECT_ENVIRONMENT" not in os.environ:
        os.environ["UV_PROJECT_ENVIRONMENT"] = sys.prefix


def setup_matplotlib():
    """Force non-interactive matplotlib backend."""
    import matplotlib

    matplotlib.use("Agg")
