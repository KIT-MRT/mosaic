## Mosaic

### Setup and Usage

We use [`uv`](https://docs.astral.sh/uv/) to manage the Python environment.
With the cloned repository, run `uv sync` to install the dependencies.
Then activate the environment with `source .venv/bin/activate` (or whichever script is appropriate for your shell).

Now you can run the main script with `python main.py`.

Notes:
- Until Flowdrive has been open-sourced, it must be cloned next to this repository for uv to find it.
- `uv run` will not work due to issues with the ray parallelization.
