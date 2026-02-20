## Mosaic

### Setup and Usage

We use [`uv`](https://docs.astral.sh/uv/) to manage the Python environment.
With the cloned repository, run `uv sync` to install the dependencies.
Then activate the environment with `source .venv/bin/activate` (or whichever script is appropriate for your shell).

You now have access to the `mosaic` command, which is the main entry point for running Mosaic.

Run `moasic simulate` to run the simulation or `mosaic results` to launch nuboard.
See the respective `--help` for more details on the available options.

Notes:
- Until Flowdrive has been open-sourced, it must be cloned next to this repository for uv to find it.
- Until Flowdrive has been open-sourced, `uv run mosaic simulate` will not work due to Ray environment issues.

### Development

To run the tests, use `uv run pytest`.
To run the linter, use `uv run ruff check .`.
To run the formatter, use `uv run ruff format --check .` to check for formatting issues or `uv run ruff format .` to automatically fix them.
