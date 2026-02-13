import os
from pathlib import Path
from typing import Union

import click
from omegaconf import DictConfig


def _find_latest_nuboard_file() -> Path:
    if "NUPLAN_EXP_ROOT" in os.environ:
        output_root = Path(os.environ["NUPLAN_EXP_ROOT"])
    else:
        output_root = Path.home() / "nuplan" / "exp"
    if not output_root.exists():
        raise click.ClickException(
            f"No output directory found at {output_root}. Run a simulation first or provide --path."
        )

    nuboard_files = sorted(
        output_root.rglob("*.nuboard"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not nuboard_files:
        raise click.ClickException(
            f"No .nuboard files found under {output_root}. Run a simulation first or provide --path."
        )

    return nuboard_files[0]


def _find_nuboard_file(path: Path) -> Path:
    if path.is_dir():
        nuboard_files = list(path.glob("*.nuboard"))
        if not nuboard_files:
            raise click.ClickException(f"No .nuboard files found in {path}")
        if len(nuboard_files) > 1:
            click.echo(
                f"Multiple .nuboard files found in {path}, using the most recent one: {nuboard_files[0]}"
            )
        return nuboard_files[0]
    else:
        if path.suffix != ".nuboard":
            raise click.ClickException(f"Provided file {path} is not a .nuboard file.")
        return path


def _configure_hydra(overrides: list[str]) -> DictConfig:
    import hydra
    from hydra import compose, initialize_config_module

    NUBOARD_CONFIG_PATH = "nuplan.planning.script.config.nuboard"
    NUBOARD_CONFIG_NAME = "default_nuboard"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_module(config_module=NUBOARD_CONFIG_PATH):
        cfg = compose(config_name=NUBOARD_CONFIG_NAME, overrides=overrides)

    return cfg


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Path to simulation output dir or .nuboard file (auto-detects latest if omitted).",
)
@click.option(
    "--port",
    type=click.INT,
    default=5006,
    help="Port number for NuBoard server (default: 5006).",
)
def results(path: Union[str, None], port: int) -> None:
    """Launch nuBoard to visualize simulation results."""
    from mosaic.cli._env import setup_matplotlib, setup_uv_env

    setup_uv_env()
    setup_matplotlib()

    if path is None:
        nuboard_file = _find_latest_nuboard_file()
        click.echo(f"Using latest .nuboard file: {nuboard_file}")
    else:
        nuboard_file = _find_nuboard_file(Path(path))
        click.echo(f"Using .nuboard file: {nuboard_file}")

    overrides = [
        "scenario_builder=nuplan",
        f"port_number={port}",
        f"simulation_path=[{nuboard_file}]",
    ]
    cfg = _configure_hydra(overrides)

    from nuplan.planning.script.run_nuboard import initialize_nuboard

    nuboard = initialize_nuboard(cfg)
    nuboard.run()
