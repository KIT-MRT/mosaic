from pathlib import Path

import click


def _find_latest_nuboard_file():
    """Find the most recent .nuboard file under the default nuplan output dir."""
    output_root = Path.home() / "nuplan" / "exp"
    if not output_root.exists():
        raise click.ClickException(f"No output directory found at {output_root}. Run a simulation first or provide --path.")

    nuboard_files = sorted(output_root.rglob("*.nuboard"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not nuboard_files:
        raise click.ClickException(f"No .nuboard files found under {output_root}. Run a simulation first or provide --path.")

    return nuboard_files[0]


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Path to simulation output dir or .nuboard file (auto-detects latest if omitted).",
)
@click.option(
    "--scenario-builder",
    default="nuplan_mini",
    help="Scenario builder for data visualization.",
)
def results(path, scenario_builder):
    """Launch nuBoard to visualize simulation results."""
    from mosaic.cli._env import setup_uv_env, setup_matplotlib

    setup_uv_env()
    setup_matplotlib()

    if path is None:
        nuboard_file = _find_latest_nuboard_file()
        click.echo(f"Using latest .nuboard file: {nuboard_file}")
    else:
        p = Path(path)
        if p.is_dir():
            nuboard_files = list(p.glob("*.nuboard"))
            if not nuboard_files:
                raise click.ClickException(f"No .nuboard files found in {p}")
            nuboard_file = nuboard_files[0]
        else:
            nuboard_file = p

    import hydra
    from hydra import compose, initialize_config_module

    NUBOARD_CONFIG_PATH = "nuplan.planning.script.config.nuboard"
    NUBOARD_CONFIG_NAME = "default_nuboard"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_module(config_module=NUBOARD_CONFIG_PATH):
        cfg = compose(
            config_name=NUBOARD_CONFIG_NAME,
            overrides=[
                f"scenario_builder={scenario_builder}",
                f"simulation_path=[{nuboard_file}]",
            ],
        )

    from nuplan.planning.script.run_nuboard import initialize_nuboard

    nuboard = initialize_nuboard(cfg)
    nuboard.run()
