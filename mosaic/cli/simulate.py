from typing import Union

import click
from omegaconf import DictConfig

from mosaic.ablation import Ablation


def _configure_hydra(overrides: list[str]) -> DictConfig:
    import hydra
    from hydra import compose, initialize_config_module

    SIM_CONFIG_PATH = "nuplan.planning.script.config.simulation"
    SIM_CONFIG_NAME = "default_simulation"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_module(config_module=SIM_CONFIG_PATH):
        cfg = compose(config_name=SIM_CONFIG_NAME, overrides=overrides)

    return cfg


INTERPLAN_CHALLENGE = "interplan"

NUPLAN_CHALLENGES = [
    "closed_loop_reactive_agents",
    "closed_loop_nonreactive_agents",
]

ALL_CHALLENGES = NUPLAN_CHALLENGES + [INTERPLAN_CHALLENGE]


@click.command()
@click.option(
    "--challenge",
    "-c",
    default="closed_loop_reactive_agents",
    type=click.Choice(ALL_CHALLENGES),
    help="Simulation challenge type.",
)
@click.option(
    "--scenario-filter",
    default=None,
    help="Scenario filter preset (default: val14_split, or interplan10 for interplan challenge).",
)
@click.option(
    "--ablation",
    type=click.Choice([a.value for a in Ablation], case_sensitive=False),
    default=Ablation.NONE.value,
    help="Ablation mode.",
)
@click.option(
    "--limit-scenarios",
    "-n",
    type=int,
    default=None,
    help="Limit total scenarios (for quick testing).",
)
@click.option(
    "--experiment-name",
    default="mosaic",
    help="Experiment name for output directory.",
)
@click.option(
    "--threads",
    type=int,
    default=160,
    help="Worker threads per node.",
)
@click.option(
    "--gpus-per-sim",
    type=float,
    default=0.05,
    help="GPUs allocated per simulation.",
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    help="Arbitrary Hydra overrides (repeatable, e.g. -o worker.threads_per_node=80).",
)
def simulate(
    challenge: str,
    scenario_filter: Union[str, None],
    ablation: str,
    limit_scenarios: Union[int, None],
    experiment_name: str,
    threads: int,
    gpus_per_sim: float,
    override: tuple[str, ...],
) -> None:
    """Run nuplan simulation."""
    from mosaic.cli._env import setup_matplotlib, setup_uv_env

    setup_uv_env()
    setup_matplotlib()

    is_interplan = challenge == INTERPLAN_CHALLENGE

    if scenario_filter is None:
        scenario_filter = "interplan10" if is_interplan else "val14_split"

    simulation = challenge
    searchpath = "pkg://mosaic.config,"

    if is_interplan:
        simulation = "default_interplan_benchmark"
        searchpath += (
            "pkg://interplan.planning.script.config.common,"
            "pkg://interplan.planning.script.config.simulation,"
            "pkg://interplan.planning.script.experiments,"
        )

    searchpath += (
        "pkg://flow_drive.config,"
        "pkg://tuplan_garage.planning.script.config.common,"
        "pkg://tuplan_garage.planning.script.config.simulation,"
        "pkg://nuplan.planning.script.config.common,"
        "pkg://nuplan.planning.script.experiments"
    )

    overrides = [
        f"experiment_name={experiment_name}",
        f"+simulation={simulation}",
        f"scenario_filter={scenario_filter}",
        "enable_simulation_progress_bar=true",
        "worker=ray_distributed",
        f"worker.threads_per_node={threads}",
        "distributed_mode=SINGLE_NODE",
        f"number_of_gpus_allocated_per_simulation={gpus_per_sim}",
        "+callback.mosaic_logging_callback._target_=mosaic.cli.callback.MosaicLoggingCallback",
        "+callback.mosaic_logging_callback.output_directory=${output_dir}",
        f"hydra.searchpath=[{searchpath}]",
    ]

    is_test_split = scenario_filter.startswith("test14")

    if is_interplan:
        overrides.append(
            "scenario_builder.data_root=${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test"
        )
    else:
        overrides.append("scenario_builder=nuplan")
        if is_test_split:
            overrides.append(
                "scenario_builder.data_root=${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test"
            )

    if limit_scenarios is not None:
        overrides.append(f"scenario_filter.limit_total_scenarios={limit_scenarios}")

    overrides.extend(override)
    cfg = _configure_hydra(overrides)

    if is_interplan:
        from interplan.planning.utils.modifications_preprocessing import (
            preprocess_scenario_filter,
        )

        preprocess_scenario_filter(cfg)
        from interplan.planning.script.run_simulation import run_simulation
    else:
        from nuplan.planning.script.run_simulation import run_simulation

    from mosaic.core.mosaic_planner import Mosaic

    parameters = Mosaic.Parameters(
        ablation=Ablation(ablation),
    )

    run_simulation(cfg, planners=Mosaic(parameters))

    click.echo(f"Simulation results are saved in: {cfg.output_dir}")
