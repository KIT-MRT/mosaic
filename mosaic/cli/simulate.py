from typing import Union

import click


@click.command()
@click.option(
    "--challenge",
    "-c",
    default="closed_loop_reactive_agents",
    type=click.Choice(
        [
            "closed_loop_reactive_agents",
            "closed_loop_nonreactive_agents",
            "open_loop_boxes",
        ]
    ),
    help="Simulation challenge type.",
)
@click.option(
    "--scenario-filter",
    default="val14_split",
    help="Scenario filter preset.",
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
    scenario_filter: str,
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

    import hydra
    from hydra import compose, initialize_config_module

    SIM_CONFIG_PATH = "nuplan.planning.script.config.simulation"
    SIM_CONFIG_NAME = "default_simulation"

    overrides = [
        f"experiment_name={experiment_name}",
        f"+simulation={challenge}",
        f"scenario_filter={scenario_filter}",
        "scenario_builder=nuplan",
        "enable_simulation_progress_bar=true",
        "worker=ray_distributed",
        f"worker.threads_per_node={threads}",
        "distributed_mode=SINGLE_NODE",
        f"number_of_gpus_allocated_per_simulation={gpus_per_sim}",
        "hydra.searchpath=[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]",
    ]

    if limit_scenarios is not None:
        overrides.append(f"scenario_filter.limit_total_scenarios={limit_scenarios}")

    overrides.extend(override)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_module(config_module=SIM_CONFIG_PATH):
        cfg = compose(config_name=SIM_CONFIG_NAME, overrides=overrides)

    from nuplan.planning.script.run_simulation import run_simulation

    from mosaic.ego_agent import EgoAgent

    planner = EgoAgent()
    run_simulation(cfg, planners=planner)

    click.echo(f"Simulation results are saved in: {cfg.output_dir}")
