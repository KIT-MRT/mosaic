from importlib import resources

from arbitration_graphs import Behavior
from arbitration_graphs.typing import Time
from hydra.utils import instantiate
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from omegaconf import OmegaConf
from typing_extensions import cast, final, override

from mosaic.core.command import Command
from mosaic.core.environment_model import EnvironmentModel


@final
class PDMClosedBehavior(Behavior):
    def __init__(
        self,
        name: str = "pdm_closed",
    ):
        super().__init__(name)

        cfg_path = (
            resources.files("tuplan_garage")
            / "planning"
            / "script"
            / "config"
            / "simulation"
            / "planner"
            / "pdm_closed_planner.yaml"
        )

        with cfg_path.open("r") as f:
            pdm_closed_cfg = OmegaConf.load(f)

        pdm_closed_cfg.pdm_closed_planner.enable_emergency_brake_fallback = False

        self.planner = cast(
            AbstractPlanner, instantiate(pdm_closed_cfg.pdm_closed_planner)
        )

    def initialize(self, environment_model: EnvironmentModel) -> None:
        self.planner.initialize(environment_model.planner_initialization)

    @override
    def get_command(self, time: Time, environment_model: EnvironmentModel) -> Command:
        trajectory: AbstractTrajectory = self.planner.compute_planner_trajectory(
            environment_model.planner_input
        )

        return Command(
            name=self.name(),
            trajectory=trajectory,
        )

    @override
    def check_invocation_condition(
        self, time: Time, environment_model: EnvironmentModel
    ) -> bool:
        return True

    @override
    def check_commitment_condition(
        self, time: Time, environment_model: EnvironmentModel
    ) -> bool:
        return True

    def __getstate__(self) -> dict[str, object]:
        """
        Custom getstate to fix pickling since this is a class that inherits from a C++ object
        """
        state = self.__dict__.copy()
        state["name"] = self.name()
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Custom setstate to fix pickling since this is a class that inherits from a C++ object
        """
        name = cast(str, state["name"])
        super().__init__(name)
        _ = state.pop("name", None)
        self.__dict__.update(state)
