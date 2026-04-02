from pathlib import Path

import flow_drive
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
class FlowDriveBehavior(Behavior):
    def __init__(
        self,
        name: str = "flow_drive",
    ):
        super().__init__(name)

        flow_drive_pkg_root = Path(flow_drive.__path__[0])

        cfg_path = str(flow_drive_pkg_root / "config" / "planner" / "flow_drive.yaml")

        with open(cfg_path, "r") as f:
            flow_drive_cfg = OmegaConf.load(f)

        # Define Flow Drive checkpoint path
        flow_drive_cfg.flow_drive.ckpt_path = str(
            flow_drive_pkg_root / "checkpoint" / "flow_drive_model.pth"
        )
        # Define model type: 0 -> without post-processing, 1 -> with moderated guidance
        flow_drive_cfg.flow_drive.post_mode = 1
        # Set unused parameters to default values
        flow_drive_cfg.flow_drive.mlflow_exp_name = "None"
        flow_drive_cfg.flow_drive.load_run_name = "None"
        flow_drive_cfg.flow_drive.load_epoch = 0
        flow_drive_cfg.flow_drive.emergency_brake_enabled = False

        self.planner = cast(AbstractPlanner, instantiate(flow_drive_cfg.flow_drive))

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
