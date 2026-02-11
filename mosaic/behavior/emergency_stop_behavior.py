from dataclasses import dataclass

from arbitration_graphs import Behavior
from arbitration_graphs.typing import Time
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from typing_extensions import cast, override

from mosaic.common.command import Command
from mosaic.common.environment_model import EnvironmentModel
from mosaic.planner.emergency_stop_planner import (
    EmergencyStopPlanner,
)


class EmergencyStopBehavior(Behavior):
    @dataclass
    class Parameters:
        emergency_brake_planner: EmergencyStopPlanner.Parameters

    def __init__(
        self,
        parameters: Parameters,
        name: str = "EmergencyStop",
    ):
        super().__init__(name)
        self.planner: EmergencyStopPlanner = EmergencyStopPlanner(
            parameters.emergency_brake_planner
        )

    @override
    def get_command(self, time: Time, environment_model: EnvironmentModel) -> Command:
        ego_state = environment_model.ego_state
        trajectory: AbstractTrajectory = self.planner.plan_trajectory(ego_state)

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
        return False

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
