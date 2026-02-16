from dataclasses import dataclass
from typing import final

from arbitration_graphs import Behavior
from arbitration_graphs.typing import Time
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from typing_extensions import cast, override

from mosaic.common.command import Command
from mosaic.common.environment_model import EnvironmentModel


@final
class EmergencyStopBehavior(Behavior):
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        max_long_accel: float = 2.40
        min_long_accel: float = -4.05

    def __init__(
        self,
        parameters: Parameters,
        name: str = "EmergencyStop",
    ):
        super().__init__(name)
        self.parameters: EmergencyStopBehavior.Parameters = parameters
        self.planner = PDMEmergencyBrake(
            trajectory_sampling=parameters.trajectory_sampling,
            max_long_accel=parameters.max_long_accel,
            min_long_accel=parameters.min_long_accel,
        )

    @override
    def get_command(self, time: Time, environment_model: EnvironmentModel) -> Command:
        ego_state = environment_model.ego_state
        trajectory: AbstractTrajectory = self.planner._generate_trajectory(ego_state)

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
