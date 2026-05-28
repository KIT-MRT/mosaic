from dataclasses import dataclass
from typing import Optional, final

import numpy as np
from arbitration_graphs import Behavior
from arbitration_graphs.typing import Time
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from typing_extensions import cast, override

import mosaic.utils.trajectory as trajectory_utils
from mosaic.core.command import Command, StampedCommand
from mosaic.core.environment_model import EnvironmentModel


@final
class EmergencyStopBehavior(Behavior):
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        max_long_accel: float = 2.40
        min_long_accel: float = -8.0

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
        self.last_command: Optional[StampedCommand] = None
        self._simulator: Optional[PDMSimulator] = None

    def initialize(self, environment_model: EnvironmentModel) -> None:
        self._simulator = PDMSimulator(environment_model.parameters.proposal_sampling)

    @override
    def get_command(self, time: Time, environment_model: EnvironmentModel) -> Command:
        assert self._simulator is not None
        ego_state = environment_model.ego_state
        trajectory: AbstractTrajectory = self.planner._generate_trajectory(ego_state)

        states = trajectory_utils.trajectory_to_state_array(
            trajectory, environment_model.parameters.proposal_sampling
        )
        simulated = self._simulator.simulate_proposals(
            np.expand_dims(states, axis=0), environment_model.ego_state
        )
        command = Command(
            name=self.name(), trajectory=trajectory, simulated_states=simulated[0]
        )
        self.last_command = StampedCommand(stamp=time, command=command)
        return command

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
