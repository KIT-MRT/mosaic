from dataclasses import dataclass

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)


class EmergencyStopPlanner:
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        max_long_accel: float = 2.40
        min_long_accel: float = -4.05

    def __init__(self, parameters: Parameters) -> None:
        self.parameters: EmergencyStopPlanner.Parameters = parameters
        self._emergency_brake = PDMEmergencyBrake(
            trajectory_sampling=parameters.trajectory_sampling,
            max_long_accel=parameters.max_long_accel,
            min_long_accel=parameters.min_long_accel,
        )

    def plan_trajectory(self, ego_state: EgoState) -> InterpolatedTrajectory:
        return self._emergency_brake._generate_trajectory(ego_state)
