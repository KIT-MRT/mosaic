from typing import cast

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)


class Command:
    def __init__(self, name: str, trajectory: InterpolatedTrajectory) -> None:
        self.name: str = name

        assert isinstance(trajectory, InterpolatedTrajectory), (
            "Command trajectory must be of type InterpolatedTrajectory"
        )
        sampled = trajectory.get_sampled_trajectory()
        assert isinstance(sampled, list), "Sampled trajectory must be a list"
        assert all(isinstance(s, EgoState) for s in sampled), (
            "Sampled trajectory must contain only EgoState instances"
        )
        self.trajectory: InterpolatedTrajectory = trajectory

    def ego_states(self):
        return cast(list[EgoState], self.trajectory.get_sampled_trajectory())
