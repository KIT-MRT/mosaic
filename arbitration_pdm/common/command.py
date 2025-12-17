from typing import cast, Optional, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


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

    def ego_states(self, sampling: Optional[TrajectorySampling] = None) -> list[EgoState]:
        """
        Return ego states sampled according to `sampling`.
        If `sampling` is None, return the trajectory's native sampled states.
        """
        if sampling is None:
            return cast(list[EgoState], self.trajectory.get_sampled_trajectory())

        # build time points anchored at trajectory start_time
        start_time = self.trajectory.start_time
        step_time_s = sampling.step_time
        num = sampling.num_poses + 1  # include initial state
        time_points = [TimePoint(start_time.time_us + int(round(k * step_time_s * 1e6))) for k in range(num)]

        return cast(list[EgoState], self.trajectory.get_state_at_times(time_points))
