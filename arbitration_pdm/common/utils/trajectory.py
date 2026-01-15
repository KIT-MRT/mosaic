from typing import cast

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    TimeDuration,
    TimePoint,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)


def trajectory_to_state_array(
    trajectory: InterpolatedTrajectory, trajectory_sampling: TrajectorySampling
) -> npt.NDArray[np.float64]:
    """Resample an InterpolatedTrajectory to desired sampling and convert to state array."""
    step_time: TimeDuration = TimeDuration.from_s(trajectory_sampling.step_time)
    num_poses = trajectory_sampling.num_poses
    assert num_poses is not None

    time_points: list[TimePoint] = [
        trajectory.start_time + step_time * i for i in range(num_poses + 1)
    ]
    ego_states = cast(list[EgoState], trajectory.get_state_at_times(time_points))

    return ego_states_to_state_array(ego_states)
