import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    TimeDuration,
    TimePoint,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)


def trajectory_to_state_array(
    trajectory: AbstractTrajectory,
    trajectory_sampling: TrajectorySampling,
    eps_us: int = 1,
) -> npt.NDArray[np.float64]:
    """
    Resample a trajectory to desired sampling and convert to state array.
    trajectory: The trajectory to be converted.
    trajectory_sampling: The desired sampling parameters.
    eps_us: Tolerance in microseconds to prevent floating point issues when matching end times.
    """
    step_time: TimeDuration = TimeDuration.from_s(trajectory_sampling.step_time)
    num_poses = trajectory_sampling.num_poses
    assert num_poses is not None

    time_points: list[TimePoint] = [
        trajectory.start_time + step_time * i for i in range(num_poses + 1)
    ]
    last = time_points[-1]
    end = trajectory.end_time
    if last.time_us > end.time_us and last.time_us - end.time_us <= eps_us:
        time_points[-1] = end

    ego_states: list[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(ego_states)
