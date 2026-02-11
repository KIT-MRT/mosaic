from dataclasses import dataclass
from typing import cast

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimeDuration,
)
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class EmergencyStopPlanner:
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        target_acceleration: float  # Should be negative (deceleration)

    def __init__(self, parameters: Parameters) -> None:
        self.parameters: EmergencyStopPlanner.Parameters = parameters

    def plan_trajectory(self, ego_state: EgoState) -> InterpolatedTrajectory:
        t0 = ego_state.time_point
        x0 = ego_state.center.x
        v0 = ego_state.dynamic_car_state.center_velocity_2d.x

        target_deceleration = abs(self.parameters.target_acceleration)

        step_time = TimeDuration.from_s(self.parameters.trajectory_sampling.step_time)
        num_poses = cast(int, self.parameters.trajectory_sampling.num_poses)

        stopping_duration = TimeDuration.from_s(
            v0 / target_deceleration if v0 > 0 else 0.0
        )

        trajectory_states: list[InterpolatableState] = []

        current_time_point = t0
        for _ in range(num_poses + 1):
            dt = TimeDuration.from_s((current_time_point - t0).time_s)

            if dt < stopping_duration:
                # Still decelerating
                v_t = max(0.0, v0 - target_deceleration * dt.time_s)
                x_t = x0 + v0 * dt.time_s - 0.5 * target_deceleration * dt.time_s**2
                a_t = -target_deceleration
            else:
                # Vehicle stopped, hold position and zero velocity/acceleration
                v_t = 0.0
                x_t = (
                    x0
                    + v0 * stopping_duration.time_s
                    - 0.5 * target_deceleration * stopping_duration.time_s**2
                )
                a_t = 0.0

            pose = relative_to_absolute_poses(
                ego_state.center, [StateSE2(x_t - x0, 0.0, 0.0)]
            )[0]

            ego_state_ = EgoState.build_from_center(
                center=pose,
                center_velocity_2d=StateVector2D(v_t, 0.0),
                center_acceleration_2d=StateVector2D(a_t, 0.0),
                tire_steering_angle=0.0,
                time_point=current_time_point,
                vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
            )
            trajectory_states.append(ego_state_)

            current_time_point += step_time

        return InterpolatedTrajectory(trajectory_states)
