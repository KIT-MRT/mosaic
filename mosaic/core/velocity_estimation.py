from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)


@dataclass
class VelocityEstimationParameters:
    """
    interPlan's jaywalking pedestrians are reported as standing still although they walk,
    which hides them from constant-velocity forecasts. For pedestrians reported as standing,
    the velocity is therefore estimated from their tracked positions. Vehicles are left
    alone: their simulated velocities are exact, and logged vehicle boxes jitter enough to
    fake motion.
    """

    estimated_types: frozenset[TrackedObjectType] = frozenset(
        {TrackedObjectType.PEDESTRIAN}
    )

    max_reported_speed: float = (
        0.1  # [m/s] below this, the reported velocity is distrusted
    )
    min_estimated_speed: float = (
        0.3  # [m/s] below this, the displacement is treated as noise
    )
    estimation_window: float = 1.0  # [s] look-back over the track history
    min_estimation_window: float = 0.5  # [s] shortest usable look-back for new tracks


def estimate_missing_velocities(
    history: SimulationHistoryBuffer, parameters: VelocityEstimationParameters
) -> Observation:
    """
    :return: The current observation where agents reported as standing still but moving
        according to their track history carry the estimated velocity instead.
    """
    current_ego_state, current_observation = history.current_state
    if not isinstance(current_observation, DetectionsTracks):
        return current_observation

    current_time = current_ego_state.time_point.time_s
    track_histories = _track_histories(history, current_time, parameters)

    corrected_objects = []
    for tracked_object in current_observation.tracked_objects:
        estimated_velocity = _estimate_velocity(
            tracked_object, track_histories, parameters
        )
        if estimated_velocity is None:
            corrected_objects.append(tracked_object)
            continue
        corrected_objects.append(
            Agent(
                tracked_object_type=tracked_object.tracked_object_type,
                oriented_box=tracked_object.box,
                velocity=estimated_velocity,
                metadata=tracked_object.metadata,
                angular_velocity=tracked_object.angular_velocity,
                predictions=tracked_object.predictions,
                past_trajectory=tracked_object.past_trajectory,
            )
        )

    return DetectionsTracks(TrackedObjects(corrected_objects))


@dataclass
class _TrackHistory:
    times_s: list[float] = field(default_factory=list)
    positions_xy: list[np.ndarray] = field(default_factory=list)


def _track_histories(
    history: SimulationHistoryBuffer,
    current_time: float,
    parameters: VelocityEstimationParameters,
) -> dict[str, _TrackHistory]:
    """
    :return: Per track token, its positions within the estimation window, oldest first.
    """
    track_histories: dict[str, _TrackHistory] = {}
    for ego_state, observation in zip(history.ego_states, history.observations):
        time_before_now = current_time - ego_state.time_point.time_s
        if time_before_now > parameters.estimation_window + 1e-6:
            continue
        if not isinstance(observation, DetectionsTracks):
            continue
        for tracked_object in observation.tracked_objects:
            track_history = track_histories.setdefault(
                tracked_object.track_token, _TrackHistory()
            )
            track_history.times_s.append(ego_state.time_point.time_s)
            track_history.positions_xy.append(tracked_object.center.array)
    return track_histories


def _estimate_velocity(
    tracked_object,
    track_histories: dict[str, _TrackHistory],
    parameters: VelocityEstimationParameters,
) -> Optional[StateVector2D]:
    if not isinstance(tracked_object, Agent):
        return None
    if tracked_object.tracked_object_type not in parameters.estimated_types:
        return None
    if tracked_object.velocity.magnitude() >= parameters.max_reported_speed:
        return None
    track_history = track_histories.get(tracked_object.track_token)
    if track_history is None:
        return None
    times_s = np.array(track_history.times_s)
    if times_s[-1] - times_s[0] < parameters.min_estimation_window - 1e-6:
        return None

    # Unlike a finite difference over the window, the median is robust to one-off jumps
    # and to the box jitter of logged tracks.
    step_velocities_xy = (
        np.diff(np.array(track_history.positions_xy), axis=0)
        / (np.diff(times_s)[:, None])
    )
    velocity_xy = np.median(step_velocities_xy, axis=0)
    if np.linalg.norm(velocity_xy) < parameters.min_estimated_speed:
        return None
    return StateVector2D(*velocity_xy)
