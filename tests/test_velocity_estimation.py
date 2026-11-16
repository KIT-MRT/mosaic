from collections import deque
from unittest.mock import MagicMock

import pytest
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from mosaic.core.velocity_estimation import (
    VelocityEstimationParameters,
    estimate_missing_velocities,
)

SAMPLE_INTERVAL = 0.1
NUM_SAMPLES = 21


def _agent(
    x: float,
    y: float,
    reported_velocity: StateVector2D,
    tracked_object_type: TrackedObjectType = TrackedObjectType.PEDESTRIAN,
) -> Agent:
    return Agent(
        tracked_object_type=tracked_object_type,
        oriented_box=OrientedBox(StateSE2(x, y, 0.0), 0.9, 0.8, 1.8),
        velocity=reported_velocity,
        metadata=SceneObjectMetadata(
            timestamp_us=0, token="ped", track_id=1, track_token="pedestrian_track"
        ),
    )


def _history(
    agent_velocity_xy: tuple[float, float],
    reported_velocity: StateVector2D,
    num_samples: int = NUM_SAMPLES,
    tracked_object_type: TrackedObjectType = TrackedObjectType.PEDESTRIAN,
) -> SimulationHistoryBuffer:
    ego_states = []
    observations = []
    for sample_idx in range(num_samples):
        time_s = sample_idx * SAMPLE_INTERVAL
        ego_state = MagicMock()
        ego_state.time_point = TimePoint(int(time_s * 1e6))
        ego_states.append(ego_state)
        agent = _agent(
            agent_velocity_xy[0] * time_s,
            agent_velocity_xy[1] * time_s,
            reported_velocity,
            tracked_object_type,
        )
        observations.append(DetectionsTracks(TrackedObjects([agent])))
    return SimulationHistoryBuffer(
        deque(ego_states, maxlen=num_samples),
        deque(observations, maxlen=num_samples),
        SAMPLE_INTERVAL,
    )


def _current_velocity(history: SimulationHistoryBuffer) -> StateVector2D:
    observation = estimate_missing_velocities(history, VelocityEstimationParameters())
    return observation.tracked_objects.tracked_objects[0].velocity


def test_moving_agent_reported_as_standing_gets_estimated_velocity():
    velocity = _current_velocity(_history((0.0, 0.55), StateVector2D(0.0, 0.0)))
    assert velocity.x == pytest.approx(0.0, abs=1e-6)
    assert velocity.y == pytest.approx(0.55)


def test_reported_velocity_is_kept_when_present():
    reported = StateVector2D(1.0, 0.0)
    velocity = _current_velocity(_history((0.0, 0.55), reported))
    assert velocity.x == pytest.approx(1.0)
    assert velocity.y == pytest.approx(0.0)


def test_small_displacement_is_treated_as_noise():
    velocity = _current_velocity(_history((0.1, 0.0), StateVector2D(0.0, 0.0)))
    assert velocity.magnitude() == pytest.approx(0.0)


def test_track_shorter_than_minimum_window_is_not_estimated():
    velocity = _current_velocity(
        _history((0.0, 0.55), StateVector2D(0.0, 0.0), num_samples=3)
    )
    assert velocity.magnitude() == pytest.approx(0.0)


def test_single_jump_in_track_is_ignored():
    history = _history((0.0, 0.0), StateVector2D(0.0, 0.0))
    snapped_observation = DetectionsTracks(
        TrackedObjects([_agent(0.0, 1.5, StateVector2D(0.0, 0.0))])
    )
    jump_idx = NUM_SAMPLES - 8
    observations = list(history.observations)
    for sample_idx in range(jump_idx, NUM_SAMPLES):
        observations[sample_idx] = snapped_observation
    history = SimulationHistoryBuffer(
        deque(history.ego_states, maxlen=NUM_SAMPLES),
        deque(observations, maxlen=NUM_SAMPLES),
        SAMPLE_INTERVAL,
    )
    velocity = _current_velocity(history)
    assert velocity.magnitude() == pytest.approx(0.0)


def test_vehicles_are_not_estimated():
    velocity = _current_velocity(
        _history(
            (0.0, 0.55),
            StateVector2D(0.0, 0.0),
            tracked_object_type=TrackedObjectType.VEHICLE,
        )
    )
    assert velocity.magnitude() == pytest.approx(0.0)
