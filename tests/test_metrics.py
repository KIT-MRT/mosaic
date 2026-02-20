from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from mosaic.scorer import (
    ComfortMetric,
    DrivableAreaComplianceMetric,
    DrivingDirectionComplianceMetric,
    NoAtFaultCollisionMetric,
    ProgressMetric,
    TTCMetric,
)
from mosaic.scorer.scoring_input import ScoringInput


N_PROPOSALS = 2
N_POSES = 10


@pytest.fixture
def proposal_sampling():
    return TrajectorySampling(time_horizon=1.0, interval_length=0.1)


@pytest.fixture
def env_model():
    return MagicMock()


def _make_scoring_input(
    proposal_sampling,
    ego_areas=None,
    ego_coords=None,
    states=None,
    ego_polygons=None,
):
    """Helper to build a ScoringInput with defaults for unneeded fields."""
    if states is None:
        states = np.zeros((N_PROPOSALS, N_POSES + 1, 11), dtype=np.float64)
    if ego_coords is None:
        ego_coords = np.zeros(
            (N_PROPOSALS, N_POSES + 1, 5, 2), dtype=np.float64
        )
    if ego_polygons is None:
        ego_polygons = np.empty((N_PROPOSALS, N_POSES + 1), dtype=np.object_)
    if ego_areas is None:
        ego_areas = np.zeros((N_PROPOSALS, N_POSES + 1, 3), dtype=np.bool_)

    return ScoringInput(
        states=states,
        ego_coords=ego_coords,
        ego_polygons=ego_polygons,
        ego_areas=ego_areas,
        proposal_sampling=proposal_sampling,
    )


# === NoAtFaultCollisionMetric ===


class TestNoAtFaultCollisionMetric:
    def test_no_collision(self, proposal_sampling, env_model):
        """No intersecting obstacles -> scores all 1.0, collision_time_idcs all inf."""
        si = _make_scoring_input(proposal_sampling)
        obs = MagicMock()
        obs.collided_track_ids = []
        # Make query return empty for every timestep
        time_step_mock = MagicMock()
        time_step_mock.query.return_value = np.array([[], []])
        obs.__getitem__ = MagicMock(return_value=time_step_mock)
        env_model.observation = obs

        metric = NoAtFaultCollisionMetric()
        result = metric.compute(si, env_model)

        np.testing.assert_array_equal(result.scores, [1.0, 1.0])
        np.testing.assert_array_equal(
            result.metadata["collision_time_idcs"], [np.inf, np.inf]
        )

    def test_with_collision(self, proposal_sampling, env_model):
        """Mock observation with intersecting geometry -> score < 1.0."""
        si = _make_scoring_input(proposal_sampling)

        # Build a mock observation that reports intersection for proposal 0 at time 0
        obs = MagicMock()
        obs.collided_track_ids = []
        obs.red_light_token = "red_light"

        tracked_obj = MagicMock()
        tracked_obj.tracked_object_type = "VEHICLE"
        tracked_obj.box.geometry.area = 10.0

        obs.unique_objects = {"obj_1": tracked_obj}

        def make_time_step(time_idx):
            ts = MagicMock()
            if time_idx == 0:
                # Return intersection: proposal 0 intersects geometry 0
                ts.query.return_value = np.array([[0], [0]])
                ts.tokens = ["obj_1"]
                ts.__getitem__ = MagicMock(return_value=MagicMock())
            else:
                ts.query.return_value = np.array([[], []])
            return ts

        obs.__getitem__ = MagicMock(side_effect=make_time_step)

        env_model.observation = obs

        # Mock get_collision_type to return ACTIVE_FRONT_COLLISION
        with patch(
            "mosaic.scorer.no_at_fault_collision_metric.get_collision_type"
        ) as mock_ct:
            from nuplan.planning.metrics.utils.collision_utils import CollisionType

            mock_ct.return_value = CollisionType.ACTIVE_FRONT_COLLISION

            # Mock the ego polygon intersection
            ego_poly = MagicMock()
            ego_poly.area = 10.0
            intersection_mock = MagicMock()
            intersection_mock.area = 5.0  # 50% overlap
            ego_poly.intersection.return_value = intersection_mock

            ego_polygons = np.empty(
                (N_PROPOSALS, N_POSES + 1), dtype=np.object_
            )
            ego_polygons.fill(MagicMock())
            ego_polygons[0, 0] = ego_poly

            si = _make_scoring_input(proposal_sampling, ego_polygons=ego_polygons)

            metric = NoAtFaultCollisionMetric()
            result = metric.compute(si, env_model)

            assert result.scores[0] < 1.0
            assert result.metadata["collision_time_idcs"][0] < np.inf


# === DrivableAreaComplianceMetric ===


class TestDrivableAreaComplianceMetric:
    def test_all_in_drivable(self, proposal_sampling, env_model):
        """No NON_DRIVABLE_AREA -> score 1.0."""
        ego_areas = np.zeros((N_PROPOSALS, N_POSES + 1, 3), dtype=np.bool_)
        si = _make_scoring_input(proposal_sampling, ego_areas=ego_areas)

        metric = DrivableAreaComplianceMetric()
        result = metric.compute(si, env_model)

        np.testing.assert_array_almost_equal(result.scores, [1.0, 1.0])

    def test_partial_drivable(self, proposal_sampling, env_model):
        """Some NON_DRIVABLE_AREA -> score between 0 and 1."""
        ego_areas = np.zeros((N_PROPOSALS, N_POSES + 1, 3), dtype=np.bool_)
        # Mark half the timesteps as non-drivable for proposal 0
        ego_areas[0, : (N_POSES + 1) // 2, 1] = True  # EgoAreaIndex.NON_DRIVABLE_AREA = 1

        si = _make_scoring_input(proposal_sampling, ego_areas=ego_areas)

        metric = DrivableAreaComplianceMetric()
        result = metric.compute(si, env_model)

        assert 0.0 < result.scores[0] < 1.0
        assert result.scores[1] == 1.0


# === DrivingDirectionComplianceMetric ===


class TestDrivingDirectionComplianceMetric:
    def test_compliant(self, proposal_sampling, env_model):
        """No ONCOMING_TRAFFIC -> score 1.0."""
        ego_areas = np.zeros((N_PROPOSALS, N_POSES + 1, 3), dtype=np.bool_)
        si = _make_scoring_input(proposal_sampling, ego_areas=ego_areas)

        metric = DrivingDirectionComplianceMetric()
        result = metric.compute(si, env_model)

        np.testing.assert_array_equal(result.scores, [1.0, 1.0])

    def test_violation(self, proposal_sampling, env_model):
        """Sustained oncoming traffic > violation_threshold -> score 0.0."""
        ego_areas = np.zeros((N_PROPOSALS, N_POSES + 1, 3), dtype=np.bool_)
        # Mark all timesteps as oncoming traffic for proposal 0
        ego_areas[0, :, 2] = True  # EgoAreaIndex.ONCOMING_TRAFFIC = 2

        # Create ego_coords where proposal 0 moves 1.0m per timestep
        ego_coords = np.zeros(
            (N_PROPOSALS, N_POSES + 1, 5, 2), dtype=np.float64
        )
        for t in range(N_POSES + 1):
            ego_coords[0, t, :, 0] = float(t)  # x increases by 1 each step

        si = _make_scoring_input(
            proposal_sampling, ego_areas=ego_areas, ego_coords=ego_coords
        )

        metric = DrivingDirectionComplianceMetric(
            compliance_threshold=2.0, violation_threshold=6.0
        )
        result = metric.compute(si, env_model)

        assert result.scores[0] == 0.0  # large oncoming progress
        assert result.scores[1] == 1.0  # no oncoming traffic


# === ProgressMetric ===


class TestProgressMetric:
    def test_stationary(self, proposal_sampling, env_model):
        """No movement along centerline -> score 0.0."""
        si = _make_scoring_input(proposal_sampling)
        centerline = MagicMock()
        centerline.project.return_value = [0.0, 0.0]  # no progress
        centerline.length = 100.0
        env_model.route_center_line = centerline
        ego_state = MagicMock()
        ego_state.dynamic_car_state.speed = 10.0
        env_model.ego_state = ego_state

        metric = ProgressMetric()
        result = metric.compute(si, env_model)

        np.testing.assert_array_almost_equal(result.scores, [0.0, 0.0])

    def test_full_progress(self, proposal_sampling, env_model):
        """Movement matching expected -> score 1.0."""
        si = _make_scoring_input(proposal_sampling)
        # ego_speed=10, horizon_time=1.0s -> expected=10m
        # but fallback_max_meters=10 and min_expected=0.1
        # project returns [0, 10] -> progress=10m, expected=10m -> score=1.0
        centerline = MagicMock()
        centerline.project.return_value = [0.0, 10.0]
        centerline.length = 100.0
        env_model.route_center_line = centerline
        ego_state = MagicMock()
        ego_state.dynamic_car_state.speed = 10.0
        env_model.ego_state = ego_state

        metric = ProgressMetric()
        result = metric.compute(si, env_model)

        np.testing.assert_array_almost_equal(result.scores, [1.0, 1.0])


# === TTCMetric ===


class TestTTCMetric:
    def test_no_collision(self, proposal_sampling, env_model):
        """No projected collisions -> score 1.0."""
        si = _make_scoring_input(proposal_sampling)
        obs = MagicMock()
        obs.collided_track_ids = []
        time_step_mock = MagicMock()
        time_step_mock.query.return_value = np.array([[], []])
        obs.__getitem__ = MagicMock(return_value=time_step_mock)
        env_model.observation = obs
        env_model.map_api = MagicMock()

        metric = TTCMetric()
        result = metric.compute(si, env_model)

        np.testing.assert_array_equal(result.scores, [1.0, 1.0])


# === ComfortMetric ===


class TestComfortMetric:
    @patch("mosaic.scorer.comfort_metric.ego_is_comfortable")
    def test_comfortable(self, mock_comfort, proposal_sampling, env_model):
        """All timesteps within comfort bounds -> score 1.0."""
        mock_comfort.return_value = np.ones(
            (N_PROPOSALS, N_POSES + 1), dtype=np.bool_
        )
        si = _make_scoring_input(proposal_sampling)

        metric = ComfortMetric()
        result = metric.compute(si, env_model)

        np.testing.assert_array_equal(result.scores, [1.0, 1.0])


# === Properties ===


class TestMetricProperties:
    def test_metric_names(self):
        assert NoAtFaultCollisionMetric().name == "no_at_fault_collision"
        assert DrivableAreaComplianceMetric().name == "drivable_area"
        assert DrivingDirectionComplianceMetric().name == "driving_direction"
        assert ProgressMetric().name == "progress"
        assert TTCMetric().name == "ttc"
        assert ComfortMetric().name == "comfort"

    def test_weighted_metric_weight(self):
        assert ProgressMetric(weight=5.0).weight == 5.0
        assert TTCMetric(weight=7.0).weight == 7.0
        assert ComfortMetric(weight=2.0).weight == 2.0
        assert ProgressMetric(weight=3.14).weight == 3.14
