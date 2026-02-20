from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from mosaic.scorer.scoring_input import ScoringInput


N_PROPOSALS = 3
N_POSES = 10
STATE_DIM = 11


@pytest.fixture
def proposal_sampling():
    return TrajectorySampling(
        time_horizon=1.0, interval_length=0.1
    )  # num_poses = 10


@pytest.fixture
def mock_states():
    return np.zeros((N_PROPOSALS, N_POSES + 1, STATE_DIM), dtype=np.float64)


@pytest.fixture
def mock_ego_state():
    ego = MagicMock()
    ego.car_footprint.vehicle_parameters = MagicMock()
    return ego


@pytest.fixture
def mock_drivable_area_map():
    dam = MagicMock()
    dam.__len__ = MagicMock(return_value=2)
    dam.tokens = ["lane_1", "lane_2"]
    dam.points_in_polygons.return_value = np.ones(
        (2, N_PROPOSALS * (N_POSES + 1) * 5), dtype=np.bool_
    )
    return dam


@pytest.fixture
def mock_route_lane_dict():
    return {"lane_1": MagicMock()}


@patch(
    "mosaic.scorer.scoring_input.state_array_to_coords_array",
    return_value=np.zeros((N_PROPOSALS, N_POSES + 1, 5, 2), dtype=np.float64),
)
@patch(
    "mosaic.scorer.scoring_input.coords_array_to_polygon_array",
    return_value=np.empty((N_PROPOSALS, N_POSES + 1), dtype=np.object_),
)
@patch("mosaic.scorer.scoring_input.StateIndex")
class TestScoringInput:
    def test_create_shapes(
        self,
        mock_state_index,
        mock_poly_fn,
        mock_coords_fn,
        mock_states,
        mock_ego_state,
        mock_drivable_area_map,
        mock_route_lane_dict,
        proposal_sampling,
    ):
        mock_state_index.size.return_value = STATE_DIM
        si = ScoringInput.create(
            mock_states,
            mock_ego_state,
            mock_drivable_area_map,
            mock_route_lane_dict,
            proposal_sampling,
        )

        assert si.ego_coords.shape == (N_PROPOSALS, N_POSES + 1, 5, 2)
        assert si.ego_polygons.shape == (N_PROPOSALS, N_POSES + 1)
        assert si.ego_areas.shape == (N_PROPOSALS, N_POSES + 1, 3)

    def test_create_immutable(
        self,
        mock_state_index,
        mock_poly_fn,
        mock_coords_fn,
        mock_states,
        mock_ego_state,
        mock_drivable_area_map,
        mock_route_lane_dict,
        proposal_sampling,
    ):
        mock_state_index.size.return_value = STATE_DIM
        si = ScoringInput.create(
            mock_states,
            mock_ego_state,
            mock_drivable_area_map,
            mock_route_lane_dict,
            proposal_sampling,
        )

        with pytest.raises(FrozenInstanceError):
            si.states = np.zeros_like(si.states)

    def test_num_proposals(
        self,
        mock_state_index,
        mock_poly_fn,
        mock_coords_fn,
        mock_states,
        mock_ego_state,
        mock_drivable_area_map,
        mock_route_lane_dict,
        proposal_sampling,
    ):
        mock_state_index.size.return_value = STATE_DIM
        si = ScoringInput.create(
            mock_states,
            mock_ego_state,
            mock_drivable_area_map,
            mock_route_lane_dict,
            proposal_sampling,
        )

        assert si.num_proposals == N_PROPOSALS
