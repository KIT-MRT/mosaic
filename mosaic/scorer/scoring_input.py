from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    EgoAreaIndex,
    StateIndex,
)

from mosaic.core.environment_model import EnvironmentModel


@dataclass(frozen=True)
class ScoringInput:
    """Frozen dataclass holding trajectory-derived geometry for scoring."""

    states: npt.NDArray[np.float64]  # (n, T, 11)
    ego_coords: npt.NDArray[np.float64]  # (n, T, 5, 2)
    ego_polygons: npt.NDArray[np.object_]  # (n, T)
    ego_areas: npt.NDArray[np.bool_]  # (n, T, 3)
    proposal_sampling: TrajectorySampling

    @property
    def num_proposals(self) -> int:
        return self.states.shape[0]

    @classmethod
    def create(
        cls,
        states: npt.NDArray[np.float64],
        environment_model: EnvironmentModel,
    ) -> ScoringInput:
        assert states.ndim == 3
        assert (
            states.shape[1]
            == environment_model.parameters.proposal_sampling.num_poses + 1
        )
        assert states.shape[2] == StateIndex.size()

        n_proposals = states.shape[0]

        ego_coords = state_array_to_coords_array(
            states, environment_model.ego_state.car_footprint.vehicle_parameters
        )
        ego_polygons = coords_array_to_polygon_array(ego_coords)

        ego_areas = np.zeros(
            (
                n_proposals,
                environment_model.parameters.proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        _calculate_ego_areas(
            ego_coords,
            environment_model.drivable_area_map,
            environment_model.route_lane_dict,
            ego_areas,
        )

        return cls(
            states=states,
            ego_coords=ego_coords,
            ego_polygons=ego_polygons,
            ego_areas=ego_areas,
            proposal_sampling=environment_model.parameters.proposal_sampling,
        )


def _calculate_ego_areas(
    ego_coords: npt.NDArray[np.float64],
    drivable_area_map: PDMOccupancyMap,
    route_lane_dict: Dict[str, object],
    ego_areas: npt.NDArray[np.bool_],
) -> None:
    """Populate ego_areas in-place with area classifications."""
    n_proposals, n_horizon, n_points, _ = ego_coords.shape
    coordinates = ego_coords.reshape(n_proposals * n_horizon * n_points, 2)

    in_polygons = drivable_area_map.points_in_polygons(coordinates)
    in_polygons = in_polygons.reshape(
        len(drivable_area_map), n_proposals, n_horizon, n_points
    ).transpose(1, 2, 0, 3)

    drivable_area_on_route_idcs: List[int] = [
        idx
        for idx, token in enumerate(drivable_area_map.tokens)
        if token in route_lane_dict.keys()
    ]

    corners_in_polygon = in_polygons[..., :-1]
    center_in_polygon = in_polygons[..., -1]

    # in_multiple_lanes
    batch_multiple_lanes_mask = (corners_in_polygon.sum(axis=-1) > 0).sum(axis=-1) > 1
    batch_not_single_lanes_mask = np.all(corners_in_polygon.sum(axis=-1) != 4, axis=-1)
    multiple_lanes_mask = np.logical_and(
        batch_multiple_lanes_mask, batch_not_single_lanes_mask
    )
    ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

    # in_nondrivable_area
    batch_nondrivable_area_mask = (corners_in_polygon.sum(axis=-2) > 0).sum(axis=-1) < 4
    ego_areas[batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA] = True

    # in_oncoming_traffic
    batch_oncoming_traffic_mask = (
        center_in_polygon[..., drivable_area_on_route_idcs].sum(axis=-1) == 0
    )
    ego_areas[batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC] = True
