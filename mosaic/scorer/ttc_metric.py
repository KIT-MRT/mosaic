import copy

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.idm.utils import (
    is_agent_ahead,
    is_agent_behind,
)
from shapely import creation
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    StateIndex,
)
from typing_extensions import override

from mosaic.core.environment_model import EnvironmentModel
from mosaic.scorer.abstract_metric import MetricResult, WeightedMetric
from mosaic.scorer.scoring_input import ScoringInput


class TTCMetric(WeightedMetric):
    def __init__(
        self,
        weight: float = 7.0,
        time_horizon: float = 3.0,
        stopped_speed_threshold: float = 5e-03,
    ) -> None:
        super().__init__(weight)
        self._time_horizon = time_horizon
        self._stopped_speed_threshold = stopped_speed_threshold

    @property
    @override
    def name(self) -> str:
        return "ttc"

    @override
    def compute(
        self, scoring_input: ScoringInput, environment_model: EnvironmentModel
    ) -> MetricResult:
        n = scoring_input.num_proposals
        observation = environment_model.observation
        map_api = environment_model.map_api

        ttc_scores = np.ones(n, dtype=np.float64)
        ttc_time_idcs = np.full(n, np.inf, dtype=np.float64)

        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(observation.collided_track_ids)
            for proposal_idx in range(n)
        }

        ttc_steps = int(
            self._time_horizon / scoring_input.proposal_sampling.interval_length
        )
        future_time_idcs = np.arange(0, ttc_steps + 1, 1)
        n_future_steps = len(future_time_idcs)

        coords_exterior = scoring_input.ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[
            :, :, BBCoordsIndex.FRONT_LEFT, :
        ]
        coords_exterior_time_steps = np.repeat(
            coords_exterior[:, :, None], n_future_steps, axis=2
        )

        speeds = np.hypot(
            scoring_input.states[..., StateIndex.VELOCITY_X],
            scoring_input.states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(scoring_input.states[..., StateIndex.HEADING]) * speeds,
                np.sin(scoring_input.states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = (
                float(future_time_idx) * scoring_input.proposal_sampling.interval_length
            )
            coords_exterior_time_steps[:, :, idx] = (
                coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            )

        polygons = creation.polygons(coords_exterior_time_steps)

        from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            EgoAreaIndex,
        )

        for time_idx in range(scoring_input.proposal_sampling.num_poses + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = observation[current_time_idx].query(
                    polygons_at_time_step, predicate="intersects"
                )

                if len(intersecting) == 0:
                    continue

                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = observation[current_time_idx].tokens[geometry_idx]
                    if (
                        (observation.red_light_token in token)
                        or (token in temp_collided_track_ids[proposal_idx])
                        or (
                            speeds[proposal_idx, time_idx]
                            < self._stopped_speed_threshold
                        )
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        scoring_input.ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES
                        ]
                        or scoring_input.ego_areas[
                            proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                        ]
                    )
                    ego_rear_axle: StateSE2 = StateSE2(
                        *scoring_input.states[
                            proposal_idx, time_idx, StateIndex.STATE_SE2
                        ]
                    )

                    centroid = observation[current_time_idx][token].centroid
                    track_heading = observation.unique_objects[token].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or map_api.is_in_layer(
                                ego_rear_axle, layer=SemanticMapLayer.INTERSECTION
                            )
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_seconds = (
                            float(future_time_idx)
                            * scoring_input.proposal_sampling.interval_length
                        )
                        ttc_score = float(
                            np.clip(ttc_seconds / self._time_horizon, 0.0, 1.0)
                        )
                        ttc_scores[proposal_idx] = min(
                            ttc_scores[proposal_idx], ttc_score
                        )
                        ttc_time_idcs[proposal_idx] = min(
                            time_idx, ttc_time_idcs[proposal_idx]
                        )
                    else:
                        temp_collided_track_ids[proposal_idx].append(token)

        return MetricResult(
            scores=ttc_scores,
            metadata={"ttc_time_idcs": ttc_time_idcs},
        )
