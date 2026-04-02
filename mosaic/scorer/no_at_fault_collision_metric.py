import copy

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import (
    get_collision_type,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    EgoAreaIndex,
)
from typing_extensions import override

from mosaic.core.environment_model import EnvironmentModel
from mosaic.scorer.abstract_metric import MetricResult, MultiplicativeMetric
from mosaic.scorer.scoring_input import ScoringInput


class NoAtFaultCollisionMetric(MultiplicativeMetric):
    @property
    @override
    def name(self) -> str:
        return "no_at_fault_collision"

    @override
    def compute(
        self, scoring_input: ScoringInput, environment_model: EnvironmentModel
    ) -> MetricResult:
        n = scoring_input.num_proposals
        observation = environment_model.observation

        no_collision_scores = np.ones(n, dtype=np.float64)
        collision_time_idcs = np.full(n, np.inf, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(observation.collided_track_ids)
            for proposal_idx in range(n)
        }

        for time_idx in range(scoring_input.proposal_sampling.num_poses + 1):
            ego_polygons = scoring_input.ego_polygons[:, time_idx]
            intersecting = observation[time_idx].query(
                ego_polygons, predicate="intersects"
            )

            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = observation[time_idx].tokens[geometry_idx]
                if (observation.red_light_token in token) or (
                    token in proposal_collided_track_ids[proposal_idx]
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

                tracked_object = observation.unique_objects[token]

                collision_type: CollisionType = get_collision_type(
                    scoring_input.states[proposal_idx, time_idx],
                    scoring_input.ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    observation[time_idx][token],
                )
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )

                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    ego_poly = scoring_input.ego_polygons[proposal_idx, time_idx]
                    track_poly = tracked_object.box.geometry

                    overlap_area = ego_poly.intersection(track_poly).area
                    ego_area = ego_poly.area + 1e-6
                    overlap_ratio = np.clip(overlap_area / ego_area, 0.0, 1.0)

                    collision_score = 1.0 - overlap_ratio

                    if tracked_object.tracked_object_type not in AGENT_TYPES:
                        collision_score = 0.5 + 0.5 * collision_score

                    no_collision_scores[proposal_idx] = min(
                        no_collision_scores[proposal_idx],
                        float(np.clip(collision_score, 0.0, 1.0)),
                    )
                    collision_time_idcs[proposal_idx] = min(
                        time_idx, collision_time_idcs[proposal_idx]
                    )
                else:
                    proposal_collided_track_ids[proposal_idx].append(token)

        return MetricResult(
            scores=no_collision_scores,
            metadata={"collision_time_idcs": collision_time_idcs},
        )
