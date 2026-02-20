import numpy as np
from shapely import Point
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)
from typing_extensions import override

from mosaic.common.environment_model import EnvironmentModel
from mosaic.scorer.abstract_metric import MetricResult, WeightedMetric
from mosaic.scorer.scoring_input import ScoringInput


class ProgressMetric(WeightedMetric):
    def __init__(
        self,
        weight: float = 5.0,
        ref_speed_factor: float = 1.0,
        fallback_max_meters: float = 10.0,
        min_expected_meters: float = 0.1,
        cap_to_centerline: bool = True,
    ) -> None:
        super().__init__(weight)
        self._ref_speed_factor = ref_speed_factor
        self._fallback_max_meters = fallback_max_meters
        self._min_expected_meters = min_expected_meters
        self._cap_to_centerline = cap_to_centerline

    @property
    @override
    def name(self) -> str:
        return "progress"

    @override
    def compute(
        self, scoring_input: ScoringInput, environment_model: EnvironmentModel
    ) -> MetricResult:
        n = scoring_input.num_proposals
        centerline = environment_model.route_center_line

        progress_in_meter = np.zeros(n, dtype=np.float64)
        for proposal_idx in range(n):
            start_point = Point(
                *scoring_input.ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER]
            )
            end_point = Point(
                *scoring_input.ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER]
            )
            progress = centerline.project([start_point, end_point])
            progress_in_meter[proposal_idx] = progress[1] - progress[0]

        horizon_time_s = (
            scoring_input.proposal_sampling.num_poses
            * scoring_input.proposal_sampling.interval_length
        )
        ego_speed = float(environment_model.ego_state.dynamic_car_state.speed)
        expected_by_speed = ego_speed * horizon_time_s * self._ref_speed_factor

        progress_scores = np.zeros(n, dtype=np.float64)
        for proposal_idx in range(n):
            expected = expected_by_speed

            if self._cap_to_centerline and centerline is not None:
                try:
                    proj_start = centerline.project(
                        Point(
                            *scoring_input.ego_coords[
                                proposal_idx, 0, BBCoordsIndex.CENTER
                            ]
                        )
                    )
                    centerline_remaining = max(
                        0.0, centerline.length - float(proj_start)
                    )
                    expected = min(expected, centerline_remaining)
                except Exception:
                    pass

            expected = min(expected, self._fallback_max_meters)
            expected = max(expected, self._min_expected_meters)

            progress_scores[proposal_idx] = float(
                np.clip(progress_in_meter[proposal_idx] / expected, 0.0, 1.0)
            )

        return MetricResult(scores=progress_scores)
