import numpy as np
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
)
from typing_extensions import override

from mosaic.core.environment_model import EnvironmentModel
from mosaic.scorer.abstract_metric import MetricResult, MultiplicativeMetric
from mosaic.scorer.scoring_input import ScoringInput


class DrivingDirectionComplianceMetric(MultiplicativeMetric):
    def __init__(
        self,
        compliance_threshold: float = 2.0,
        violation_threshold: float = 6.0,
    ) -> None:
        self._compliance_threshold = compliance_threshold
        self._violation_threshold = violation_threshold

    @property
    @override
    def name(self) -> str:
        return "driving_direction"

    @override
    def compute(
        self, scoring_input: ScoringInput, environment_model: EnvironmentModel
    ) -> MetricResult:
        n = scoring_input.num_proposals
        center_coordinates = scoring_input.ego_coords[:, :, BBCoordsIndex.CENTER]

        cum_progress = np.zeros(
            (n, scoring_input.proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        cum_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        oncoming_traffic_masks = scoring_input.ego_areas[
            :, :, EgoAreaIndex.ONCOMING_TRAFFIC
        ]
        cum_progress[~oncoming_traffic_masks] = 0.0

        scores = np.ones(n, dtype=np.float64)

        for proposal_idx in range(n):
            oncoming_traffic_progress = cum_progress[proposal_idx]
            oncoming_traffic_mask = oncoming_traffic_masks[proposal_idx]

            oncoming_progress_splits = np.split(
                oncoming_traffic_progress,
                np.where(np.diff(oncoming_traffic_mask))[0] + 1,
            )

            max_oncoming_traffic_progress = max(
                oncoming_progress.sum()
                for oncoming_progress in oncoming_progress_splits
            )

            if max_oncoming_traffic_progress < self._compliance_threshold:
                scores[proposal_idx] = 1.0
            elif max_oncoming_traffic_progress < self._violation_threshold:
                scores[proposal_idx] = 0.5
            else:
                scores[proposal_idx] = 0.0

        return MetricResult(scores=scores)
