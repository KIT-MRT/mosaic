import numpy as np
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    EgoAreaIndex,
)
from typing_extensions import override

from mosaic.common.environment_model import EnvironmentModel
from mosaic.scorer.abstract_metric import MetricResult, MultiplicativeMetric
from mosaic.scorer.scoring_input import ScoringInput


class DrivableAreaComplianceMetric(MultiplicativeMetric):
    @property
    @override
    def name(self) -> str:
        return "drivable_area"

    @override
    def compute(
        self, scoring_input: ScoringInput, environment_model: EnvironmentModel
    ) -> MetricResult:
        on_route_mask = ~scoring_input.ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
        fraction_in_drivable = on_route_mask.sum(axis=1) / float(
            scoring_input.proposal_sampling.num_poses + 1
        )
        fraction_in_drivable = np.clip(fraction_in_drivable, 0.0, 1.0)
        return MetricResult(scores=fraction_in_drivable)
