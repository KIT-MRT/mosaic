import numpy as np
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)
from typing_extensions import override

from mosaic.common.environment_model import EnvironmentModel
from mosaic.scorer.abstract_metric import MetricResult, WeightedMetric
from mosaic.scorer.scoring_input import ScoringInput


class ComfortMetric(WeightedMetric):
    def __init__(self, weight: float = 2.0) -> None:
        super().__init__(weight)

    @property
    @override
    def name(self) -> str:
        return "comfort"

    @override
    def compute(
        self, scoring_input: ScoringInput, environment_model: EnvironmentModel
    ) -> MetricResult:
        time_point_s = (
            np.arange(0, scoring_input.proposal_sampling.num_poses + 1).astype(
                np.float64
            )
            * scoring_input.proposal_sampling.interval_length
        )
        is_comfortable = ego_is_comfortable(scoring_input.states, time_point_s)
        scores = np.all(is_comfortable, axis=-1).astype(np.float64)
        return MetricResult(scores=scores)
