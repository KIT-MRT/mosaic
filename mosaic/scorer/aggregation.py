import numpy as np
import numpy.typing as npt

from mosaic.scorer.abstract_metric import MetricResult


def aggregate_scores(
    multiplicative_results: list[MetricResult],
    weighted_results: list[tuple[float, MetricResult]],
) -> npt.NDArray[np.float64]:
    """
    Aggregate metric results into final proposal scores.

    :param multiplicative_results: Safety-gate metric results (multiplied together).
    :param weighted_results: Pairs of (weight, result) for weighted-average metrics.
    :return: (n_proposals,) final scores.
    """
    # safety_gate = product of all multiplicative scores
    safety_gate = np.ones_like(multiplicative_results[0].scores)
    for r in multiplicative_results:
        safety_gate = safety_gate * r.scores

    # weighted average
    total_weight = 0.0
    weighted_sum = np.zeros_like(safety_gate)
    for weight, result in weighted_results:
        weighted_sum += weight * result.scores
        total_weight += weight

    weighted_average = weighted_sum / total_weight

    return safety_gate * weighted_average
