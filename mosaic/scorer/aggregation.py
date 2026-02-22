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
    multiplicative_score = np.ones_like(multiplicative_results[0].scores)
    for r in multiplicative_results:
        multiplicative_score = multiplicative_score * r.scores

    # weighted average
    total_weight = 0.0
    weighted_sum = np.zeros_like(multiplicative_score)
    for weight, result in weighted_results:
        weighted_sum += weight * result.scores
        total_weight += weight

    weighted_average = weighted_sum / total_weight

    return multiplicative_score * weighted_average
