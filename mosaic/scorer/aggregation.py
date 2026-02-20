import numpy as np
import numpy.typing as npt

from mosaic.scorer.abstract_metric import MetricResult


def aggregate_scores(
    multiplicative_results: list[MetricResult],
    weighted_results: list[tuple[float, MetricResult]],
    progress_result: MetricResult,
    progress_gate_threshold: float = 0.2,
) -> npt.NDArray[np.float64]:
    """
    Aggregate metric results into final proposal scores.

    :param multiplicative_results: Safety-gate metric results (multiplied together).
    :param weighted_results: Pairs of (weight, result) for weighted-average metrics.
        Must include the progress entry with its weight.
    :param progress_result: The progress MetricResult (passed explicitly for cross-dependency).
    :param progress_gate_threshold: Progress below this is penalized as a soft gate.
    :return: (n_proposals,) final scores.
    """
    # safety_gate = product of all multiplicative scores
    safety_gate = np.ones_like(multiplicative_results[0].scores)
    for r in multiplicative_results:
        safety_gate = safety_gate * r.scores

    # normalize progress: zero out where safety_gate is zero
    normalized_progress = progress_result.scores.copy()
    normalized_progress[safety_gate == 0.0] = 0.0

    # progress gate: ramp from 0 to 1 over [0, threshold]
    progress_gate = np.minimum(normalized_progress / progress_gate_threshold, 1.0)

    # weighted average
    total_weight = 0.0
    weighted_sum = np.zeros_like(safety_gate)
    for weight, result in weighted_results:
        # use normalized_progress for the progress entry
        scores = (
            normalized_progress if result is progress_result else result.scores
        )
        weighted_sum += weight * scores
        total_weight += weight

    weighted_average = weighted_sum / total_weight

    return safety_gate * progress_gate * weighted_average
