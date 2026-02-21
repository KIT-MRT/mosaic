import numpy as np

from mosaic.scorer import MetricResult, aggregate_scores


def _result(scores):
    return MetricResult(scores=np.array(scores, dtype=np.float64))


class TestAggregateScores:
    def test_all_perfect(self):
        """All scores 1.0 -> final score 1.0."""
        multi = [_result([1.0, 1.0])]
        progress = _result([1.0, 1.0])
        weighted = [
            (5.0, progress),
            (7.0, _result([1.0, 1.0])),
            (2.0, _result([1.0, 1.0])),
        ]

        scores = aggregate_scores(multi, weighted, progress)

        np.testing.assert_array_almost_equal(scores, [1.0, 1.0])

    def test_safety_gate_zero(self):
        """One multiplicative score is 0.0 -> final score 0.0."""
        multi = [_result([0.0, 1.0]), _result([1.0, 1.0])]
        progress = _result([1.0, 1.0])
        weighted = [
            (5.0, progress),
            (7.0, _result([1.0, 1.0])),
        ]

        scores = aggregate_scores(multi, weighted, progress)

        assert scores[0] == 0.0
        assert scores[1] > 0.0

    def test_progress_zeroed_by_safety_gate(self):
        """When safety_gate is 0, progress contribution is zeroed."""
        multi = [_result([0.0, 1.0])]
        progress = _result([0.8, 0.8])
        weighted = [
            (5.0, progress),
            (7.0, _result([1.0, 1.0])),
        ]

        scores = aggregate_scores(multi, weighted, progress)

        # For proposal 0: safety_gate=0 -> everything is 0
        assert scores[0] == 0.0

    def test_weighted_average(self):
        """Known weights and scores -> verify weighted average math."""
        multi = [_result([1.0])]
        progress = _result([1.0])
        w_ttc = _result([0.5])
        weighted = [
            (5.0, progress),
            (7.0, w_ttc),
        ]

        scores = aggregate_scores(multi, weighted, progress)

        # safety_gate = 1.0
        # normalized_progress = 1.0, progress_gate = min(1.0/0.2, 1.0) = 1.0
        # weighted_avg = (5*1.0 + 7*0.5) / 12 = 8.5/12
        expected = 1.0 * 1.0 * (8.5 / 12.0)
        np.testing.assert_array_almost_equal(scores, [expected])

    def test_progress_gate_ramp(self):
        """Progress below threshold gets ramped, above threshold is 1.0."""
        multi = [_result([1.0, 1.0])]
        progress_low = _result([0.1, 0.4])  # 0.1 < 0.2, 0.4 > 0.2
        weighted = [
            (1.0, progress_low),
        ]

        scores = aggregate_scores(
            multi, weighted, progress_low, progress_gate_threshold=0.2
        )

        # For proposal 0: progress=0.1, gate=0.1/0.2=0.5
        #   weighted_avg = 0.1 (normalized progress, since only weight)
        #   final = 1.0 * 0.5 * 0.1 = 0.05
        # For proposal 1: progress=0.4, gate=min(0.4/0.2, 1.0)=1.0
        #   weighted_avg = 0.4
        #   final = 1.0 * 1.0 * 0.4 = 0.4
        np.testing.assert_array_almost_equal(scores, [0.05, 0.4])
