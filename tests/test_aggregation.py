import numpy as np

from mosaic.scorer import MetricResult, aggregate_scores


def _result(scores):
    return MetricResult(scores=np.array(scores, dtype=np.float64))


class TestAggregateScores:
    def test_all_perfect(self):
        """All scores 1.0 -> final score 1.0."""
        multi = [_result([1.0, 1.0])]
        weighted = [
            (5.0, _result([1.0, 1.0])),
            (7.0, _result([1.0, 1.0])),
            (2.0, _result([1.0, 1.0])),
        ]

        scores = aggregate_scores(multi, weighted)

        np.testing.assert_array_almost_equal(scores, [1.0, 1.0])

    def test_safety_gate_zero(self):
        """One multiplicative score is 0.0 -> final score 0.0."""
        multi = [_result([0.0, 1.0]), _result([1.0, 1.0])]
        weighted = [
            (5.0, _result([1.0, 1.0])),
            (7.0, _result([1.0, 1.0])),
        ]

        scores = aggregate_scores(multi, weighted)

        assert scores[0] == 0.0
        assert scores[1] > 0.0

    def test_safety_gate_zeroes_everything(self):
        """When safety_gate is 0, final score is 0 regardless of weighted scores."""
        multi = [_result([0.0, 1.0])]
        weighted = [
            (5.0, _result([0.8, 0.8])),
            (7.0, _result([1.0, 1.0])),
        ]

        scores = aggregate_scores(multi, weighted)

        assert scores[0] == 0.0

    def test_weighted_average(self):
        """Known weights and scores -> verify weighted average math."""
        multi = [_result([1.0])]
        w_progress = _result([1.0])
        w_ttc = _result([0.5])
        weighted = [
            (5.0, w_progress),
            (7.0, w_ttc),
        ]

        scores = aggregate_scores(multi, weighted)

        # safety_gate = 1.0
        # weighted_avg = (5*1.0 + 7*0.5) / 12 = 8.5/12
        expected = 1.0 * (8.5 / 12.0)
        np.testing.assert_array_almost_equal(scores, [expected])

    def test_progress_gate_in_multiplicative(self):
        """Progress gate as a multiplicative metric ramps the score."""
        # Simulate progress_gate values: 0.5 (below threshold) and 1.0 (above)
        multi = [_result([1.0, 1.0]), _result([0.5, 1.0])]
        weighted = [
            (1.0, _result([0.8, 0.8])),
        ]

        scores = aggregate_scores(multi, weighted)

        # Proposal 0: 1.0 * 0.5 * 0.8 = 0.4
        # Proposal 1: 1.0 * 1.0 * 0.8 = 0.8
        np.testing.assert_array_almost_equal(scores, [0.4, 0.8])
