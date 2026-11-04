from datetime import timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from mosaic.core.verifier import TrajectoryVerifier
from mosaic.scorer import MetricResult


@pytest.fixture
def verifier():
    params = TrajectoryVerifier.Parameters(
        proposal_sampling=TrajectorySampling(time_horizon=4.0, interval_length=0.1),
    )
    v = TrajectoryVerifier(params)
    # Mock the expensive internals so we don't need real simulation data
    v._collision_metric = MagicMock()
    v._collision_metric.compute.return_value = MetricResult(
        scores=np.array([1.0]),
        metadata={
            "collision_time_idcs": np.array([np.inf]),
            "collision_tokens": [None],
        },
    )
    return v


def _make_command(name: str) -> MagicMock:
    cmd = MagicMock()
    cmd.name = name
    cmd.simulated_states = np.zeros((41, 11))
    return cmd


@patch("mosaic.core.verifier.ScoringInput.create")
class TestVerifierCaching:
    def test_same_command_same_time_returns_cached_result(
        self, mock_si_create, verifier
    ):
        """Calling analyze twice with the same time and command should only check collisions once."""
        time = timedelta(seconds=1.0)
        env = MagicMock()
        cmd = _make_command("pdm_closed")

        result1 = verifier.analyze(time, env, cmd)
        result2 = verifier.analyze(time, env, cmd)

        assert result1 is result2
        verifier._collision_metric.compute.assert_called_once()

    def test_different_commands_same_time_both_computed(
        self, mock_si_create, verifier
    ):
        """Different commands at the same time should each be computed."""
        time = timedelta(seconds=1.0)
        env = MagicMock()

        verifier.analyze(time, env, _make_command("flow_drive"))
        verifier.analyze(time, env, _make_command("pdm_closed"))

        assert verifier._collision_metric.compute.call_count == 2

    def test_same_command_different_time_recomputed(
        self, mock_si_create, verifier
    ):
        """Same command at a new timestep should be recomputed, not cached."""
        env = MagicMock()
        cmd = _make_command("pdm_closed")

        verifier.analyze(timedelta(seconds=1.0), env, cmd)
        verifier.analyze(timedelta(seconds=2.0), env, cmd)

        assert verifier._collision_metric.compute.call_count == 2

    def test_new_timestep_clears_old_cache(self, mock_si_create, verifier):
        """Advancing to a new timestep should clear all cached entries from the previous one."""
        env = MagicMock()

        verifier.analyze(timedelta(seconds=1.0), env, _make_command("flow_drive"))
        verifier.analyze(timedelta(seconds=1.0), env, _make_command("pdm_closed"))
        assert len(verifier._cache) == 2

        # New timestep should clear the cache
        verifier.analyze(timedelta(seconds=2.0), env, _make_command("flow_drive"))
        assert len(verifier._cache) == 1

    def test_cached_result_is_logged_only_once(
        self, mock_si_create, verifier
    ):
        """A cached hit should not produce a duplicate log entry."""
        time = timedelta(seconds=1.0)
        env = MagicMock()
        cmd = _make_command("pdm_closed")

        verifier.analyze(time, env, cmd)
        verifier.analyze(time, env, cmd)

        assert len(verifier._log_buffer) == 1
