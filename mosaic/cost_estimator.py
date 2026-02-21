from dataclasses import dataclass
from datetime import timedelta
from typing import cast, final

import numpy as np
import numpy.typing as npt
from arbitration_graphs import BatchCostEstimator
from arbitration_graphs.typing import Time
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from typing_extensions import override

import mosaic.common.utils.trajectory as trajectory_utils
from mosaic.common.command import Command
from mosaic.common.environment_model import EnvironmentModel
from mosaic.scorer import (
    ComfortMetric,
    DrivableAreaComplianceMetric,
    DrivingDirectionComplianceMetric,
    MultiplicativeMetric,
    NoAtFaultCollisionMetric,
    ProgressGateMetric,
    ProgressMetric,
    ScoringInput,
    TTCMetric,
    WeightedMetric,
    aggregate_scores,
)


@final
class TrajectoryCostEstimator(BatchCostEstimator):
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        logging_enabled: bool

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()

        self.parameters: TrajectoryCostEstimator.Parameters = parameters
        self._simulator: PDMSimulator = PDMSimulator(
            self.parameters.trajectory_sampling
        )

        progress_metric = ProgressMetric()
        self._multiplicative_metrics: list[MultiplicativeMetric] = [
            NoAtFaultCollisionMetric(),
            DrivableAreaComplianceMetric(),
            DrivingDirectionComplianceMetric(),
            ProgressGateMetric(progress_metric),
        ]
        self._weighted_metrics: list[WeightedMetric] = [
            progress_metric,
            TTCMetric(),
            ComfortMetric(),
        ]

        self._log_buffer: list[dict[str, object]] = []

    @override
    def estimate_costs(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        candidates: list[BatchCostEstimator.Candidate],
    ) -> list[float]:
        trajectories_list = [cast(Command, c.command).trajectory for c in candidates]

        states_list = [
            trajectory_utils.trajectory_to_state_array(
                traj, environment_model.parameters.proposal_sampling
            )
            for traj in trajectories_list
        ]

        states = np.stack(states_list, axis=0)

        states = self._simulator.simulate_proposals(states, environment_model.ego_state)

        scoring_input = ScoringInput.create(
            states,
            environment_model,
        )

        multi_results = [
            m.compute(scoring_input, environment_model)
            for m in self._multiplicative_metrics
        ]
        weighted_results = [
            (m.weight, m.compute(scoring_input, environment_model))
            for m in self._weighted_metrics
        ]

        scores = aggregate_scores(multi_results, weighted_results)

        if scores.shape != (len(candidates),):
            raise ValueError(
                f"Expected score shape ({len(candidates)},), got {scores.shape}"
            )

        if self.parameters.logging_enabled:
            self._log_scoring(
                time,
                candidates,
                scores,
                multi_results,
                weighted_results,
            )

        costs = [-score for score in scores]

        return costs

    def _log_scoring(
        self,
        time: Time,
        candidates: list[BatchCostEstimator.Candidate],
        scores: npt.NDArray[np.float64],
        multi_results: list,
        weighted_results: list[tuple[float, object]],
    ) -> None:
        # Build a name->result map for weighted metrics by pairing with metric list
        weighted_named: list[tuple[str, object]] = []
        for metric, (_, result) in zip(self._weighted_metrics, weighted_results):
            weighted_named.append((metric.name, result))

        proposals_log: list[dict[str, object]] = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            command = cast(Command, candidate.command)

            components: dict[str, float] = {}
            for j, m in enumerate(self._multiplicative_metrics):
                components[m.name] = float(multi_results[j].scores[i])
            for name, result in weighted_named:
                components[name] = float(result.scores[i])

            proposals_log.append(
                {
                    "command": command.name,
                    "is_active": candidate.is_active,
                    "final_score": float(score),
                    "components": components,
                }
            )

        assert isinstance(time, timedelta)
        entry: dict[str, object] = {
            "time": time.total_seconds(),
            "num_candidates": len(candidates),
            "proposals": proposals_log,
        }

        self._log_buffer.append(entry)

    def __getstate__(self) -> dict[str, object]:
        """
        Custom getstate to fix pickling since this is a class that inherits from a C++ object
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Custom setstate to fix pickling since this is a class that inherits from a C++ object
        """
        super().__init__()
        self.__dict__.update(state)
