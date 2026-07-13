import json
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import cast, final

import numpy as np
import numpy.typing as npt
from arbitration_graphs import BatchCostEstimator
from arbitration_graphs.typing import Time
from typing_extensions import override

from mosaic.core.command import Command
from mosaic.core.environment_model import EnvironmentModel
from mosaic.scorer import (
    MultiplicativeMetric,
    ProgressGateMetric,
    ProgressMetric,
    ScoringInput,
    WeightedMetric,
    aggregate_scores,
)


@final
class TrajectoryCostEstimator(BatchCostEstimator):
    @dataclass
    class Parameters:
        logging_enabled: bool = True

    def __init__(
        self,
        parameters: Parameters,
        multiplicative_metrics: list[MultiplicativeMetric],
        weighted_metrics: list[WeightedMetric],
    ) -> None:
        super().__init__()

        self.parameters: TrajectoryCostEstimator.Parameters = parameters

        progress_metric = next(
            m for m in weighted_metrics if isinstance(m, ProgressMetric)
        )
        self._multiplicative_metrics: list[MultiplicativeMetric] = [
            *multiplicative_metrics,
            ProgressGateMetric(progress_metric),
        ]
        self._weighted_metrics: list[WeightedMetric] = weighted_metrics

        self._log_buffer: list[dict[str, object]] = []

    @override
    def estimate_costs(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        candidates: list[BatchCostEstimator.Candidate],
    ) -> list[float]:
        states = np.stack(
            [cast(Command, c.command).simulated_states for c in candidates], axis=0
        )

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

    def flush_logs(self, log_dir: str, scenario_name: str) -> None:
        if not self._log_buffer:
            return

        path = os.path.join(log_dir, f"{scenario_name}_trajectory_costs.jsonl")
        with open(path, "w") as f:
            config_entry = {"type": "config", "weights": {m.name: m.weight for m in self._weighted_metrics}}
            _ = f.write(json.dumps(config_entry) + "\n")
            for entry in self._log_buffer:
                _ = f.write(json.dumps(entry) + "\n")

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
