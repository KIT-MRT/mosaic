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
from mosaic.scorer.mosaic_scorer import MosaicScorer


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
        self._scorer: MosaicScorer = MosaicScorer(self.parameters.trajectory_sampling)

        self._log_buffer: list[dict[str, object]] = []

    @override
    def estimate_costs(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        candidates: list[BatchCostEstimator.Candidate],
    ) -> list[float]:
        trajectories_list = [cast(Command, c.command).trajectory for c in candidates]

        # Convert each trajectory to a (T, state_dim) array
        states_list = [
            trajectory_utils.trajectory_to_state_array(
                traj, environment_model.parameters.proposal_sampling
            )
            for traj in trajectories_list
        ]

        # Stack into shape (n, T, state_dim)
        states = np.stack(states_list, axis=0)

        # simulate closed-loop execution traces starting from the real ego state
        states = self._simulator.simulate_proposals(states, environment_model.ego_state)

        scores = self._scorer.score_proposals(
            states,
            environment_model.ego_state,
            environment_model.observation,
            environment_model.route_center_line,
            environment_model.route_lane_dict,
            environment_model.drivable_area_map,
            environment_model.map_api,
        )

        if scores.shape != (len(candidates),):
            raise ValueError(
                f"Expected score shape ({len(candidates)},), got {scores.shape}"
            )

        if self.parameters.logging_enabled:
            self._log_scoring(time, candidates, scores)

        costs = [-score for score in scores]

        return costs

    def _log_scoring(
        self,
        time: Time,
        candidates: list[BatchCostEstimator.Candidate],
        scores: npt.NDArray[np.float64],
    ) -> None:
        proposals_log: list[dict[str, object]] = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            command = cast(Command, candidate.command)

            # TODO: Add an interface to the scorer to get the individual metric components
            # instead of reaching into protected members here
            proposals_log.append(
                {
                    "command": command.name,
                    "is_active": candidate.is_active,
                    "final_score": float(score),
                    "components": {
                        "multi_metrics": list(
                            map(float, self._scorer._multi_metrics[:, i].tolist())
                        ),
                        "weighted_metrics": list(
                            map(float, self._scorer._weighted_metrics[:, i].tolist())
                        ),
                    },
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
