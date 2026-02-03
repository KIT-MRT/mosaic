import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import cast, final

import git
import numpy as np
import numpy.typing as npt
import ray
from arbitration_graphs import BatchCostEstimator
from arbitration_graphs.typing import Time
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from typing_extensions import override

import arbitration_pdm.common.utils.trajectory as trajectory_utils
from arbitration_pdm.common.command import Command
from arbitration_pdm.common.environment_model import EnvironmentModel
from arbitration_pdm.scorer.pdm_scorer import PDMScorer


@final
class TrajectoryCostEstimator(BatchCostEstimator):
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        logging_enabled: bool
        log_base_dir: str = "logs"

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()

        self.parameters: TrajectoryCostEstimator.Parameters = parameters
        self._simulator: PDMSimulator = PDMSimulator(
            self.parameters.trajectory_sampling
        )
        self._scorer: PDMScorer = PDMScorer(self.parameters.trajectory_sampling)

        self._logger: logging.Logger = self._setup_logger()
        self._ray_metadata: dict[str, object] = {}

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

    def _setup_logger(self) -> logging.Logger:
        """
        Create a per-worker logger that writes JSON lines including Ray metadata.
        """
        logger = logging.getLogger(f"trajectory_cost_estimator_{id(self)}")
        logger.setLevel(logging.INFO)
        # Disable propagation to parent loggers (prevents printing to stdout)
        logger.propagate = False

        if not logger.hasHandlers():
            # Get Ray metadata
            worker_id = ray.get_runtime_context().get_worker_id()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            task_id = ray.get_runtime_context().get_task_id()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

            # Get current git commit hash
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha

            log_dir = os.path.join(self.parameters.log_base_dir, sha)

            # Ensure log directory exists
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(
                log_dir, f"trajectory_costs_{worker_id}_{task_id}.jsonl"
            )

            handler = logging.FileHandler(log_file, mode="a")
            formatter = logging.Formatter("%(message)s")  # we log full JSON ourselves
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # Store metadata for reuse in log entries
            self._ray_metadata = {
                "task_id": task_id,
                "worker_id": worker_id,
            }

        return logger

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
        entry = {
            "time": time.total_seconds(),
            "num_candidates": len(candidates),
            "proposals": proposals_log,
            **self._ray_metadata,
        }

        self._logger.info(json.dumps(entry))

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
