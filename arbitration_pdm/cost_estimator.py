import json
import logging
import os
from dataclasses import dataclass
from typing import final

import numpy as np
import ray
from arbitration_graphs import CostEstimator
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
class TrajectoryCostEstimator(CostEstimator):
    @dataclass
    class Parameters:
        trajectory_sampling: TrajectorySampling
        logging_enabled: bool
        log_dir: str = "logs"

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
    def estimate_cost(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        command: Command,
        is_active: bool,
    ) -> float:
        trajectories_list = [command.trajectory]

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
        if scores.shape != (1,):
            raise ValueError(
                f"Expected score shape (1,), got {scores.shape} when scoring trajectory."
            )

        if self.parameters.logging_enabled:
            self._log_scoring(time, command, is_active, float(scores[0]))

        return -scores[0]

    def _setup_logger(self) -> logging.Logger:
        """
        Create a per-worker logger that writes JSON lines including Ray metadata.
        """
        logger = logging.getLogger(f"trajectory_cost_estimator_{id(self)}")
        logger.setLevel(logging.INFO)
        # Disable propagation to parent loggers (prevents printing to stdout)
        logger.propagate = False

        if not logger.hasHandlers():
            # Ensure log directory exists
            os.makedirs(self.parameters.log_dir, exist_ok=True)

            # Get Ray metadata
            worker_id = ray.get_runtime_context().get_worker_id()
            task_id = ray.get_runtime_context().get_task_id()

            log_file = os.path.join(
                self.parameters.log_dir, f"trajectory_costs_{worker_id}_{task_id}.jsonl"
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
        command: Command,
        is_active: bool,
        score: float,
    ) -> None:

        log = {
            "time": time.total_seconds(),
            "command": command.name,
            "is_active": is_active,
            "final_score": score,
            "components": {
                "multi_metrics": list(
                    map(float, self._scorer._multi_metrics[:, 0].tolist())
                ),
                "weighted_metrics": list(
                    map(float, self._scorer._weighted_metrics[:, 0].tolist())
                ),
            },
            **self._ray_metadata,
        }

        self._logger.info(json.dumps(log))

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
