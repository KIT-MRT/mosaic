import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import git
import numpy as np
import ray
from arbitration_graphs.typing import Time
from arbitration_graphs.verification import Result, Verifier
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from typing_extensions import override

import mosaic.common.utils.trajectory as trajectory_utils
from mosaic.common.command import Command
from mosaic.common.environment_model import EnvironmentModel
from mosaic.scorer.mosaic_scorer import MosaicScorer


class VerificationResult(Result):
    def __init__(self, is_ok: bool = True, message: str = ""):
        Result.__init__(self)
        self._is_ok: bool = is_ok
        self.message: str = message

    @override
    def is_ok(self) -> bool:
        return self._is_ok

    @override
    def __str__(self):
        return (
            "Verification successful."
            if self._is_ok
            else f"Verification failed: {self.message}"
        )


class TrajectoryVerifier(Verifier):
    @dataclass
    class Parameters:
        proposal_sampling: TrajectorySampling
        time_to_infraction_threshold: float = 2.0
        max_ego_speed: float = 5.0
        logging_enabled: bool = True
        log_base_dir: str = "logs"

    def __init__(self, parameters: Parameters):
        super().__init__()

        self.parameters: TrajectoryVerifier.Parameters = parameters
        self._simulator: PDMSimulator = PDMSimulator(self.parameters.proposal_sampling)
        self._scorer: MosaicScorer = MosaicScorer(self.parameters.proposal_sampling)

        self._logger: logging.Logger = self._setup_logger()
        self._ray_metadata: dict[str, object] = {}

    @override
    def analyze(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        command: Command,
    ) -> VerificationResult:
        # Convert trajectory to state array (1, T, state_dim)
        states = trajectory_utils.trajectory_to_state_array(
            command.trajectory, environment_model.parameters.proposal_sampling
        )
        states = np.expand_dims(states, axis=0)

        # Simulate closed-loop
        states = self._simulator.simulate_proposals(states, environment_model.ego_state)

        # Score to populate collision time indices
        self._scorer.score_proposals(
            states,
            environment_model.ego_state,
            environment_model.observation,
            environment_model.route_center_line,
            environment_model.route_lane_dict,
            environment_model.drivable_area_map,
            environment_model.map_api,
        )

        time_to_infraction = self._scorer.time_to_at_fault_collision(0)
        ego_speed: float = environment_model.ego_state.dynamic_car_state.speed

        is_ok = not (
            time_to_infraction <= self.parameters.time_to_infraction_threshold
            and ego_speed <= self.parameters.max_ego_speed
        )

        if self.parameters.logging_enabled:
            self._log_verification(time, command, time_to_infraction, ego_speed, is_ok)

        if is_ok:
            return VerificationResult(True)

        return VerificationResult(
            False,
            f"Imminent collision: time_to_infraction={time_to_infraction:.2f}s, "
            f"ego_speed={ego_speed:.2f}m/s",
        )

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"trajectory_verifier_{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.hasHandlers():
            worker_id = ray.get_runtime_context().get_worker_id()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            task_id = ray.get_runtime_context().get_task_id()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha

            log_dir = os.path.join(self.parameters.log_base_dir, sha)
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(
                log_dir, f"verification_{worker_id}_{task_id}.jsonl"
            )

            handler = logging.FileHandler(log_file, mode="a")
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            self._ray_metadata = {
                "task_id": task_id,
                "worker_id": worker_id,
            }

        return logger

    def _log_verification(
        self,
        time: Time,
        command: Command,
        time_to_infraction: float,
        ego_speed: float,
        is_ok: bool,
    ) -> None:
        assert isinstance(time, timedelta)
        entry = {
            "time": time.total_seconds(),
            "command": command.name,
            "time_to_infraction": float(time_to_infraction),
            "ego_speed": float(ego_speed),
            "result": "pass" if is_ok else "fail",
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
