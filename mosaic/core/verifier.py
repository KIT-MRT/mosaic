import json
import os
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from arbitration_graphs.typing import Time
from arbitration_graphs.verification import Result, Verifier
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from typing_extensions import override

import mosaic.utils.trajectory as trajectory_utils
from mosaic.core.command import Command
from mosaic.core.environment_model import EnvironmentModel
from mosaic.scorer import NoAtFaultCollisionMetric, ScoringInput


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

    def __init__(self, parameters: Parameters):
        super().__init__()

        self.parameters: TrajectoryVerifier.Parameters = parameters
        self._simulator: PDMSimulator = PDMSimulator(self.parameters.proposal_sampling)
        self._collision_metric: NoAtFaultCollisionMetric = NoAtFaultCollisionMetric()

        self._log_buffer: list[dict[str, object]] = []
        self._cache: dict[str, VerificationResult] = {}
        self._cache_time: float = float("nan")

    @override
    def analyze(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        command: Command,
    ) -> VerificationResult:
        assert isinstance(time, timedelta)
        t = time.total_seconds()
        if t != self._cache_time:
            self._cache.clear()
            self._cache_time = t
        if command.name in self._cache:
            return self._cache[command.name]

        # Convert trajectory to state array (1, T, state_dim)
        states = trajectory_utils.trajectory_to_state_array(
            command.trajectory, environment_model.parameters.proposal_sampling
        )
        states = np.expand_dims(states, axis=0)

        # Simulate closed-loop
        states = self._simulator.simulate_proposals(states, environment_model.ego_state)

        # Create scoring input and run only collision metric
        scoring_input = ScoringInput.create(states, environment_model)

        result = self._collision_metric.compute(scoring_input, environment_model)
        collision_time_idcs = result.metadata["collision_time_idcs"]

        time_to_infraction = float(
            collision_time_idcs[0] * self.parameters.proposal_sampling.interval_length
        )
        ego_speed: float = float(environment_model.ego_state.dynamic_car_state.speed)

        is_ok = not (
            time_to_infraction <= self.parameters.time_to_infraction_threshold
            and ego_speed <= self.parameters.max_ego_speed
        )

        if self.parameters.logging_enabled:
            self._log_verification(time, command, time_to_infraction, ego_speed, is_ok)

        if is_ok:
            verification_result = VerificationResult(True)
        else:
            verification_result = VerificationResult(
                False,
                f"Imminent collision: time_to_infraction={time_to_infraction:.2f}s, "
                f"ego_speed={ego_speed:.2f}m/s",
            )

        self._cache[command.name] = verification_result
        return verification_result

    def flush_logs(self, log_dir: str, scenario_name: str) -> None:
        if not self._log_buffer:
            return

        path = os.path.join(log_dir, f"{scenario_name}_verification.jsonl")
        with open(path, "w") as f:
            for entry in self._log_buffer:
                _ = f.write(json.dumps(entry) + "\n")

    def _log_verification(
        self,
        time: Time,
        command: Command,
        time_to_infraction: float,
        ego_speed: float,
        is_ok: bool,
    ) -> None:
        assert isinstance(time, timedelta)
        entry: dict[str, object] = {
            "time": time.total_seconds(),
            "command": command.name,
            "time_to_infraction": float(time_to_infraction),
            "ego_speed": float(ego_speed),
            "result": "pass" if is_ok else "fail",
        }
        self._log_buffer.append(entry)

    def __getstate__(self) -> dict[str, object]:
        """
        Custom getstate to fix pickling since this is a class that inherits from a C++ object
        """
        state = self.__dict__.copy()
        # VerificationResult inherits from C++ Result and can't be pickled
        state["_cache"] = {}
        state["_cache_time"] = float("nan")
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Custom setstate to fix pickling since this is a class that inherits from a C++ object
        """
        super().__init__()
        self.__dict__.update(state)
