from arbitration_graphs.typing import Time
from arbitration_graphs.verification import Result, Verifier
from typing_extensions import override

from arbitration_pdm.common.command import Command
from arbitration_pdm.common.environment_model import EnvironmentModel
from arbitration_pdm.common.utils.trajectory_conversion import trajectory_to_points
from arbitration_pdm.trajectory_evaluator import ImprovedTrajectoryEvaluator


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
    def __init__(self):
        super().__init__()
        self.trajectory_evaluator: ImprovedTrajectoryEvaluator = (
            ImprovedTrajectoryEvaluator()
        )

    @override
    def analyze(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        command: Command,
    ) -> VerificationResult:
        ego_state = environment_model.custom_vehicle_state
        surrounding_objects = environment_model.custom_objects
        traj_points = trajectory_to_points(command.trajectory)

        score, _details = self.trajectory_evaluator.evaluate_trajectory_detailed(
            ego_state=ego_state,
            surrounding_objects=surrounding_objects,
            trajectory=traj_points,
        )

        print(f"Trajectory verification scores: {score}")

        if score.collision_risk:
            return VerificationResult(
                False, "Trajectory verification failed: Collision risk detected."
            )

        return VerificationResult(True)

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
