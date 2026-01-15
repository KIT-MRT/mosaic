from arbitration_graphs.typing import Time
from arbitration_graphs.verification import Result, Verifier
from typing_extensions import override

from arbitration_pdm.common.command import Command
from arbitration_pdm.common.environment_model import EnvironmentModel


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

    @override
    def analyze(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        command: Command,
    ) -> VerificationResult:
        # TODO: implement actual verification logic
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
