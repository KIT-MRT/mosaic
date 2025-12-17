from typing import final

from arbitration_graphs import CostEstimator
from arbitration_graphs.typing import Time
from typing_extensions import override

from arbitration_pdm.common.command import Command
from arbitration_pdm.common.environment_model import EnvironmentModel


@final
class TrajectoryCostEstimator(CostEstimator):
    def __init__(self):
        super().__init__()

    @override
    def estimate_cost(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        command: Command,
        is_active: bool,
    ) -> float:
        print(
            f"Estimating cost for {command.name} with a trajectory of length {len(command.ego_states())}"
        )
        score = environment_model.scorer.score(command.trajectory)
        if score.shape != (1,):
            raise ValueError(
                f"Expected score shape (1,), got {score.shape} when scoring trajectory."
            )
        print(f"Raw PDM score: {score[0]}")
        return -score[0]

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
