from typing import final

from arbitration_graphs import CostEstimator
from arbitration_graphs.typing import Time
from typing_extensions import override

from arbitration_pdm.common.command import Command
from arbitration_pdm.common.environment_model import EnvironmentModel
from arbitration_pdm.trajectory_evaluator import ImprovedTrajectoryEvaluator


@final
class TrajectoryCostEstimator(CostEstimator):
    def __init__(self):
        super().__init__()
        self.trajectory_evaluator: ImprovedTrajectoryEvaluator = (
            ImprovedTrajectoryEvaluator()
        )

    @override
    def estimate_cost(
        self,
        time: Time,
        environment_model: EnvironmentModel,
        command: Command,
        is_active: bool,
    ) -> float:
        # TODO: Make configurable
        SWITCHING_PENALTY = 20.0

        ego_state = environment_model.ego_state
        agents = environment_model.agents

        score, _details = self.trajectory_evaluator.evaluate_trajectory_detailed(
            ego_state=ego_state,
            surrounding_objects=agents,
            trajectory=command.ego_states(),
        )

        # TODO: Try replicating the original decision logic more closely

        # Lower cost = better trajectory
        cost = -score.total_score

        # Penalize switching: if this trajectory is NOT currently active,
        # make it more expensive to choose
        if not is_active:
            cost += SWITCHING_PENALTY

        return cost

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
