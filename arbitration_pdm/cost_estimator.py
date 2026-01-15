from dataclasses import dataclass
from typing import final

import numpy as np
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

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()

        self.parameters: TrajectoryCostEstimator.Parameters = parameters
        self._simulator: PDMSimulator = PDMSimulator(
            self.parameters.trajectory_sampling
        )
        self._scorer: PDMScorer = PDMScorer(self.parameters.trajectory_sampling)

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
        return -scores[0]

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
