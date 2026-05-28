from importlib import resources
from typing import Optional

import numpy as np
from arbitration_graphs import Behavior
from arbitration_graphs.typing import Time
from hydra.utils import instantiate
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from omegaconf import OmegaConf
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from typing_extensions import cast, final, override

import mosaic.utils.trajectory as trajectory_utils
from mosaic.core.command import Command, StampedCommand
from mosaic.core.environment_model import EnvironmentModel


@final
class PDMClosedBehavior(Behavior):
    def __init__(
        self,
        name: str = "pdm_closed",
    ):
        super().__init__(name)

        cfg_path = (
            resources.files("tuplan_garage")
            / "planning"
            / "script"
            / "config"
            / "simulation"
            / "planner"
            / "pdm_closed_planner.yaml"
        )

        with cfg_path.open("r") as f:
            pdm_closed_cfg = OmegaConf.load(f)

        pdm_closed_cfg.pdm_closed_planner.enable_emergency_brake_fallback = False

        self.planner = cast(
            AbstractPlanner, instantiate(pdm_closed_cfg.pdm_closed_planner)
        )
        self.last_command: Optional[StampedCommand] = None
        self._simulator: Optional[PDMSimulator] = None

    def initialize(self, environment_model: EnvironmentModel) -> None:
        self.planner.initialize(environment_model.planner_initialization)
        self._simulator = PDMSimulator(environment_model.parameters.proposal_sampling)

    @override
    def get_command(self, time: Time, environment_model: EnvironmentModel) -> Command:
        assert self._simulator is not None
        trajectory: AbstractTrajectory = self.planner.compute_planner_trajectory(
            environment_model.planner_input
        )

        states = trajectory_utils.trajectory_to_state_array(
            trajectory, environment_model.parameters.proposal_sampling
        )
        simulated = self._simulator.simulate_proposals(
            np.expand_dims(states, axis=0), environment_model.ego_state
        )
        command = Command(
            name=self.name(), trajectory=trajectory, simulated_states=simulated[0]
        )
        self.last_command = StampedCommand(stamp=time, command=command)
        return command

    @override
    def check_invocation_condition(
        self, time: Time, environment_model: EnvironmentModel
    ) -> bool:
        return True

    @override
    def check_commitment_condition(
        self, time: Time, environment_model: EnvironmentModel
    ) -> bool:
        return True

    def __getstate__(self) -> dict[str, object]:
        """
        Custom getstate to fix pickling since this is a class that inherits from a C++ object
        """
        state = self.__dict__.copy()
        state["name"] = self.name()
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Custom setstate to fix pickling since this is a class that inherits from a C++ object
        """
        name = cast(str, state["name"])
        super().__init__(name)
        _ = state.pop("name", None)
        self.__dict__.update(state)
