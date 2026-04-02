import json
import os
from dataclasses import dataclass
from typing import Optional, cast, final

from arbitration_graphs import CostArbitrator, PriorityArbitrator
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from typing_extensions import override

from mosaic.ablation import Ablation
from mosaic.behavior.emergency_stop_behavior import EmergencyStopBehavior
from mosaic.behavior.flow_drive import FlowDriveBehavior
from mosaic.behavior.pdm_closed import PDMClosedBehavior
from mosaic.core.command import Command
from mosaic.core.environment_model import EnvironmentModel
from mosaic.core.cost_estimator import TrajectoryCostEstimator
from mosaic.core.verifier import TrajectoryVerifier


@final
class Mosaic(AbstractPlanner):
    """Mosaic planner using arbitration graphs"""

    @dataclass
    class Parameters:
        ablation: Ablation = Ablation.NONE

        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4.0, interval_length=0.1
        )
        scoring_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4.0, interval_length=0.1
        )
        map_radius: float = 50.0
        cost_estimator: TrajectoryCostEstimator.Parameters = (
            TrajectoryCostEstimator.Parameters(
                trajectory_sampling=scoring_sampling,
                logging_enabled=True,
            )
        )
        emergency_stop_behavior: EmergencyStopBehavior.Parameters = (
            EmergencyStopBehavior.Parameters(
                trajectory_sampling=trajectory_sampling,
            )
        )
        verifier: TrajectoryVerifier.Parameters = TrajectoryVerifier.Parameters(
            proposal_sampling=scoring_sampling,
        )

    def __init__(
        self,
        parameters: Optional[Parameters] = None,
    ) -> None:
        if parameters is None:
            parameters = Mosaic.Parameters()
        self.parameters: Mosaic.Parameters = parameters

        self.environment_model = EnvironmentModel(
            EnvironmentModel.Parameters(
                self.parameters.trajectory_sampling,
                self.parameters.scoring_sampling,
                self.parameters.map_radius,
            )
        )
        self.initialize_arbitration_graph()

    def initialize_arbitration_graph(self) -> None:
        self.flow_drive_behavior = FlowDriveBehavior()
        self.pdm_closed_behavior = PDMClosedBehavior()
        self.emergency_stop_behavior: EmergencyStopBehavior = EmergencyStopBehavior(
            self.parameters.emergency_stop_behavior
        )

        self._cost_estimator = TrajectoryCostEstimator(self.parameters.cost_estimator)
        if self.parameters.ablation == Ablation.NO_VERIFIER:
            self.root_arbitrator: PriorityArbitrator = PriorityArbitrator("Mosaic")
            self.cost_arbitrator = CostArbitrator("Composer", self._cost_estimator)
        else:
            self._verifier = TrajectoryVerifier(self.parameters.verifier)
            self.root_arbitrator: PriorityArbitrator = PriorityArbitrator(
                "Mosaic", self._verifier
            )
            self.cost_arbitrator = CostArbitrator(
                "Composer", self._cost_estimator, self._verifier
            )

        if self.parameters.ablation != Ablation.PDM_CLOSED_ONLY:
            self.cost_arbitrator.add_option(
                self.flow_drive_behavior,
                CostArbitrator.Option.Flags.INTERRUPTABLE,
            )
        if self.parameters.ablation != Ablation.FLOW_DRIVE_ONLY:
            self.cost_arbitrator.add_option(
                self.pdm_closed_behavior,
                CostArbitrator.Option.Flags.INTERRUPTABLE,
            )

        self.root_arbitrator.add_option(
            self.cost_arbitrator, int(PriorityArbitrator.Option.Flags.NO_FLAGS)
        )
        self.root_arbitrator.add_option(
            self.emergency_stop_behavior, int(PriorityArbitrator.Option.Flags.FALLBACK)
        )

    @override
    def initialize(self, initialization: PlannerInitialization) -> None:
        super().initialize(initialization)

        self.environment_model.initialize(initialization)
        self.flow_drive_behavior.initialize(self.environment_model)
        self.pdm_closed_behavior.initialize(self.environment_model)

    @override
    def name(self) -> str:
        return self.__class__.__name__

    @override
    def observation_type(self) -> type[Observation]:
        return DetectionsTracks

    @override
    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        self.environment_model.update(current_input)

        current_time = self.environment_model.current_time_delta
        command = cast(
            Command,
            self.root_arbitrator.get_command(current_time, self.environment_model),
        )

        return command.trajectory

    def flush_logs(self, output_dir: str, scenario_name: str) -> None:
        """Write buffered cost estimator and verifier logs to the output directory."""
        log_dir = os.path.join(output_dir, "mosaic_logs")
        os.makedirs(log_dir, exist_ok=True)

        if self._cost_estimator._log_buffer:
            path = os.path.join(log_dir, f"{scenario_name}_trajectory_costs.jsonl")
            with open(path, "w") as f:
                for entry in self._cost_estimator._log_buffer:
                    f.write(json.dumps(entry) + "\n")

        if (
            self.parameters.ablation != Ablation.NO_VERIFIER
            and self._verifier._log_buffer
        ):
            path = os.path.join(log_dir, f"{scenario_name}_verification.jsonl")
            with open(path, "w") as f:
                for entry in self._verifier._log_buffer:
                    f.write(json.dumps(entry) + "\n")

    def __getstate__(self) -> dict[str, object]:
        """
        Custom getstate to avoid pickling the arbitration graph (C++ object).
        """
        state = self.__dict__.copy()
        state["root_arbitrator"] = None
        state["cost_arbitrator"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Custom setstate to re-initialize the arbitration graph
        """
        self.__dict__.update(state)
        self.initialize_arbitration_graph()
