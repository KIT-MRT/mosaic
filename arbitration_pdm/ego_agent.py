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

from arbitration_pdm.behavior.emergency_stop_behavior import EmergencyStopBehavior
from arbitration_pdm.behavior.pdm_closed import PDMClosedBehavior
from arbitration_pdm.behavior.pdm_open import PDMOpenBehavior
from arbitration_pdm.common.command import Command
from arbitration_pdm.common.environment_model import EnvironmentModel
from arbitration_pdm.cost_estimator import TrajectoryCostEstimator
from arbitration_pdm.planner.emergency_stop_planner import EmergencyStopPlanner
from arbitration_pdm.verifier import TrajectoryVerifier


@final
class EgoAgent(AbstractPlanner):
    """EgoAgent using improved evaluator"""

    class Parameters:
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=8.0, interval_length=0.2
        )
        emergency_stop_behavior: EmergencyStopBehavior.Parameters = (
            EmergencyStopBehavior.Parameters(
                emergency_brake_planner=EmergencyStopPlanner.Parameters(
                    trajectory_sampling=trajectory_sampling,
                    target_acceleration=-5.0,
                )
            )
        )

    def __init__(
        self,
        parameters: Optional[Parameters] = None,
    ) -> None:
        if parameters is None:
            parameters = EgoAgent.Parameters()
        self.parameters: EgoAgent.Parameters = parameters

        self.environment_model = EnvironmentModel(
            EnvironmentModel.Parameters(self.parameters.trajectory_sampling)
        )
        self.initialize_arbitration_graph()

        print("EgoAgent initialized")

    def initialize_arbitration_graph(self) -> None:
        self.pdm_closed_behavior = PDMClosedBehavior()
        self.pdm_open_behavior = PDMOpenBehavior()
        self.emergency_stop_behavior: EmergencyStopBehavior = EmergencyStopBehavior(
            self.parameters.emergency_stop_behavior
        )

        self.verifier = TrajectoryVerifier()
        self.cost_estimator = TrajectoryCostEstimator()
        self.cost_arbitrator = CostArbitrator(
            name="CostArbitrator", verifier=self.verifier
        )

        self.cost_arbitrator.add_option(
            self.pdm_closed_behavior,
            CostArbitrator.Option.Flags.NO_FLAGS,
            self.cost_estimator,
        )
        self.cost_arbitrator.add_option(
            self.pdm_open_behavior,
            CostArbitrator.Option.Flags.NO_FLAGS,
            self.cost_estimator,
        )

        self.root_arbitrator: PriorityArbitrator = PriorityArbitrator(
            "RootArbitrator", self.verifier
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
        self.pdm_closed_behavior.initialize(self.environment_model)
        self.pdm_open_behavior.initialize(self.environment_model)

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
