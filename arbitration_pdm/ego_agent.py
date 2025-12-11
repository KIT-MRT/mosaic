from typing import List, Optional, Type

from arbitration_graphs import CostArbitrator
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

from arbitration_pdm.behavior.pdm_closed import PDMClosedBehavior
from arbitration_pdm.behavior.pdm_open import PDMOpenBehavior
from arbitration_pdm.common.environment_model import EnvironmentModel
from arbitration_pdm.common.types import (
    SurroundingObject,
    TrajectoryScore,
    VehicleState,
)
from arbitration_pdm.cost_estimator import TrajectoryCostEstimator
from arbitration_pdm.trajectory_evaluator import ImprovedTrajectoryEvaluator
from arbitration_pdm.verifier import TrajectoryVerifier


class EgoAgent(AbstractPlanner):
    """EgoAgent using improved evaluator"""

    class Parameters:
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=8.0, interval_length=0.2
        )

    def __init__(
        self,
        parameters: Optional[Parameters] = None,
        detailed_logging: bool = False,
    ) -> None:
        if parameters is None:
            parameters = EgoAgent.Parameters()
        self.parameters: EgoAgent.Parameters = parameters

        self.environment_model = EnvironmentModel()
        self.initialize_arbitration_graph()

        self.trajectory_evaluator = ImprovedTrajectoryEvaluator()
        self.detailed_logging = detailed_logging

        print("EgoAgent initialized")
        print(f"Detailed logging: {'Enabled' if detailed_logging else 'Disabled'}")

    def initialize_arbitration_graph(self) -> None:
        self.pdm_closed_behavior = PDMClosedBehavior()
        self.pdm_open_behavior = PDMOpenBehavior()

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

    def initialize(self, initialization: PlannerInitialization) -> None:
        super().initialize(initialization)

        self.environment_model.initialize(initialization)
        self.pdm_closed_behavior.initialize(self.environment_model)
        self.pdm_open_behavior.initialize(self.environment_model)

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def _extract_nuplan_data(self, current_input: PlannerInput):
        """Extract evaluation data from nuPlan input"""
        # Extract ego state
        ego_state_history = current_input.history.ego_states
        if ego_state_history:
            current_ego = ego_state_history[-1]
            ego_state = VehicleState(
                x=current_ego.rear_axle.x,
                y=current_ego.rear_axle.y,
                heading=current_ego.rear_axle.heading,
                velocity=getattr(
                    current_ego.dynamic_car_state.rear_axle_velocity_2d,
                    "magnitude",
                    lambda: 0.0,
                )(),
            )
        else:
            ego_state = VehicleState(x=0.0, y=0.0, heading=0.0)

        # Extract surrounding objects
        surrounding_objects = []
        observations = current_input.history.observations
        if observations:
            latest_observation = observations[-1]
            if hasattr(latest_observation, "tracked_objects"):
                for tracked_obj in latest_observation.tracked_objects.tracked_objects:
                    surrounding_objects.append(
                        SurroundingObject(
                            x=tracked_obj.center.x,
                            y=tracked_obj.center.y,
                            heading=tracked_obj.center.heading,
                            length=tracked_obj.box.length,
                            width=tracked_obj.box.width,
                            object_type=getattr(
                                tracked_obj.tracked_object_type, "name", "vehicle"
                            ).lower(),
                        )
                    )

        return ego_state, surrounding_objects

    def _extract_trajectory_points(self, trajectory):
        """Extract coordinate points from trajectory object"""
        if trajectory is None:
            return []

        try:
            if hasattr(trajectory, "get_sampled_trajectory"):
                sampled_traj = trajectory.get_sampled_trajectory()
                return [
                    (state.rear_axle.x, state.rear_axle.y) for state in sampled_traj
                ]
            elif hasattr(trajectory, "trajectory"):
                return [
                    (state.rear_axle.x, state.rear_axle.y)
                    for state in trajectory.trajectory
                ]
            else:
                return []
        except:
            return []

    def _log_detailed_decision(
        self,
        score1: TrajectoryScore,
        score2: TrajectoryScore,
        selected_index: int,
        decision_text: str,
        ego_state: VehicleState,
        surrounding_objects: List[SurroundingObject],
    ):
        """Enhanced logging for decision details"""
        if not self.detailed_logging:
            return

        winner = "Open" if selected_index == 0 else "Closed"
        winner_score = score1 if selected_index == 0 else score2
        loser_score = score2 if selected_index == 0 else score1

        frame = self.trajectory_evaluator.evaluation_count
        for part in decision_text.split("|"):
            print(f"   • {part.strip()}")

        # Show score statistics every 50 frames
        if frame % 50 == 0:
            stats = self.trajectory_evaluator.get_score_statistics()
            if stats:
                print(
                    f"\nScore Statistics (last {len(self.trajectory_evaluator.score_history)} frames):"
                )
                for metric, stat in stats.items():
                    print(
                        f"   {metric.capitalize()}: mean={stat['mean']:.1f} std={stat['std']:.1f} range=[{stat['min']:.1f}, {stat['max']:.1f}]"
                    )

        print(f"{'=' * 100}\n")

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        self.environment_model.update(current_input)

        current_time = self.environment_model.current_time_delta
        command = self.cost_arbitrator.get_command(current_time, self.environment_model)

        return command.trajectory

        # Generate trajectories
        trajectory1 = self.pdm_open_behavior.get_command(
            self.environment_model.current_time_delta, self.environment_model
        ).trajectory
        trajectory2 = self.pdm_closed_behavior.get_command(
            self.environment_model.current_time_delta, self.environment_model
        ).trajectory

        # Extract evaluation data
        ego_state, surrounding_objects = self._extract_nuplan_data(current_input)
        traj1_points = self._extract_trajectory_points(trajectory1)
        traj2_points = self._extract_trajectory_points(trajectory2)

        # Enhanced trajectory evaluation with detailed scores
        selected_index, decision_text, score1, score2, detailed1, detailed2 = (
            self.trajectory_evaluator.select_better_trajectory(
                ego_state, surrounding_objects, traj1_points, traj2_points
            )
        )

        # 调试信息：检查detailed字典
        print(
            f"DEBUG MAIN: detailed1 type: {type(detailed1)}, keys: {list(detailed1.keys()) if detailed1 else 'None'}"
        )
        print(
            f"DEBUG MAIN: detailed2 type: {type(detailed2)}, keys: {list(detailed2.keys()) if detailed2 else 'None'}"
        )

        self._log_detailed_decision(
            score1,
            score2,
            selected_index,
            decision_text,
            ego_state,
            surrounding_objects,
        )

        if selected_index == 0:
            return trajectory1
        else:
            return trajectory2

    def finalize_evaluation(self):
        """Finalize evaluation and generate outputs"""
        print("\nFinalizing Trajectory Evaluation")
        print(f"Total frames evaluated: {self.trajectory_evaluator.evaluation_count}")

        # Print summary statistics
        if self.trajectory_evaluator.score_history:
            stats = self.trajectory_evaluator.get_score_statistics()
            print("Final Score Statistics:")
            for metric, stat in stats.items():
                print(
                    f"  {metric.capitalize()}: mean={stat['mean']:.1f} std={stat['std']:.1f} range=[{stat['min']:.1f}, {stat['max']:.1f}]"
                )

    def __del__(self):
        try:
            self.finalize_evaluation()
        except:
            pass

    def __getstate__(self) -> dict[str, object]:
        """
        Custom getstate to avoid pickling the arbitration graph (C++ object).
        """
        state = self.__dict__.copy()
        state["cost_arbitrator"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Custom setstate to re-initialize the arbitration graph
        """
        self.__dict__.update(state)
        self.initialize_arbitration_graph()
