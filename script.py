# (Optional) Increase notebook width for all embedded cells to display properly
from IPython.core.display import HTML, display

display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
display(HTML("<style>.container { width:100% !important; }</style>"))
import inspect

from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

print(">>> TYPE:", type(build_torch_module_wrapper))
print(">>> PATH:", inspect.getfile(build_torch_module_wrapper))
# Complete improved trajectory evaluator with normalized metrics

# Enhanced trajectory evaluation module
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, cast

# Safe matplotlib setup for container environment
import matplotlib
import numpy as np
from arbitration_graphs import Behavior, PriorityArbitrator
from behavior_arbitration_nuplan.common.command import Command
from behavior_arbitration_nuplan.common.environment_model import EnvironmentModel
from behavior_arbitration_nuplan.common.utils.time_conversion import to_timedelta
from hydra.utils import instantiate
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
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
from omegaconf import OmegaConf
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_open_planner import (
    PDMOpenPlanner,
)

matplotlib.use("Agg")  # Use non-interactive backend to prevent crashes
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Disable interactive mode to prevent display issues
plt.ioff()

print("Matplotlib configured for container environment (non-interactive mode)")
print("Trajectory evaluation will use safe logging and optional image generation")

# ========================= Trajectory Evaluation Module =========================


@dataclass
class VehicleState:
    """Ego vehicle state"""

    x: float
    y: float
    heading: float  # radians
    length: float = 4.5
    width: float = 2.0
    velocity: Optional[float] = None


@dataclass
class SurroundingObject:
    """Surrounding object"""

    x: float
    y: float
    heading: float
    length: float
    width: float
    object_type: str = "vehicle"


@dataclass
class TrajectoryScore:
    """Detailed trajectory scoring"""

    safety_score: float
    comfort_score: float
    efficiency_score: float
    total_score: float
    min_distance: float
    collision_risk: bool
    collision_reason: str
    trajectory_length: float
    curvature: float
    forward_progress: float


class ImprovedTrajectoryEvaluator:
    """Improved trajectory evaluator with normalized metrics and proper weighting"""

    def __init__(self):
        self.last_selected_index = 0
        self.score_history = []
        self.evaluation_count = 0

        # Statistical parameters for adaptive normalization
        self.metric_stats = {
            "min_distance": {"values": [], "min": float("inf"), "max": 0},
            "avg_curvature": {"values": [], "min": float("inf"), "max": 0},
            "forward_progress": {"values": [], "min": float("inf"), "max": 0},
            "lateral_deviation": {"values": [], "min": float("inf"), "max": 0},
            "path_efficiency": {"values": [], "min": float("inf"), "max": 0},
        }

        # Weight configuration - safety is more important
        self.weights = {
            "safety": 0.5,  # 50% - Safety is most important
            "comfort": 0.25,  # 25% - Comfort
            "efficiency": 0.0,  # 25% - Efficiency
        }

        print(
            f"Improved Evaluator initialized with weights: Safety={self.weights['safety']:.1%}, "
            f"Comfort={self.weights['comfort']:.1%}, Efficiency={self.weights['efficiency']:.1%}"
        )

    def _update_metric_stats(self, metric_name: str, value: float):
        """Update metric statistics for adaptive normalization"""
        if metric_name not in self.metric_stats:
            return

        # 确保值是有效的数字
        if (
            not isinstance(value, (int, float))
            or math.isnan(value)
            or math.isinf(value)
        ):
            return

        # 确保值为非负数（除非是某些允许负值的特殊指标）
        if metric_name in ["forward_progress"]:
            # 这些指标允许负值
            pass
        else:
            # 其他指标强制非负
            value = max(0.0, value)

        stats = self.metric_stats[metric_name]
        if len(stats["values"]) >= 200:  # Keep last 200 values
            stats["values"] = stats["values"][-100:]

        stats["values"].append(float(value))
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)

    def _normalize_metric(
        self,
        value: float,
        metric_name: str,
        ideal_min: float = 0.0,
        ideal_max: float = 100.0,
        reverse: bool = False,
    ) -> float:
        """
        Normalize single metric to [0, 100] range

        Args:
            value: Raw metric value
            metric_name: Metric name
            ideal_min: Ideal minimum value
            ideal_max: Ideal maximum value
            reverse: Whether to reverse (for "smaller is better" metrics)
        """
        # 确保输入值为实数且非负
        if (
            not isinstance(value, (int, float))
            or math.isnan(value)
            or math.isinf(value)
        ):
            return 50.0  # 默认中等分数

        value = float(value)  # 确保是浮点数

        if metric_name in self.metric_stats:
            stats = self.metric_stats[metric_name]
            if len(stats["values"]) > 10:
                # Use percentiles for more robust normalization
                values = np.array(stats["values"])
                # 过滤掉异常值
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    p5, p95 = np.percentile(values, [5, 95])

                    # Dynamic range adjustment
                    range_min = max(ideal_min, p5 * 0.8)
                    range_max = min(ideal_max, p95 * 1.2)
                else:
                    range_min = ideal_min
                    range_max = ideal_max
            else:
                range_min = ideal_min
                range_max = ideal_max
        else:
            range_min = ideal_min
            range_max = ideal_max

        # 确保范围值为非负数
        range_min = max(0.0, range_min)
        range_max = max(range_min + 0.1, range_max)  # 确保max > min

        # Clamp value to reasonable range
        clamped_value = np.clip(value, range_min, range_max)

        # Normalize to [0, 1]
        if range_max - range_min > 1e-6:
            normalized = (clamped_value - range_min) / (range_max - range_min)
        else:
            normalized = 0.5

        # 确保归一化值在有效范围内
        normalized = max(0.0, min(1.0, normalized))

        # Reverse for "smaller is better" metrics
        if reverse:
            normalized = 1.0 - normalized

        # Scale to [0, 100] and apply smooth function to avoid extremes
        score = normalized * 100.0

        # Use sigmoid function to smooth extremes (安全版本)
        try:
            if score > 90:
                exp_arg = -(score - 90) / 5
                exp_arg = max(-10, min(10, exp_arg))  # 限制指数范围
                score = 90 + 10 * (1 / (1 + math.exp(exp_arg)))
            elif score < 10:
                exp_arg = -(score - 5) / 2
                exp_arg = max(-10, min(10, exp_arg))  # 限制指数范围
                score = 10 * (1 / (1 + math.exp(exp_arg)))
        except (OverflowError, ValueError):
            # 如果sigmoid计算失败，直接使用线性值
            pass

        # 最终确保分数在有效范围内
        return max(0.0, min(100.0, score))

    def _calculate_safety_score_normalized(
        self,
        ego_state: VehicleState,
        surrounding_objects: List[SurroundingObject],
        trajectory: List[Tuple[float, float]],
    ) -> Tuple[float, Dict]:
        """Calculate normalized safety score and return detailed components"""
        if not surrounding_objects or not trajectory:
            detailed = {
                "collision_risk_score": 95.0,
                "min_distance_score": 95.0,
                "avg_distance_score": 95.0,
                "encounter_score": 95.0,
                "variance_score": 95.0,
            }
            return 95.0, detailed

        # 1. Calculate minimum distance
        min_distance = float("inf")
        avg_distance = 0.0
        critical_encounters = 0

        distances = []
        for x, y in trajectory[:10]:  # Check first 10 points
            for obj in surrounding_objects:
                dist = math.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2)
                distances.append(dist)
                min_distance = min(min_distance, dist)

                # Calculate critical encounters
                critical_threshold = 8.0 + (ego_state.velocity or 0.0) * 0.3
                if dist < critical_threshold:
                    critical_encounters += 1

        if distances:
            avg_distance = np.mean(distances)
            distance_variance = np.var(distances)
        else:
            detailed = {
                "collision_risk_score": 95.0,
                "min_distance_score": 95.0,
                "avg_distance_score": 95.0,
                "encounter_score": 95.0,
                "variance_score": 95.0,
            }
            return 95.0, detailed

        # 确保最小距离为非负数
        min_distance = max(0.0, min_distance)
        avg_distance = max(0.0, avg_distance)
        distance_variance = max(0.0, distance_variance)

        # Update statistics
        self._update_metric_stats("min_distance", min_distance)

        # 2. 修复碰撞风险计算 - 避免复数和负值
        collision_risk_score = 100.0
        if min_distance < 25.0:
            # 确保指数计算的安全性
            # 使用更安全的指数衰减函数
            safe_distance_factor = max(0.1, min_distance - 2.0)  # 确保非负
            decay_rate = 8.0  # 衰减率

            # 限制指数范围避免数值溢出
            exponent = -safe_distance_factor / decay_rate
            exponent = max(-10.0, min(10.0, exponent))  # 限制在合理范围内

            try:
                risk_factor = math.exp(exponent)
                # 确保风险因子在合理范围内
                risk_factor = max(0.0, min(1.0, risk_factor))
                collision_risk_score = 100.0 * (1.0 - risk_factor)
            except (OverflowError, ValueError):
                # 如果计算出现问题，使用线性衰减作为备用
                if min_distance <= 2.0:
                    collision_risk_score = 10.0  # 很近时给低分
                else:
                    # 线性衰减
                    collision_risk_score = max(
                        10.0, 100.0 * (min_distance - 2.0) / 23.0
                    )

        # 确保碰撞风险分数为非负数
        collision_risk_score = max(0.0, min(100.0, collision_risk_score))

        # 3. Distance distribution score
        distance_score = self._normalize_metric(
            min_distance, "min_distance", ideal_min=2.0, ideal_max=30.0, reverse=False
        )

        # 4. Average distance score
        avg_distance_score = self._normalize_metric(
            avg_distance, "avg_distance", ideal_min=5.0, ideal_max=25.0, reverse=False
        )

        # 5. Critical encounter penalty
        encounter_penalty = min(30.0, max(0.0, critical_encounters * 8.0))
        encounter_score = max(0.0, 100.0 - encounter_penalty)

        # 6. Distance variance score (distance changes should be smooth)
        variance_score = self._normalize_metric(
            distance_variance,
            "distance_variance",
            ideal_min=0.0,
            ideal_max=50.0,
            reverse=True,
        )

        # 确保所有详细分数为非负数
        detailed = {
            "collision_risk_score": max(0.0, collision_risk_score),
            "min_distance_score": max(0.0, distance_score),
            "avg_distance_score": max(0.0, avg_distance_score),
            "encounter_score": max(0.0, encounter_score),
            "variance_score": max(0.0, variance_score),
        }

        # Weighted combination of safety sub-metrics
        safety_components = {
            "collision_risk": detailed["collision_risk_score"]
            * 0.4,  # 40% - Collision risk most important
            "min_distance": detailed["min_distance_score"]
            * 0.3,  # 30% - Minimum distance
            "avg_distance": detailed["avg_distance_score"]
            * 0.15,  # 15% - Average distance
            "encounters": detailed["encounter_score"]
            * 0.1,  # 10% - Critical encounters
            "variance": detailed["variance_score"] * 0.05,  # 5% - Distance variance
        }

        total_safety = sum(safety_components.values())

        # 确保总安全分数为非负数
        total_safety = max(0.0, min(100.0, total_safety))

        return total_safety, detailed

    def _calculate_comfort_score_normalized(
        self, trajectory: List[Tuple[float, float]]
    ) -> Tuple[float, Dict]:
        """Calculate normalized comfort score and return detailed components"""
        if len(trajectory) < 3:
            detailed = {
                "avg_curvature_score": 50.0,
                "max_curvature_score": 50.0,
                "curvature_var_score": 50.0,
                "length_var_score": 50.0,
                "length_score": 50.0,
            }
            return 50.0, detailed

        # 1. Calculate curvature
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i - 1], trajectory[i], trajectory[i + 1]

            # Vector method for curvature calculation
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0.01 and v2_norm > 0.01:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_change = math.acos(cos_angle)
                curvature = angle_change / max((v1_norm + v2_norm) / 2, 0.1)
                curvatures.append(curvature)

        if not curvatures:
            detailed = {
                "avg_curvature_score": 80.0,
                "max_curvature_score": 80.0,
                "curvature_var_score": 80.0,
                "length_var_score": 80.0,
                "length_score": 80.0,
            }
            return 80.0, detailed

        avg_curvature = np.mean(curvatures)
        max_curvature = max(curvatures)
        curvature_variance = np.var(curvatures)

        # Update statistics
        self._update_metric_stats("avg_curvature", avg_curvature)

        # 2. Trajectory length and smoothness
        total_length = 0.0
        segment_lengths = []
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            total_length += length
            segment_lengths.append(length)

        length_variance = np.var(segment_lengths) if len(segment_lengths) > 1 else 0.0

        # 3. Normalize various comfort metrics

        # Average curvature score (smaller is better)
        curvature_score = self._normalize_metric(
            avg_curvature, "avg_curvature", ideal_min=0.0, ideal_max=0.3, reverse=True
        )

        # Maximum curvature score (smaller is better)
        max_curvature_score = self._normalize_metric(
            max_curvature, "max_curvature", ideal_min=0.0, ideal_max=0.5, reverse=True
        )

        # Curvature variance score (smaller is better, indicates smooth curvature changes)
        curvature_var_score = self._normalize_metric(
            curvature_variance,
            "curvature_variance",
            ideal_min=0.0,
            ideal_max=0.01,
            reverse=True,
        )

        # Length variance score (smaller is better, indicates smooth speed changes)
        length_var_score = self._normalize_metric(
            length_variance,
            "length_variance",
            ideal_min=0.0,
            ideal_max=1.0,
            reverse=True,
        )

        # Trajectory total length score (moderate is best)
        ideal_length = 20.0  # Ideal trajectory length
        length_deviation = abs(total_length - ideal_length)
        length_score = self._normalize_metric(
            length_deviation,
            "length_deviation",
            ideal_min=0.0,
            ideal_max=15.0,
            reverse=True,
        )

        # 保存详细分数
        detailed = {
            "avg_curvature_score": curvature_score,
            "max_curvature_score": max_curvature_score,
            "curvature_var_score": curvature_var_score,
            "length_var_score": length_var_score,
            "length_score": length_score,
        }

        # Weighted combination of comfort sub-metrics
        comfort_components = {
            "avg_curvature": curvature_score * 0.35,  # 35% - Average curvature
            "max_curvature": max_curvature_score * 0.25,  # 25% - Maximum curvature
            "curvature_variance": curvature_var_score
            * 0.2,  # 20% - Curvature smoothness
            "length_variance": length_var_score * 0.15,  # 15% - Speed smoothness
            "length_score": length_score
            * 0.05,  # 5% - Trajectory length reasonableness
        }

        total_comfort = sum(comfort_components.values())

        return np.clip(total_comfort, 0.0, 100.0), detailed

    def _calculate_efficiency_score_normalized(
        self, ego_state: VehicleState, trajectory: List[Tuple[float, float]]
    ) -> Tuple[float, Dict]:
        """Calculate normalized efficiency score and return detailed components"""
        if len(trajectory) < 2:
            detailed = {
                "progress_score": 30.0,
                "lateral_score": 30.0,
                "efficiency_score": 30.0,
                "alignment_score": 30.0,
                "consistency_score": 30.0,
            }
            return 30.0, detailed

        # 1. Forward progress
        end_point = trajectory[-1]
        dx = end_point[0] - ego_state.x
        dy = end_point[1] - ego_state.y

        # Forward distance in ego coordinate frame
        forward_progress = dx * math.cos(ego_state.heading) + dy * math.sin(
            ego_state.heading
        )
        lateral_deviation = abs(
            -dx * math.sin(ego_state.heading) + dy * math.cos(ego_state.heading)
        )

        # Update statistics
        self._update_metric_stats("forward_progress", forward_progress)
        self._update_metric_stats("lateral_deviation", lateral_deviation)

        # 2. Path efficiency
        straight_distance = math.sqrt(dx**2 + dy**2)
        actual_length = 0.0
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            actual_length += math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        path_efficiency = straight_distance / max(actual_length, 0.1)
        self._update_metric_stats("path_efficiency", path_efficiency)

        # 3. Goal alignment
        if straight_distance > 0.1:
            goal_direction = math.atan2(dy, dx)
            heading_diff = abs(goal_direction - ego_state.heading)
            heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
            goal_alignment = 1.0 - (heading_diff / math.pi)
        else:
            goal_alignment = 1.0

        # 4. Progress consistency
        segment_progresses = []
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            seg_dx = p2[0] - p1[0]
            seg_dy = p2[1] - p1[1]
            seg_progress = seg_dx * math.cos(ego_state.heading) + seg_dy * math.sin(
                ego_state.heading
            )
            segment_progresses.append(seg_progress)

        progress_variance = (
            np.var(segment_progresses) if len(segment_progresses) > 1 else 0.0
        )

        # 5. Normalize various efficiency metrics

        # Forward progress score
        progress_score = self._normalize_metric(
            forward_progress,
            "forward_progress",
            ideal_min=0.0,
            ideal_max=25.0,
            reverse=False,
        )

        # Lateral deviation score (smaller is better)
        lateral_score = self._normalize_metric(
            lateral_deviation,
            "lateral_deviation",
            ideal_min=0.0,
            ideal_max=5.0,
            reverse=True,
        )

        # Path efficiency score
        efficiency_score = self._normalize_metric(
            path_efficiency,
            "path_efficiency",
            ideal_min=0.7,
            ideal_max=1.0,
            reverse=False,
        )

        # Goal alignment score
        alignment_score = goal_alignment * 100.0

        # Progress consistency score (smaller is better)
        consistency_score = self._normalize_metric(
            progress_variance,
            "progress_variance",
            ideal_min=0.0,
            ideal_max=2.0,
            reverse=True,
        )

        # 保存详细分数
        detailed = {
            "progress_score": progress_score,
            "lateral_score": lateral_score,
            "efficiency_score": efficiency_score,
            "alignment_score": alignment_score,
            "consistency_score": consistency_score,
        }

        # Weighted combination of efficiency sub-metrics
        efficiency_components = {
            "forward_progress": progress_score
            * 0.4,  # 40% - Forward progress most important
            "lateral_deviation": lateral_score * 0.25,  # 25% - Lateral deviation
            "path_efficiency": efficiency_score * 0.2,  # 20% - Path efficiency
            "goal_alignment": alignment_score * 0.1,  # 10% - Goal alignment
            "progress_consistency": consistency_score
            * 0.05,  # 5% - Progress consistency
        }

        total_efficiency = sum(efficiency_components.values())

        return np.clip(total_efficiency, 0.0, 100.0), detailed

    def evaluate_trajectory_detailed(
        self,
        ego_state: VehicleState,
        surrounding_objects: List[SurroundingObject],
        trajectory: List[Tuple[float, float]],
    ) -> Tuple[TrajectoryScore, Dict]:
        """Evaluate trajectory using improved normalized assessment and return detailed scores"""
        self.evaluation_count += 1

        if not trajectory:
            empty_detailed = {
                "collision_risk_score": 0.0,
                "min_distance_score": 0.0,
                "avg_distance_score": 0.0,
                "encounter_score": 0.0,
                "variance_score": 0.0,
                "avg_curvature_score": 0.0,
                "max_curvature_score": 0.0,
                "curvature_var_score": 0.0,
                "length_var_score": 0.0,
                "length_score": 0.0,
                "progress_score": 0.0,
                "lateral_score": 0.0,
                "efficiency_score": 0.0,
                "alignment_score": 0.0,
                "consistency_score": 0.0,
            }
            return TrajectoryScore(
                safety_score=0.0,
                comfort_score=0.0,
                efficiency_score=0.0,
                total_score=0.0,
                min_distance=0.0,
                collision_risk=True,
                collision_reason="Empty trajectory",
                trajectory_length=0.0,
                curvature=0.0,
                forward_progress=0.0,
            ), empty_detailed

        # Calculate normalized main metrics with detailed components
        safety_score, safety_detailed = self._calculate_safety_score_normalized(
            ego_state, surrounding_objects, trajectory
        )
        comfort_score, comfort_detailed = self._calculate_comfort_score_normalized(
            trajectory
        )
        efficiency_score, efficiency_detailed = (
            self._calculate_efficiency_score_normalized(ego_state, trajectory)
        )

        # Combine all detailed scores
        all_detailed = {**safety_detailed, **comfort_detailed, **efficiency_detailed}

        # 调试信息：检查合并的详细分数
        print(f"DEBUG EVAL: all_detailed keys: {list(all_detailed.keys())}")
        print(f"DEBUG EVAL: sample values: {dict(list(all_detailed.items())[:5])}")

        # Collision detection
        collision_risk, collision_reason = self._check_collision_risk_simple(
            ego_state, surrounding_objects, trajectory
        )

        # If collision risk exists, severely penalize safety score
        if collision_risk:
            safety_score = min(safety_score, 38)

        # Calculate weighted total score
        total_score = (
            safety_score * self.weights["safety"]
            + comfort_score * self.weights["comfort"]
            + efficiency_score * self.weights["efficiency"]
        )

        # Calculate auxiliary metrics for display
        min_distance = self._calculate_min_distance(trajectory, surrounding_objects)
        trajectory_length = self._calculate_trajectory_length(trajectory)
        avg_curvature = self._calculate_avg_curvature(trajectory)
        forward_progress = self._calculate_forward_progress(ego_state, trajectory)

        score = TrajectoryScore(
            safety_score=safety_score,
            comfort_score=comfort_score,
            efficiency_score=efficiency_score,
            total_score=total_score,
            min_distance=min_distance,
            collision_risk=collision_risk,
            collision_reason=collision_reason,
            trajectory_length=trajectory_length,
            curvature=avg_curvature,
            forward_progress=forward_progress,
        )

        # Record score history
        self.score_history.append(
            {
                "safety": safety_score,
                "comfort": comfort_score,
                "efficiency": efficiency_score,
                "total": total_score,
                "frame": self.evaluation_count,
            }
        )

        # Keep history at reasonable size
        if len(self.score_history) > 150:
            self.score_history = self.score_history[-100:]

        return score, all_detailed

    def _check_collision_risk_simple(
        self,
        ego_state: VehicleState,
        surrounding_objects: List[SurroundingObject],
        trajectory: List[Tuple[float, float]],
    ) -> Tuple[bool, str]:
        """改进的碰撞风险检测：区分纵向和侧向距离，使用TTC判断纵向风险"""
        if not surrounding_objects:
            return False, ""

        for i, (x, y) in enumerate(trajectory[:8]):
            for obj in surrounding_objects:
                # 计算到物体的总距离
                total_distance = math.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2)

                # 如果距离很远，直接跳过
                if total_distance > 50.0:
                    continue

                # 计算相对于ego车当前朝向的纵向和侧向距离
                dx = x - obj.x
                dy = y - obj.y

                # 使用ego车当前朝向计算纵向和侧向分量
                longitudinal_dist = dx * math.cos(ego_state.heading) + dy * math.sin(
                    ego_state.heading
                )
                lateral_dist = abs(
                    -dx * math.sin(ego_state.heading) + dy * math.cos(ego_state.heading)
                )

                # === 侧向安全检查（极度宽松）===
                if "pedestrian" in obj.object_type.lower():
                    # 行人：侧向距离只要大于1米就认为安全
                    required_lateral = 1.0
                else:
                    # 车辆：考虑车宽，但给予很大余量
                    vehicle_width = max(ego_state.width, obj.width)
                    required_lateral = vehicle_width * 0.6  # 只要0.6倍车宽的侧向间距

                if lateral_dist < required_lateral:
                    return (
                        True,
                        f"Point {i + 1}: {obj.object_type} lateral too close: {lateral_dist:.1f}m (need {required_lateral:.1f}m)",
                    )

                # === 纵向安全检查（使用TTC和相对速度）===
                # 只有当轨迹点在物体前方时才检查纵向碰撞风险
                if longitudinal_dist > 0:  # ego在物体前方
                    # 估算相对速度（简化版本）
                    ego_velocity = ego_state.velocity or 0.0

                    # 假设物体静止或慢速（保守估计）
                    # 在实际应用中，可以从tracking数据获取物体速度
                    assumed_obj_velocity = 0.0  # 保守假设物体静止

                    # 计算纵向相对速度（ego - obj）
                    relative_velocity = ego_velocity - assumed_obj_velocity

                    if relative_velocity > 0.1:  # ego比物体快，有追尾风险
                        # 计算TTC (Time to Collision)
                        ttc = longitudinal_dist / relative_velocity

                        # 根据速度和对象类型设定TTC阈值
                        if "pedestrian" in obj.object_type.lower():
                            # 行人：给予更多反应时间
                            min_ttc = 3.0  # 3秒
                            min_distance = 5.0  # 最小5米距离
                        else:
                            # 车辆：标准反应时间
                            min_ttc = 2.5  # 2.5秒
                            vehicle_length = max(ego_state.length, obj.length)
                            min_distance = vehicle_length + 2.0  # 车长 + 2米缓冲

                        # 双重检查：TTC和最小距离
                        if ttc < min_ttc and longitudinal_dist < min_distance:
                            return (
                                True,
                                f"Point {i + 1}: {obj.object_type} TTC risk: {ttc:.1f}s (need {min_ttc:.1f}s), dist: {longitudinal_dist:.1f}m",
                            )

                    else:  # 相对速度很小或负数，使用静态距离判断
                        if "pedestrian" in obj.object_type.lower():
                            min_static_dist = 3.0
                        else:
                            vehicle_length = max(ego_state.length, obj.length)
                            min_static_dist = vehicle_length * 0.8

                        if longitudinal_dist < min_static_dist:
                            return (
                                True,
                                f"Point {i + 1}: {obj.object_type} static longitudinal too close: {longitudinal_dist:.1f}m (need {min_static_dist:.1f}m)",
                            )

                # === 极近距离紧急检查 ===
                # 无论方向，如果总距离极小，都认为有风险
                emergency_distance = (
                    1.5 if "pedestrian" in obj.object_type.lower() else 2.5
                )
                if total_distance < emergency_distance:
                    return (
                        True,
                        f"Point {i + 1}: {obj.object_type} emergency close: {total_distance:.1f}m (critical: {emergency_distance:.1f}m)",
                    )

        return False, ""

    def _calculate_min_distance(
        self,
        trajectory: List[Tuple[float, float]],
        surrounding_objects: List[SurroundingObject],
    ) -> float:
        """Calculate minimum distance"""
        if not surrounding_objects or not trajectory:
            return float("inf")

        min_dist = float("inf")
        for x, y in trajectory:
            for obj in surrounding_objects:
                dist = math.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2)
                min_dist = min(min_dist, dist)

        return min_dist

    def _calculate_trajectory_length(
        self, trajectory: List[Tuple[float, float]]
    ) -> float:
        """Calculate trajectory length"""
        if len(trajectory) < 2:
            return 0.0

        length = 0.0
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            length += math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        return length

    def _calculate_avg_curvature(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calculate average curvature"""
        if len(trajectory) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i - 1], trajectory[i], trajectory[i + 1]

            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0.01 and v2_norm > 0.01:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_change = math.acos(cos_angle)
                curvature = angle_change / max((v1_norm + v2_norm) / 2, 0.1)
                curvatures.append(curvature)

        return np.mean(curvatures) if curvatures else 0.0

    def _calculate_forward_progress(
        self, ego_state: VehicleState, trajectory: List[Tuple[float, float]]
    ) -> float:
        """Calculate forward progress"""
        if len(trajectory) < 2:
            return 0.0

        end_point = trajectory[-1]
        dx = end_point[0] - ego_state.x
        dy = end_point[1] - ego_state.y

        return dx * math.cos(ego_state.heading) + dy * math.sin(ego_state.heading)

    def select_better_trajectory(
        self,
        ego_state: VehicleState,
        surrounding_objects: List[SurroundingObject],
        trajectory1: List[Tuple[float, float]],
        trajectory2: List[Tuple[float, float]],
    ) -> Tuple[int, str, TrajectoryScore, TrajectoryScore, Dict, Dict]:
        """Select better trajectory and return detailed scores"""
        score1, detailed1 = self.evaluate_trajectory_detailed(
            ego_state, surrounding_objects, trajectory1
        )
        score2, detailed2 = self.evaluate_trajectory_detailed(
            ego_state, surrounding_objects, trajectory2
        )

        # Priority decision making (保持原有逻辑)
        selected_index = 0
        decision_factors = []

        # 1. Collision risk priority
        if score1.collision_risk and not score2.collision_risk:
            selected_index = 1
            decision_factors.append("Avoid collision risk")
        elif score2.collision_risk and not score1.collision_risk:
            selected_index = 0
            decision_factors.append("Avoid collision risk")
        else:
            # 2. Significant safety difference
            safety_diff = abs(score2.safety_score - score1.safety_score)
            if safety_diff > 15.0:
                selected_index = 1 if score2.safety_score > score1.safety_score else 0
                decision_factors.append(
                    f"Significant safety difference: {safety_diff:.1f}"
                )
            else:
                # 3. Total score comparison (with hysteresis)
                total_diff = score2.total_score - score1.total_score
                hysteresis = 2.0 if self.last_selected_index == 0 else -2.0

                if total_diff > hysteresis:
                    selected_index = 1
                    decision_factors.append(f"Total score advantage: {total_diff:.1f}")
                else:
                    selected_index = 0
                    decision_factors.append(
                        f"Maintain selection (hysteresis: {hysteresis:.1f})"
                    )

        # Generate decision explanation
        winner = "Open Planner" if selected_index == 0 else "Closed Planner"
        winner_score = score1 if selected_index == 0 else score2
        loser_score = score2 if selected_index == 0 else score1

        reason_parts = [f"Selected {winner}"]
        reason_parts.extend(decision_factors)
        reason_parts.append(
            f"Scores: {winner_score.total_score:.1f} vs {loser_score.total_score:.1f}"
        )
        reason_parts.append(
            f"Weights: Safety{self.weights['safety']:.1%} Comfort{self.weights['comfort']:.1%} Efficiency{self.weights['efficiency']:.1%}"
        )

        if winner_score.collision_risk:
            reason_parts.append(f"Warning Winner risk: {winner_score.collision_reason}")
        if loser_score.collision_risk:
            reason_parts.append(f"Error Loser risk: {loser_score.collision_reason}")

        self.last_selected_index = selected_index
        return (
            selected_index,
            " | ".join(reason_parts),
            score1,
            score2,
            detailed1,
            detailed2,
        )

    def get_score_statistics(self) -> Dict:
        """Get score statistics"""
        if not self.score_history:
            return {}

        safety_scores = [h["safety"] for h in self.score_history]
        comfort_scores = [h["comfort"] for h in self.score_history]
        efficiency_scores = [h["efficiency"] for h in self.score_history]
        total_scores = [h["total"] for h in self.score_history]

        return {
            "safety": {
                "mean": np.mean(safety_scores),
                "std": np.std(safety_scores),
                "min": np.min(safety_scores),
                "max": np.max(safety_scores),
            },
            "comfort": {
                "mean": np.mean(comfort_scores),
                "std": np.std(comfort_scores),
                "min": np.min(comfort_scores),
                "max": np.max(comfort_scores),
            },
            "efficiency": {
                "mean": np.mean(efficiency_scores),
                "std": np.std(efficiency_scores),
                "min": np.min(efficiency_scores),
                "max": np.max(efficiency_scores),
            },
            "total": {
                "mean": np.mean(total_scores),
                "std": np.std(total_scores),
                "min": np.min(total_scores),
                "max": np.max(total_scores),
            },
        }


# ========================= Updated Behavior Classes =========================

# class Open(Behavior):
#     def __init__(self, invocation: bool, commitment: bool, environment_model: EnvironmentModel, name: str = "open_planner"):
#         super().__init__(name)
#         self.invocation_condition = invocation
#         self.commitment_condition = commitment
#         self.environment_model = environment_model
#         cfg = OmegaConf.load('/workspace/tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_open_planner.yaml')
#         model = build_torch_module_wrapper(cfg.pdm_open_planner.model)
#         self.planner = PDMOpenPlanner(
#             model=model,
#             checkpoint_path='/workspace/pdm_open_checkpoint.ckpt',
#             map_radius=50.0,
#         )

#     def get_command(self, time):
#         current_input = PlannerInput(
#             iteration=self.environment_model._iteration,
#             history=self.environment_model._history,
#             traffic_light_data=self.environment_model._traffic_light_data
#         )
#         command = self.planner.compute_planner_trajectory(current_input)
#         return command

#     def check_invocation_condition(self, time) -> bool:
#         return self.invocation_condition

#     def check_commitment_condition(self, time) -> bool:
#         return self.commitment_condition

#     def set_invocation_condition(self, condition: bool):
#         self.invocation_condition = condition

# def __getstate__(self):
#     state = self.__dict__.copy()
#     state["name"] = self.name
#     return state

# def __setstate__(self, state):
#     super().__init__(state["name"])
#     self.__dict__.update(state)

# class Close(Behavior):
#     def __init__(self, invocation: bool, commitment: bool, environment_model: EnvironmentModel, name: str = "closed_planner"):
#         super().__init__(name)
#         self.invocation_condition = invocation
#         self.commitment_condition = commitment
#         self.environment_model = environment_model
#         planner_cfg = OmegaConf.load('/workspace/tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_closed_planner.yaml')
#         self.planner = cast(AbstractPlanner, instantiate(planner_cfg.pdm_closed_planner))

#     def get_command(self, time):
#         current_input = PlannerInput(
#             iteration=self.environment_model._iteration,
#             history=self.environment_model._history,
#             traffic_light_data=self.environment_model._traffic_light_data
#         )
#         command = self.planner.compute_planner_trajectory(current_input)
#         return command

#     def check_invocation_condition(self, time) -> bool:
#         return self.invocation_condition

#     def check_commitment_condition(self, time) -> bool:
#         return self.commitment_condition

#     def set_invocation_condition(self, condition: bool):
#         self.invocation_condition = condition

# def __getstate__(self):
#     state = self.__dict__.copy()
#     state["name"] = self.name
#     return state

# def __setstate__(self, state):
#     super().__init__(state["name"])
#     self.__dict__.update(state)


class EgoAgent(AbstractPlanner):
    """EgoAgent using improved evaluator"""

    class Parameters:
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=8.0, interval_length=0.2
        )

    def __init__(
        self,
        parameters: Optional[Parameters] = None,
        enable_video: bool = True,
        detailed_logging: bool = False,
    ) -> None:
        if parameters is None:
            parameters = EgoAgent.Parameters()
        self.parameters: EgoAgent.Parameters = parameters
        self.environment_model: EnvironmentModel = EnvironmentModel(
            EnvironmentModel.Parameters(self.parameters.trajectory_sampling)
        )

        # 🔧 修复：直接实例化规划器，不使用包装器
        print("🔄 Initializing Open Planner...")
        try:
            open_planner_cfg = OmegaConf.load(
                "../tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_open_planner.yaml"  # TODO: Hardcoded path
            )
            self.open = cast(
                AbstractPlanner, instantiate(open_planner_cfg.pdm_open_planner)
            )
            print("✅ Open Planner initialized successfully")
        except Exception as e:
            print(f"❌ Open Planner initialization failed: {e}")
            # 创建一个简单的备用规划器
            self.open = None

        print("🔄 Initializing Closed Planner...")
        try:
            close_planner_cfg = OmegaConf.load(
                "../tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_closed_planner.yaml"  # TODO: Hardcoded path
            )
            self.close = cast(
                AbstractPlanner, instantiate(close_planner_cfg.pdm_closed_planner)
            )
            print("✅ Closed Planner initialized successfully")
        except Exception as e:
            print(f"❌ Closed Planner initialization failed: {e}")
            # 创建一个简单的备用规划器
            self.close = None

        # Use improved evaluator
        self.trajectory_evaluator = ImprovedTrajectoryEvaluator()
        self.detailed_logging = detailed_logging

        print(f"EgoAgent initialized with improved normalized evaluator")
        print(f"Video generation: {'Enabled' if enable_video else 'Disabled'}")
        print(f"Detailed logging: {'Enabled' if detailed_logging else 'Disabled'}")

    def initialize(self, initialization: PlannerInitialization) -> None:
        """🔧 修复：正确初始化规划器，不访问.planner属性"""
        super().initialize(initialization)
        self.environment_model.initialize(initialization)

        print("🔄 Initializing planners...")

        # 初始化Open Planner
        if self.open is not None:
            try:
                self.open.initialize(initialization)
                print("✅ Open planner initialized successfully")
            except Exception as e:
                print(f"❌ Open planner initialization failed: {e}")
                self.open = None

        # 初始化Closed Planner
        if self.close is not None:
            try:
                self.close.initialize(initialization)
                print("✅ Closed planner initialized successfully")
            except Exception as e:
                print(f"❌ Closed planner initialization failed: {e}")
                self.close = None

        # 检查至少有一个规划器可用
        if self.open is None and self.close is None:
            raise RuntimeError("❌ Both planners failed to initialize!")

        print(
            f"✅ EgoAgent initialization complete. Available planners: "
            f"Open={'✓' if self.open else '✗'} Closed={'✓' if self.close else '✗'}"
        )

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
        # current_time = self.environment_model.current_time_point()

        # # Force both planners to compute trajectories
        # self.open.set_invocation_condition(False)
        # self.close.set_invocation_condition(False)

        # Generate trajectories
        trajectory1 = self.open.compute_planner_trajectory(current_input)
        trajectory2 = self.close.compute_planner_trajectory(current_input)

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

        # Enhanced logging
        self._log_detailed_decision(
            score1,
            score2,
            selected_index,
            decision_text,
            ego_state,
            surrounding_objects,
        )

        # # Activate selected planner
        # if selected_index == 0:
        #     self.open.set_invocation_condition(True)
        #     self.close.set_invocation_condition(False)
        # else:
        #     self.open.set_invocation_condition(False)
        #     self.close.set_invocation_condition(True)

        # # Return final trajectory
        # final_trajectory = self.root_arbitrator.get_command(to_timedelta(current_time))

        # return final_trajectory

        if selected_index == 0:
            return trajectory1
        else:
            return trajectory2

    def finalize_evaluation(self):
        """Finalize evaluation and generate outputs"""
        print(f"\nFinalizing Trajectory Evaluation")
        print(f"Total frames evaluated: {self.trajectory_evaluator.evaluation_count}")

        # Save score data
        self.video_generator.save_score_data_csv()

        # Print summary statistics
        if self.trajectory_evaluator.score_history:
            stats = self.trajectory_evaluator.get_score_statistics()
            print(f"Final Score Statistics:")
            for metric, stat in stats.items():
                print(
                    f"  {metric.capitalize()}: mean={stat['mean']:.1f} std={stat['std']:.1f} range=[{stat['min']:.1f}, {stat['max']:.1f}]"
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        try:
            self.finalize_evaluation()
        except:
            pass


print("Improved normalized trajectory evaluation system loaded successfully!")
print("Configured with proper metric normalization and weighted scoring")
print("All metrics will be distributed uniformly between 0.000 and 100.000")
print("Usage: planner = EgoAgent(enable_video=True, detailed_logging=True)")
import hydra

# Location of paths with all simulation configs
CONFIG_PATH = (
    "../nuplan-devkit/nuplan/planning/script/config/simulation"  # TODO: Hardcoded path
)
CONFIG_NAME = "default_simulation"

# Create a temporary directory to store the simulation artifacts
SAVE_DIR = "../experiments"  # TODO: Hardcoded path

# Select simulation parameters
CHALLENGE = "closed_loop_reactive_agents"  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
# OBSERVATION = 'idm_agents_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(
    config_name=CONFIG_NAME,
    overrides=[
        f"group={SAVE_DIR}",
        f"experiment_name=planner_tutorial",
        f"job_name=planner_tutorial",
        "experiment=${experiment_name}/${job_name}",
        "output_dir=${group}/${experiment}",
        f"+simulation={CHALLENGE}",
        # f'observation={OBSERVATION}',
        "scenario_filter=val14_split",
        "scenario_builder=nuplan",
        # 'worker=sequential',
        "hydra.searchpath=[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]",
    ],
)
import nest_asyncio
from nuplan.planning.script.run_simulation import run_simulation as main_simulation

nest_asyncio.apply()

planner = EgoAgent()  # 构造 planner

main_simulation(cfg, planners=planner)
simulation_folder = cfg.output_dir

# Print the simulation folder path
print(f"Simulation results are saved in: {simulation_folder}")

# Location of paths with all nuBoard configs
CONFIG_PATH = (
    "../nuplan-devkit/nuplan/planning/script/config/nuboard"  # TODO: Hardcoded path
)
CONFIG_NAME = "default_nuboard"

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(
    config_name=CONFIG_NAME,
    overrides=[
        "scenario_builder=nuplan_mini",  # set the database (same as simulation) used to fetch data for visualization
        f"simulation_path=../experiments/planner_tutorial/planner_tutorial",  # nuboard file path, if left empty the user can open the file inside nuBoard # TODO: Hardcoded path
    ],
)
from pathlib import Path

import pandas as pd
from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import (
    WeightedAverageMetricAggregator,
)
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame

# Step 1: 设置仿真输出目录
output_dir = Path(
    "../experiments/planner_tutorial/planner_tutorial"
)  # TODO: Hardcoded path

# Step 2: 加载所有 .parquet 指标为 dataframe 封装。NuPlan 在跑 run_simulation() 时，会在 metrics/ 目录下输出若干 .parquet 文件。
# 每个文件代表一个 metric 的结果（比如 ego_expert_l2_error.parquet）；
# 每个文件的结构是一个 dataframe，每行是一个 scenario 的得分记录：
metrics_dir = output_dir / "metrics"

# 用 MetricStatisticsDataFrame 包装这些表格，方便后续聚合。
metric_dataframes = {}
for file in metrics_dir.glob("*.parquet"):
    df = pd.read_parquet(file)
    metric_name = file.stem
    metric_dataframes[metric_name] = MetricStatisticsDataFrame(
        metric_statistic_name=metric_name, metric_statistics_dataframe=df
    )

# Step 3: 构建 aggregator。表示你希望对所有指标赋等权（或手动设定某些指标高权重），聚合每个 scenario 的多个指标，计算评分。
aggregator = WeightedAverageMetricAggregator(
    name="default_aggregator",
    metric_weights={"default": 1.0},
    file_name="aggregator_metric.parquet",
    aggregator_save_path=output_dir / "aggregator_metric",
    multiple_metrics=[],
    challenge_name=None,
)

# Step 4: 运行聚合
aggregator(metric_dataframes)

from nuplan.planning.script.run_nuboard import main as main_nuboard

# Run nuBoard
main_nuboard(cfg)
