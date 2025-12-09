# (Optional) Increase notebook width for all embedded cells to display properly
from IPython.core.display import display, HTML
display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
display(HTML("<style>.container { width:100% !important; }</style>"))
import inspect
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

print(">>> TYPE:", type(build_torch_module_wrapper))
print(">>> PATH:", inspect.getfile(build_torch_module_wrapper))
# Complete improved trajectory evaluator with normalized metrics

from typing import Type
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInput, PlannerInitialization
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from arbitration_graphs import Behavior, PriorityArbitrator
from behavior_arbitration_nuplan.common.command import Command
from omegaconf import OmegaConf
from behavior_arbitration_nuplan.common.environment_model import EnvironmentModel
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_open_planner import PDMOpenPlanner
from dataclasses import dataclass
from behavior_arbitration_nuplan.common.utils.time_conversion import to_timedelta
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from typing import Optional, cast
from hydra.utils import instantiate
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

# Enhanced trajectory evaluation module
import math
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path
import subprocess

# Safe matplotlib setup for container environment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent crashes
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
            'min_distance': {'values': [], 'min': float('inf'), 'max': 0},
            'avg_curvature': {'values': [], 'min': float('inf'), 'max': 0},
            'forward_progress': {'values': [], 'min': float('inf'), 'max': 0},
            'lateral_deviation': {'values': [], 'min': float('inf'), 'max': 0},
            'path_efficiency': {'values': [], 'min': float('inf'), 'max': 0}
        }
        
        # Weight configuration - safety is more important
        self.weights = {
            'safety': 0.5,      # 50% - Safety is most important
            'comfort': 0.25,    # 25% - Comfort
            'efficiency': 0.25  # 25% - Efficiency
        }
        
        print(f"Improved Evaluator initialized with weights: Safety={self.weights['safety']:.1%}, "
              f"Comfort={self.weights['comfort']:.1%}, Efficiency={self.weights['efficiency']:.1%}")
    
    def _update_metric_stats(self, metric_name: str, value: float):
        """Update metric statistics for adaptive normalization"""
        if metric_name not in self.metric_stats:
            return
        
        # 确保值是有效的数字
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return
        
        # 确保值为非负数（除非是某些允许负值的特殊指标）
        if metric_name in ['forward_progress']:
            # 这些指标允许负值
            pass
        else:
            # 其他指标强制非负
            value = max(0.0, value)
            
        stats = self.metric_stats[metric_name]
        if len(stats['values']) >= 200:  # Keep last 200 values
            stats['values'] = stats['values'][-100:]
        
        stats['values'].append(float(value))
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
    
    def _normalize_metric(self, value: float, metric_name: str, 
                        ideal_min: float = 0.0, ideal_max: float = 100.0,
                        reverse: bool = False) -> float:
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
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return 50.0  # 默认中等分数
        
        value = float(value)  # 确保是浮点数
        
        if metric_name in self.metric_stats:
            stats = self.metric_stats[metric_name]
            if len(stats['values']) > 10:
                # Use percentiles for more robust normalization
                values = np.array(stats['values'])
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
    
    def _calculate_safety_score_normalized(self, ego_state: VehicleState,
                                        surrounding_objects: List[SurroundingObject],
                                        trajectory: List[Tuple[float, float]]) -> Tuple[float, Dict]:
        """Calculate normalized safety score and return detailed components"""
        if not surrounding_objects or not trajectory:
            detailed = {
                'collision_risk_score': 95.0,
                'min_distance_score': 95.0,
                'avg_distance_score': 95.0,
                'encounter_score': 95.0,
                'variance_score': 95.0
            }
            return 95.0, detailed
        
        # 1. Calculate minimum distance
        min_distance = float('inf')
        avg_distance = 0.0
        critical_encounters = 0
        
        distances = []
        for x, y in trajectory[:10]:  # Check first 10 points
            for obj in surrounding_objects:
                dist = math.sqrt((x - obj.x)**2 + (y - obj.y)**2)
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
                'collision_risk_score': 95.0,
                'min_distance_score': 95.0,
                'avg_distance_score': 95.0,
                'encounter_score': 95.0,
                'variance_score': 95.0
            }
            return 95.0, detailed
        
        # 确保最小距离为非负数
        min_distance = max(0.0, min_distance)
        avg_distance = max(0.0, avg_distance)
        distance_variance = max(0.0, distance_variance)
        
        # Update statistics
        self._update_metric_stats('min_distance', min_distance)
        
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
                    collision_risk_score = max(10.0, 100.0 * (min_distance - 2.0) / 23.0)
        
        # 确保碰撞风险分数为非负数
        collision_risk_score = max(0.0, min(100.0, collision_risk_score))
        
        # 3. Distance distribution score
        distance_score = self._normalize_metric(
            min_distance, 'min_distance', 
            ideal_min=2.0, ideal_max=30.0, reverse=False
        )
        
        # 4. Average distance score
        avg_distance_score = self._normalize_metric(
            avg_distance, 'avg_distance',
            ideal_min=5.0, ideal_max=25.0, reverse=False
        )
        
        # 5. Critical encounter penalty
        encounter_penalty = min(30.0, max(0.0, critical_encounters * 8.0))
        encounter_score = max(0.0, 100.0 - encounter_penalty)
        
        # 6. Distance variance score (distance changes should be smooth)
        variance_score = self._normalize_metric(
            distance_variance, 'distance_variance',
            ideal_min=0.0, ideal_max=50.0, reverse=True
        )
        
        # 确保所有详细分数为非负数
        detailed = {
            'collision_risk_score': max(0.0, collision_risk_score),
            'min_distance_score': max(0.0, distance_score),
            'avg_distance_score': max(0.0, avg_distance_score),
            'encounter_score': max(0.0, encounter_score),
            'variance_score': max(0.0, variance_score)
        }
        
        # Weighted combination of safety sub-metrics
        safety_components = {
            'collision_risk': detailed['collision_risk_score'] * 0.4,    # 40% - Collision risk most important
            'min_distance': detailed['min_distance_score'] * 0.3,       # 30% - Minimum distance
            'avg_distance': detailed['avg_distance_score'] * 0.15,      # 15% - Average distance
            'encounters': detailed['encounter_score'] * 0.1,            # 10% - Critical encounters
            'variance': detailed['variance_score'] * 0.05               # 5% - Distance variance
        }
        
        total_safety = sum(safety_components.values())
        
        # 确保总安全分数为非负数
        total_safety = max(0.0, min(100.0, total_safety))
        
        return total_safety, detailed
    
    def _calculate_comfort_score_normalized(self, trajectory: List[Tuple[float, float]]) -> Tuple[float, Dict]:
        """Calculate normalized comfort score and return detailed components"""
        if len(trajectory) < 3:
            detailed = {
                'avg_curvature_score': 50.0,
                'max_curvature_score': 50.0,
                'curvature_var_score': 50.0,
                'length_var_score': 50.0,
                'length_score': 50.0
            }
            return 50.0, detailed
        
        # 1. Calculate curvature
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
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
                'avg_curvature_score': 80.0,
                'max_curvature_score': 80.0,
                'curvature_var_score': 80.0,
                'length_var_score': 80.0,
                'length_score': 80.0
            }
            return 80.0, detailed
        
        avg_curvature = np.mean(curvatures)
        max_curvature = max(curvatures)
        curvature_variance = np.var(curvatures)
        
        # Update statistics
        self._update_metric_stats('avg_curvature', avg_curvature)
        
        # 2. Trajectory length and smoothness
        total_length = 0.0
        segment_lengths = []
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i+1]
            length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += length
            segment_lengths.append(length)
        
        length_variance = np.var(segment_lengths) if len(segment_lengths) > 1 else 0.0
        
        # 3. Normalize various comfort metrics
        
        # Average curvature score (smaller is better)
        curvature_score = self._normalize_metric(
            avg_curvature, 'avg_curvature',
            ideal_min=0.0, ideal_max=0.3, reverse=True
        )
        
        # Maximum curvature score (smaller is better)
        max_curvature_score = self._normalize_metric(
            max_curvature, 'max_curvature',
            ideal_min=0.0, ideal_max=0.5, reverse=True
        )
        
        # Curvature variance score (smaller is better, indicates smooth curvature changes)
        curvature_var_score = self._normalize_metric(
            curvature_variance, 'curvature_variance',
            ideal_min=0.0, ideal_max=0.01, reverse=True
        )
        
        # Length variance score (smaller is better, indicates smooth speed changes)
        length_var_score = self._normalize_metric(
            length_variance, 'length_variance',
            ideal_min=0.0, ideal_max=1.0, reverse=True
        )
        
        # Trajectory total length score (moderate is best)
        ideal_length = 20.0  # Ideal trajectory length
        length_deviation = abs(total_length - ideal_length)
        length_score = self._normalize_metric(
            length_deviation, 'length_deviation',
            ideal_min=0.0, ideal_max=15.0, reverse=True
        )
        
        # 保存详细分数
        detailed = {
            'avg_curvature_score': curvature_score,
            'max_curvature_score': max_curvature_score,
            'curvature_var_score': curvature_var_score,
            'length_var_score': length_var_score,
            'length_score': length_score
        }
        
        # Weighted combination of comfort sub-metrics
        comfort_components = {
            'avg_curvature': curvature_score * 0.35,        # 35% - Average curvature
            'max_curvature': max_curvature_score * 0.25,    # 25% - Maximum curvature
            'curvature_variance': curvature_var_score * 0.2, # 20% - Curvature smoothness
            'length_variance': length_var_score * 0.15,     # 15% - Speed smoothness
            'length_score': length_score * 0.05             # 5% - Trajectory length reasonableness
        }
        
        total_comfort = sum(comfort_components.values())
        
        return np.clip(total_comfort, 0.0, 100.0), detailed
    
    def _calculate_efficiency_score_normalized(self, ego_state: VehicleState,
                                            trajectory: List[Tuple[float, float]]) -> Tuple[float, Dict]:
        """Calculate normalized efficiency score and return detailed components"""
        if len(trajectory) < 2:
            detailed = {
                'progress_score': 30.0,
                'lateral_score': 30.0,
                'efficiency_score': 30.0,
                'alignment_score': 30.0,
                'consistency_score': 30.0
            }
            return 30.0, detailed
        
        # 1. Forward progress
        end_point = trajectory[-1]
        dx = end_point[0] - ego_state.x
        dy = end_point[1] - ego_state.y
        
        # Forward distance in ego coordinate frame
        forward_progress = dx * math.cos(ego_state.heading) + dy * math.sin(ego_state.heading)
        lateral_deviation = abs(-dx * math.sin(ego_state.heading) + dy * math.cos(ego_state.heading))
        
        # Update statistics
        self._update_metric_stats('forward_progress', forward_progress)
        self._update_metric_stats('lateral_deviation', lateral_deviation)
        
        # 2. Path efficiency
        straight_distance = math.sqrt(dx**2 + dy**2)
        actual_length = 0.0
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i+1]
            actual_length += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        path_efficiency = straight_distance / max(actual_length, 0.1)
        self._update_metric_stats('path_efficiency', path_efficiency)
        
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
            p1, p2 = trajectory[i], trajectory[i+1]
            seg_dx = p2[0] - p1[0]
            seg_dy = p2[1] - p1[1]
            seg_progress = seg_dx * math.cos(ego_state.heading) + seg_dy * math.sin(ego_state.heading)
            segment_progresses.append(seg_progress)
        
        progress_variance = np.var(segment_progresses) if len(segment_progresses) > 1 else 0.0
        
        # 5. Normalize various efficiency metrics
        
        # Forward progress score
        progress_score = self._normalize_metric(
            forward_progress, 'forward_progress',
            ideal_min=0.0, ideal_max=25.0, reverse=False
        )
        
        # Lateral deviation score (smaller is better)
        lateral_score = self._normalize_metric(
            lateral_deviation, 'lateral_deviation',
            ideal_min=0.0, ideal_max=5.0, reverse=True
        )
        
        # Path efficiency score
        efficiency_score = self._normalize_metric(
            path_efficiency, 'path_efficiency',
            ideal_min=0.7, ideal_max=1.0, reverse=False
        )
        
        # Goal alignment score
        alignment_score = goal_alignment * 100.0
        
        # Progress consistency score (smaller is better)
        consistency_score = self._normalize_metric(
            progress_variance, 'progress_variance',
            ideal_min=0.0, ideal_max=2.0, reverse=True
        )
        
        # 保存详细分数
        detailed = {
            'progress_score': progress_score,
            'lateral_score': lateral_score,
            'efficiency_score': efficiency_score,
            'alignment_score': alignment_score,
            'consistency_score': consistency_score
        }
        
        # Weighted combination of efficiency sub-metrics
        efficiency_components = {
            'forward_progress': progress_score * 0.4,        # 40% - Forward progress most important
            'lateral_deviation': lateral_score * 0.25,      # 25% - Lateral deviation
            'path_efficiency': efficiency_score * 0.2,      # 20% - Path efficiency
            'goal_alignment': alignment_score * 0.1,        # 10% - Goal alignment
            'progress_consistency': consistency_score * 0.05 # 5% - Progress consistency
        }
        
        total_efficiency = sum(efficiency_components.values())
        
        return np.clip(total_efficiency, 0.0, 100.0), detailed
    
    def evaluate_trajectory_detailed(self, ego_state: VehicleState, 
                                surrounding_objects: List[SurroundingObject],
                                trajectory: List[Tuple[float, float]]) -> Tuple[TrajectoryScore, Dict]:
        """Evaluate trajectory using improved normalized assessment and return detailed scores"""
        self.evaluation_count += 1
        
        if not trajectory:
            empty_detailed = {
                'collision_risk_score': 0.0, 'min_distance_score': 0.0, 'avg_distance_score': 0.0,
                'encounter_score': 0.0, 'variance_score': 0.0, 'avg_curvature_score': 0.0,
                'max_curvature_score': 0.0, 'curvature_var_score': 0.0, 'length_var_score': 0.0,
                'length_score': 0.0, 'progress_score': 0.0, 'lateral_score': 0.0,
                'efficiency_score': 0.0, 'alignment_score': 0.0, 'consistency_score': 0.0
            }
            return TrajectoryScore(
                safety_score=0.0, comfort_score=0.0, efficiency_score=0.0,
                total_score=0.0, min_distance=0.0, collision_risk=True,
                collision_reason="Empty trajectory", trajectory_length=0.0,
                curvature=0.0, forward_progress=0.0
            ), empty_detailed
        
        # Calculate normalized main metrics with detailed components
        safety_score, safety_detailed = self._calculate_safety_score_normalized(ego_state, surrounding_objects, trajectory)
        comfort_score, comfort_detailed = self._calculate_comfort_score_normalized(trajectory)
        efficiency_score, efficiency_detailed = self._calculate_efficiency_score_normalized(ego_state, trajectory)
        
        # Combine all detailed scores
        all_detailed = {**safety_detailed, **comfort_detailed, **efficiency_detailed}
        
        # 调试信息：检查合并的详细分数
        print(f"DEBUG EVAL: all_detailed keys: {list(all_detailed.keys())}")
        print(f"DEBUG EVAL: sample values: {dict(list(all_detailed.items())[:5])}")
        
        # Collision detection
        collision_risk, collision_reason = self._check_collision_risk_simple(ego_state, surrounding_objects, trajectory)
        
        # If collision risk exists, severely penalize safety score
        if collision_risk:
            safety_score = min(safety_score, 38)
        
        # Calculate weighted total score
        total_score = (safety_score * self.weights['safety'] + 
                    comfort_score * self.weights['comfort'] + 
                    efficiency_score * self.weights['efficiency'])
        
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
            forward_progress=forward_progress
        )
        
        # Record score history
        self.score_history.append({
            'safety': safety_score,
            'comfort': comfort_score,
            'efficiency': efficiency_score,
            'total': total_score,
            'frame': self.evaluation_count
        })
        
        # Keep history at reasonable size
        if len(self.score_history) > 150:
            self.score_history = self.score_history[-100:]
        
        return score, all_detailed
    
    def _check_collision_risk_simple(self, ego_state: VehicleState, 
                                surrounding_objects: List[SurroundingObject],
                                trajectory: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """改进的碰撞风险检测：区分纵向和侧向距离，使用TTC判断纵向风险"""
        if not surrounding_objects:
            return False, ""
        
        for i, (x, y) in enumerate(trajectory[:8]):
            for obj in surrounding_objects:
                # 计算到物体的总距离
                total_distance = math.sqrt((x - obj.x)**2 + (y - obj.y)**2)
                
                # 如果距离很远，直接跳过
                if total_distance > 50.0:
                    continue
                
                # 计算相对于ego车当前朝向的纵向和侧向距离
                dx = x - obj.x
                dy = y - obj.y
                
                # 使用ego车当前朝向计算纵向和侧向分量
                longitudinal_dist = dx * math.cos(ego_state.heading) + dy * math.sin(ego_state.heading)
                lateral_dist = abs(-dx * math.sin(ego_state.heading) + dy * math.cos(ego_state.heading))
                
                # === 侧向安全检查（极度宽松）===
                if 'pedestrian' in obj.object_type.lower():
                    # 行人：侧向距离只要大于1米就认为安全
                    required_lateral = 1.0
                else:
                    # 车辆：考虑车宽，但给予很大余量
                    vehicle_width = max(ego_state.width, obj.width)
                    required_lateral = vehicle_width * 0.6  # 只要0.6倍车宽的侧向间距
                
                if lateral_dist < required_lateral:
                    return True, f"Point {i+1}: {obj.object_type} lateral too close: {lateral_dist:.1f}m (need {required_lateral:.1f}m)"
                
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
                        if 'pedestrian' in obj.object_type.lower():
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
                            return True, f"Point {i+1}: {obj.object_type} TTC risk: {ttc:.1f}s (need {min_ttc:.1f}s), dist: {longitudinal_dist:.1f}m"
                    
                    else:  # 相对速度很小或负数，使用静态距离判断
                        if 'pedestrian' in obj.object_type.lower():
                            min_static_dist = 3.0
                        else:
                            vehicle_length = max(ego_state.length, obj.length)
                            min_static_dist = vehicle_length * 0.8
                        
                        if longitudinal_dist < min_static_dist:
                            return True, f"Point {i+1}: {obj.object_type} static longitudinal too close: {longitudinal_dist:.1f}m (need {min_static_dist:.1f}m)"
                
                # === 极近距离紧急检查 ===
                # 无论方向，如果总距离极小，都认为有风险
                emergency_distance = 1.5 if 'pedestrian' in obj.object_type.lower() else 2.5
                if total_distance < emergency_distance:
                    return True, f"Point {i+1}: {obj.object_type} emergency close: {total_distance:.1f}m (critical: {emergency_distance:.1f}m)"
        
        return False, ""
    
    def _calculate_min_distance(self, trajectory: List[Tuple[float, float]], 
                               surrounding_objects: List[SurroundingObject]) -> float:
        """Calculate minimum distance"""
        if not surrounding_objects or not trajectory:
            return float('inf')
        
        min_dist = float('inf')
        for x, y in trajectory:
            for obj in surrounding_objects:
                dist = math.sqrt((x - obj.x)**2 + (y - obj.y)**2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _calculate_trajectory_length(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calculate trajectory length"""
        if len(trajectory) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i+1]
            length += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        return length
    
    def _calculate_avg_curvature(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calculate average curvature"""
        if len(trajectory) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
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
    
    def _calculate_forward_progress(self, ego_state: VehicleState, 
                                  trajectory: List[Tuple[float, float]]) -> float:
        """Calculate forward progress"""
        if len(trajectory) < 2:
            return 0.0
        
        end_point = trajectory[-1]
        dx = end_point[0] - ego_state.x
        dy = end_point[1] - ego_state.y
        
        return dx * math.cos(ego_state.heading) + dy * math.sin(ego_state.heading)
    
    def select_better_trajectory(self, ego_state: VehicleState,
                            surrounding_objects: List[SurroundingObject],
                            trajectory1: List[Tuple[float, float]],
                            trajectory2: List[Tuple[float, float]]) -> Tuple[int, str, TrajectoryScore, TrajectoryScore, Dict, Dict]:
        """Select better trajectory and return detailed scores"""
        score1, detailed1 = self.evaluate_trajectory_detailed(ego_state, surrounding_objects, trajectory1)
        score2, detailed2 = self.evaluate_trajectory_detailed(ego_state, surrounding_objects, trajectory2)
        
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
                decision_factors.append(f"Significant safety difference: {safety_diff:.1f}")
            else:
                # 3. Total score comparison (with hysteresis)
                total_diff = score2.total_score - score1.total_score
                hysteresis = 2.0 if self.last_selected_index == 0 else -2.0
                
                if total_diff > hysteresis:
                    selected_index = 1
                    decision_factors.append(f"Total score advantage: {total_diff:.1f}")
                else:
                    selected_index = 0
                    decision_factors.append(f"Maintain selection (hysteresis: {hysteresis:.1f})")
        
        # Generate decision explanation
        winner = "Open Planner" if selected_index == 0 else "Closed Planner"
        winner_score = score1 if selected_index == 0 else score2
        loser_score = score2 if selected_index == 0 else score1
        
        reason_parts = [f"Selected {winner}"]
        reason_parts.extend(decision_factors)
        reason_parts.append(f"Scores: {winner_score.total_score:.1f} vs {loser_score.total_score:.1f}")
        reason_parts.append(f"Weights: Safety{self.weights['safety']:.1%} Comfort{self.weights['comfort']:.1%} Efficiency{self.weights['efficiency']:.1%}")
        
        if winner_score.collision_risk:
            reason_parts.append(f"Warning Winner risk: {winner_score.collision_reason}")
        if loser_score.collision_risk:
            reason_parts.append(f"Error Loser risk: {loser_score.collision_reason}")
        
        self.last_selected_index = selected_index
        return selected_index, " | ".join(reason_parts), score1, score2, detailed1, detailed2

    def get_score_statistics(self) -> Dict:
        """Get score statistics"""
        if not self.score_history:
            return {}
        
        safety_scores = [h['safety'] for h in self.score_history]
        comfort_scores = [h['comfort'] for h in self.score_history]
        efficiency_scores = [h['efficiency'] for h in self.score_history]
        total_scores = [h['total'] for h in self.score_history]
        
        return {
            'safety': {
                'mean': np.mean(safety_scores),
                'std': np.std(safety_scores),
                'min': np.min(safety_scores),
                'max': np.max(safety_scores)
            },
            'comfort': {
                'mean': np.mean(comfort_scores),
                'std': np.std(comfort_scores),
                'min': np.min(comfort_scores),
                'max': np.max(comfort_scores)
            },
            'efficiency': {
                'mean': np.mean(efficiency_scores),
                'std': np.std(efficiency_scores),
                'min': np.min(efficiency_scores),
                'max': np.max(efficiency_scores)
            },
            'total': {
                'mean': np.mean(total_scores),
                'std': np.std(total_scores),
                'min': np.min(total_scores),
                'max': np.max(total_scores)
            }
        }

# ========================= Safe Video Generation Module =========================

class SafeVideoGenerator:
    """Safe video generation for container environments"""
    
    def __init__(self, output_dir="/workspace/nuplan-devkit/visualizations", enable_images=True):
        self.output_dir = Path(output_dir)
        self.enable_images = enable_images
        self.frame_count = 0
        
        if self.enable_images:
            self.output_dir.mkdir(exist_ok=True)
            print(f"Video frames will be saved to: {self.output_dir}")
        
        # Score tracking for video generation
        self.score_data = []
    
    def save_frame_data(self, ego_state: VehicleState, surrounding_objects: List[SurroundingObject],
                    trajectory1: List[Tuple[float, float]], trajectory2: List[Tuple[float, float]],
                    score1: TrajectoryScore, score2: TrajectoryScore, selected_index: int, decision_text: str,
                    detailed1: Dict = None, detailed2: Dict = None):
        """Save frame data for later video generation with detailed scores"""
        self.frame_count += 1
        
        # Store frame data with detailed scores and decision information
        frame_data = {
            'frame': self.frame_count,
            'ego_x': ego_state.x,
            'ego_y': ego_state.y,
            'ego_heading': ego_state.heading,
            'ego_velocity': ego_state.velocity or 0.0,
            'surrounding_count': len(surrounding_objects),
            'open_safety': score1.safety_score,
            'open_comfort': score1.comfort_score,
            'open_efficiency': score1.efficiency_score,
            'open_total': score1.total_score,
            'closed_safety': score2.safety_score,
            'closed_comfort': score2.comfort_score,
            'closed_efficiency': score2.efficiency_score,
            'closed_total': score2.total_score,
            'selected': 'Open' if selected_index == 0 else 'Closed',
            'collision_risk_open': score1.collision_risk,
            'collision_risk_closed': score2.collision_risk,
            'min_distance_open': score1.min_distance,
            'min_distance_closed': score2.min_distance,
            # 保存轨迹和环境数据
            'ego_state': ego_state,
            'surrounding_objects': surrounding_objects,
            'trajectory1': trajectory1,
            'trajectory2': trajectory2,
            'selected_index': selected_index,
            'decision_text': decision_text,  # 新增：保存决策文本
            # 新增：保存详细分数
            'open_detailed': detailed1 or {},
            'closed_detailed': detailed2 or {}
        }
        
        self.score_data.append(frame_data)
        
        # Generate image if enabled and reasonable frequency
        if self.enable_images and self.frame_count % 5 == 0:
            self._generate_frame_image(frame_data)
    
    def _generate_frame_image(self, frame_data):
        """Generate a single frame image safely"""
        try:
            # 2x3布局按照新要求
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 16))
            
            # 第一个位置：详细分数对比（所有15个子指标）
            self._plot_comprehensive_score_comparison(ax1, frame_data)
            
            # 第二个位置：总分历史对比（和以前一样）
            self._plot_score_history(ax2)
            
            # 第三个位置：环境和轨迹视图（和以前一样）
            self._plot_environment_and_trajectories(ax3, frame_data)
            
            # 第四个位置：决策思维链路可视化过程
            self._plot_decision_thinking_chain(ax4, frame_data)
            
            # 第五个位置：关键指标概览（和以前一样）
            self._plot_metrics_radar(ax5, frame_data)
            
            # 第六个位置：距离分析视图（和以前一样）
            self._plot_distance_analysis(ax6, frame_data)
            
            plt.suptitle(f'nuPlan Trajectory Evaluation - Frame {frame_data["frame"]}', fontsize=16)
            plt.tight_layout()
            
            # Save image
            image_path = self.output_dir / f"frame_{frame_data['frame']:06d}.png"
            plt.savefig(image_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            if frame_data['frame'] % 25 == 0:
                print(f"Saved frame {frame_data['frame']}")
                
        except Exception as e:
            print(f"Frame generation error: {e}")
            plt.close('all')  # Ensure cleanup
    
    def _plot_comprehensive_score_comparison(self, ax, frame_data):
        """第一个位置：绘制详细的分数对比，包含所有15个子指标"""
        open_detailed = frame_data.get('open_detailed', {})
        closed_detailed = frame_data.get('closed_detailed', {})
        
        # 创建完整的分数数据结构（所有15个子指标）
        score_data = {
            'Open Planner': {
                'Total': frame_data['open_total'],
                'Safety': frame_data['open_safety'], 
                'Comfort': frame_data['open_comfort'],
                'Efficiency': frame_data['open_efficiency'],
                # Safety子指标 (5个)
                'Safety_Collision': open_detailed.get('collision_risk_score', 0),
                'Safety_MinDist': open_detailed.get('min_distance_score', 0),
                'Safety_AvgDist': open_detailed.get('avg_distance_score', 0),
                'Safety_Encounters': open_detailed.get('encounter_score', 0),
                'Safety_Variance': open_detailed.get('variance_score', 0),
                # Comfort子指标 (5个)
                'Comfort_AvgCurv': open_detailed.get('avg_curvature_score', 0),
                'Comfort_MaxCurv': open_detailed.get('max_curvature_score', 0),
                'Comfort_CurvVar': open_detailed.get('curvature_var_score', 0),
                'Comfort_LenVar': open_detailed.get('length_var_score', 0),
                'Comfort_Length': open_detailed.get('length_score', 0),
                # Efficiency子指标 (5个)
                'Efficiency_Progress': open_detailed.get('progress_score', 0),
                'Efficiency_Lateral': open_detailed.get('lateral_score', 0),
                'Efficiency_PathEff': open_detailed.get('efficiency_score', 0),
                'Efficiency_Alignment': open_detailed.get('alignment_score', 0),
                'Efficiency_Consistency': open_detailed.get('consistency_score', 0)
            },
            'Closed Planner': {
                'Total': frame_data['closed_total'],
                'Safety': frame_data['closed_safety'],
                'Comfort': frame_data['closed_comfort'], 
                'Efficiency': frame_data['closed_efficiency'],
                # Safety子指标
                'Safety_Collision': closed_detailed.get('collision_risk_score', 0),
                'Safety_MinDist': closed_detailed.get('min_distance_score', 0),
                'Safety_AvgDist': closed_detailed.get('avg_distance_score', 0),
                'Safety_Encounters': closed_detailed.get('encounter_score', 0),
                'Safety_Variance': closed_detailed.get('variance_score', 0),
                # Comfort子指标
                'Comfort_AvgCurv': closed_detailed.get('avg_curvature_score', 0),
                'Comfort_MaxCurv': closed_detailed.get('max_curvature_score', 0),
                'Comfort_CurvVar': closed_detailed.get('curvature_var_score', 0),
                'Comfort_LenVar': closed_detailed.get('length_var_score', 0),
                'Comfort_Length': closed_detailed.get('length_score', 0),
                # Efficiency子指标
                'Efficiency_Progress': closed_detailed.get('progress_score', 0),
                'Efficiency_Lateral': closed_detailed.get('lateral_score', 0),
                'Efficiency_PathEff': closed_detailed.get('efficiency_score', 0),
                'Efficiency_Alignment': closed_detailed.get('alignment_score', 0),
                'Efficiency_Consistency': closed_detailed.get('consistency_score', 0)
            }
        }
        
        ax.clear()
        ax.set_title('Comprehensive Score Comparison (All Sub-metrics)', fontsize=12, fontweight='bold')
        
        # 分层显示结构 - 总分 + 3大类 + 15子指标
        categories = [
            'Total',
            'Safety', 'S_Collision', 'S_MinDist', 'S_AvgDist', 'S_Encounters', 'S_Variance',
            'Comfort', 'C_AvgCurv', 'C_MaxCurv', 'C_CurvVar', 'C_LenVar', 'C_Length',
            'Efficiency', 'E_Progress', 'E_Lateral', 'E_PathEff', 'E_Alignment', 'E_Consistency'
        ]
        
        # 对应的数据键
        data_keys = [
            'Total',
            'Safety', 'Safety_Collision', 'Safety_MinDist', 'Safety_AvgDist', 'Safety_Encounters', 'Safety_Variance',
            'Comfort', 'Comfort_AvgCurv', 'Comfort_MaxCurv', 'Comfort_CurvVar', 'Comfort_LenVar', 'Comfort_Length',
            'Efficiency', 'Efficiency_Progress', 'Efficiency_Lateral', 'Efficiency_PathEff', 'Efficiency_Alignment', 'Efficiency_Consistency'
        ]
        
        # 获取分数数据
        open_scores = [score_data['Open Planner'].get(key, 0) for key in data_keys]
        closed_scores = [score_data['Closed Planner'].get(key, 0) for key in data_keys]
        
        # 设置条形图
        y_pos = np.arange(len(categories))
        height = 0.35
        
        # 获胜者高亮显示
        winner = frame_data['selected']
        open_color = '#FF8C00' if winner == 'Open' else '#FFB84D'
        closed_color = '#32CD32' if winner == 'Closed' else '#90EE90'
        
        # 绘制水平条形图
        bars1 = ax.barh(y_pos - height/2, open_scores, height, 
                    label=f'Open {"★" if winner == "Open" else ""}', 
                    color=open_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        bars2 = ax.barh(y_pos + height/2, closed_scores, height,
                    label=f'Closed {"★" if winner == "Closed" else ""}',
                    color=closed_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 设置轴
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Score (0-100)', fontsize=10)
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # 添加分数值
        for i, (bar1, bar2, open_score, closed_score) in enumerate(zip(bars1, bars2, open_scores, closed_scores)):
            if open_score > 5:
                ax.text(open_score/2, bar1.get_y() + bar1.get_height()/2, 
                    f'{open_score:.1f}', ha='center', va='center', 
                    fontsize=7, fontweight='bold', color='white')
            if closed_score > 5:
                ax.text(closed_score/2, bar2.get_y() + bar2.get_height()/2,
                    f'{closed_score:.1f}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')
        
        # 添加分组分隔线
        separator_positions = [0.5, 6.5, 12.5]  # Total后、Safety后、Comfort后
        for pos in separator_positions:
            ax.axhline(y=pos, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        ax.legend(loc='lower right', fontsize=9)
        
        # 添加权重信息
        weights_text = "Weights: Safety 50% | Comfort 25% | Efficiency 25%"
        ax.text(0.02, 0.98, weights_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    def _plot_decision_thinking_chain(self, ax, frame_data):
        """第四个位置：决策思维链路可视化"""
        ax.clear()
        ax.axis('off')
        ax.set_title('Decision Making Chain', fontsize=16, fontweight='bold')
        
        # 解析决策文本
        decision_text = frame_data.get('decision_text', '')
        decision_parts = decision_text.split('|') if decision_text else []
        
        # 获取关键决策信息
        winner = frame_data['selected']
        open_total = frame_data['open_total']
        closed_total = frame_data['closed_total']
        open_safety = frame_data['open_safety']
        closed_safety = frame_data['closed_safety']
        collision_risk_open = frame_data['collision_risk_open']
        collision_risk_closed = frame_data['collision_risk_closed']
        
        # 计算关键差值
        total_diff = abs(open_total - closed_total)
        safety_diff = abs(open_safety - closed_safety)
        
        # 定义三个决策节点的位置
        node_positions = {
            'step1': (0.15, 0.75),
            'step2': (0.15, 0.5),
            'step3': (0.15, 0.25),
            'result': (0.75, 0.4)
        }
        
        # 定义节点大小
        node_radius = 0.08
        
        # 决策逻辑和节点状态
        decision_made_at = None
        decision_reason = ""
        
        # Step 1: 碰撞风险检查
        step1_active = False
        if collision_risk_open or collision_risk_closed:
            if collision_risk_open and not collision_risk_closed:
                step1_active = True
                decision_made_at = 'step1'
                decision_reason = "Closed Planner chosen due to Open collision risk"
            elif collision_risk_closed and not collision_risk_open:
                step1_active = True
                decision_made_at = 'step1' 
                decision_reason = "Open Planner chosen due to Closed collision risk"
            else:
                step1_text = "Both have collision risk"
                step1_color = 'orange'
        else:
            step1_text = "No collision risks detected"
            step1_color = 'green'
        
        if not step1_active:
            if collision_risk_open or collision_risk_closed:
                step1_text = "Both have collision risk"
                step1_color = 'orange'
            else:
                step1_text = "No collision risks"
                step1_color = 'green'
        
        # Step 2: 安全性差异检查
        step2_active = False
        if decision_made_at is None:
            if safety_diff > 15.0:
                step2_active = True
                decision_made_at = 'step2'
                better_safety = "Open" if open_safety > closed_safety else "Closed"
                decision_reason = f"{better_safety} Planner chosen for safety advantage"
                step2_text = f"Large safety gap ({safety_diff:.1f})"
                step2_color = 'orange'
            else:
                step2_text = f"Small safety difference ({safety_diff:.1f})"
                step2_color = 'blue'
        else:
            step2_text = f"Safety difference: {safety_diff:.1f}"
            step2_color = 'lightgray'
        
        # Step 3: 总分对比
        step3_active = False
        if decision_made_at is None:
            step3_active = True
            decision_made_at = 'step3'
            hysteresis = 2.0
            if total_diff > hysteresis:
                better_total = "Open" if open_total > closed_total else "Closed"
                decision_reason = f"{better_total} Planner chosen for score advantage"
                step3_text = f"Score advantage ({total_diff:.1f})"
                step3_color = 'green'
            else:
                decision_reason = f"{winner} Planner maintained (hysteresis)"
                step3_text = f"Maintain choice (diff: {total_diff:.1f})"
                step3_color = 'gray'
        else:
            step3_text = f"Score difference: {total_diff:.1f}"
            step3_color = 'lightgray'
        
        # 绘制决策节点
        nodes_data = [
            ('step1', 'STEP 1\nCollision Risk Check', step1_text, 
            'red' if step1_active else ('green' if step1_color == 'green' else 'orange'), step1_active),
            ('step2', 'STEP 2\nSafety Difference Check', step2_text,
            'red' if step2_active else step2_color, step2_active),
            ('step3', 'STEP 3\nTotal Score Comparison', step3_text,
            'red' if step3_active else step3_color, step3_active)
        ]
        
        # 绘制节点
        for node_id, title, desc, color, is_active in nodes_data:
            x, y = node_positions[node_id]
            
            # 绘制节点圆圈
            circle = plt.Circle((x, y), node_radius, 
                            facecolor='red' if is_active else color,
                            edgecolor='black', 
                            linewidth=3 if is_active else 1,
                            alpha=0.8 if is_active else 0.6,
                            transform=ax.transAxes)
            ax.add_patch(circle)
            
            # 绘制节点标题
            ax.text(x, y + 0.03, title, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='center')
            
            # 绘制节点描述
            ax.text(x, y - 0.03, desc, transform=ax.transAxes,
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 绘制结果节点
        result_x, result_y = node_positions['result']
        result_color = '#FF8C00' if winner == 'Open' else '#32CD32'
        
        # 结果节点（更大）
        result_circle = plt.Circle((result_x, result_y), node_radius * 1.2,
                                facecolor=result_color, edgecolor='black',
                                linewidth=3, alpha=0.9, transform=ax.transAxes)
        ax.add_patch(result_circle)
        
        ax.text(result_x, result_y + 0.04, 'FINAL DECISION', transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='center', va='center')
        ax.text(result_x, result_y, f'{winner}\nPlanner', transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')
        ax.text(result_x, result_y - 0.06, f'Score: {open_total if winner == "Open" else closed_total:.1f}', 
            transform=ax.transAxes, fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # 绘制箭头连接
        arrow_props = dict(arrowstyle='->', lw=2, color='darkblue')
        
        # 步骤间的连接箭头（如果决策没有在该步骤做出）
        if decision_made_at not in ['step1']:
            # Step 1 到 Step 2
            ax.annotate('', xy=(node_positions['step2'][0], node_positions['step2'][1] + node_radius),
                    xytext=(node_positions['step1'][0], node_positions['step1'][1] - node_radius),
                    arrowprops=arrow_props, transform=ax.transAxes)
        
        if decision_made_at not in ['step1', 'step2']:
            # Step 2 到 Step 3
            ax.annotate('', xy=(node_positions['step3'][0], node_positions['step3'][1] + node_radius),
                    xytext=(node_positions['step2'][0], node_positions['step2'][1] - node_radius),
                    arrowprops=arrow_props, transform=ax.transAxes)
        
        # 从决策点到结果的箭头（红色高亮）
        decision_arrow_props = dict(arrowstyle='->', lw=4, color='red')
        
        if decision_made_at:
            decision_node_pos = node_positions[decision_made_at]
            ax.annotate('', xy=(result_x - node_radius * 1.2, result_y),
                    xytext=(decision_node_pos[0] + node_radius, decision_node_pos[1]),
                    arrowprops=decision_arrow_props, transform=ax.transAxes)
            
            # 在箭头上添加决策原因
            mid_x = (decision_node_pos[0] + node_radius + result_x - node_radius * 1.2) / 2
            mid_y = (decision_node_pos[1] + result_y) / 2
            ax.text(mid_x, mid_y + 0.05, decision_reason, transform=ax.transAxes,
                fontsize=11, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # 添加详细信息
        info_text = f"Score Comparison: Open {open_total:.1f} vs Closed {closed_total:.1f}\n"
        info_text += f"Safety Comparison: Open {open_safety:.1f} vs Closed {closed_safety:.1f}\n"
        info_text += f"Decision Weights: Safety 50% | Comfort 25% | Efficiency 25%"
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
   
    def _plot_score_history(self, ax):
        """第二个位置：分数历史（和以前一样）"""
        if len(self.score_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for history', ha='center', va='center')
            ax.set_title('Score History')
            return
        
        frames = [d['frame'] for d in self.score_data[-50:]]
        open_scores = [d['open_total'] for d in self.score_data[-50:]]
        closed_scores = [d['closed_total'] for d in self.score_data[-50:]]
        
        ax.plot(frames, open_scores, 'o-', color='orange', label='Open Planner', 
               linewidth=2, markersize=3)
        ax.plot(frames, closed_scores, 's-', color='green', label='Closed Planner', 
               linewidth=2, markersize=3)
        
        # Highlight current frame
        if frames:
            current_open = open_scores[-1]
            current_closed = closed_scores[-1]
            current_frame = frames[-1]
            
            if current_open > current_closed:
                ax.scatter(current_frame, current_open, s=80, c='orange', marker='*', zorder=5)
            else:
                ax.scatter(current_frame, current_closed, s=80, c='green', marker='*', zorder=5)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Total Score')
        ax.set_title('Score History (Last 50 Frames)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def _plot_environment_and_trajectories(self, ax, frame_data):
        """第三个位置：环境和轨迹视图（修复轨迹起点问题）"""
        if 'ego_state' not in frame_data:
            ax.text(0.5, 0.5, 'Environment data not available', ha='center', va='center')
            ax.set_title('Environment View')
            return
        
        ego_state = frame_data['ego_state']
        surrounding_objects = frame_data['surrounding_objects']
        trajectory1 = frame_data['trajectory1']
        trajectory2 = frame_data['trajectory2']
        selected_index = frame_data['selected_index']
        
        # 设置坐标范围（以ego为中心的50x50米区域）
        plot_range = 25
        ax.set_xlim(ego_state.x - plot_range, ego_state.x + plot_range)
        ax.set_ylim(ego_state.y - plot_range, ego_state.y + plot_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Environment & Trajectories')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # 绘制道路网格
        self._draw_road_network(ax, ego_state, plot_range)
        
        # 绘制ego车辆
        ego_center_x = ego_state.x
        ego_center_y = ego_state.y
        
        ego_width, ego_length = ego_state.width, ego_state.length
        ego_rect = patches.Rectangle(
            (ego_state.x - ego_length/2, ego_state.y - ego_width/2),
            ego_length, ego_width,
            angle=math.degrees(ego_state.heading),
            facecolor='blue', alpha=0.8, edgecolor='darkblue', linewidth=2
        )
        ax.add_patch(ego_rect)
        
        # 绘制ego车辆中心点和方向箭头
        ax.scatter(ego_center_x, ego_center_y, c='darkblue', s=30, marker='o', zorder=10)
        
        arrow_length = 3
        arrow_dx = arrow_length * math.cos(ego_state.heading)
        arrow_dy = arrow_length * math.sin(ego_state.heading)
        ax.arrow(ego_center_x, ego_center_y, arrow_dx, arrow_dy,
                head_width=1, head_length=1, fc='blue', ec='blue', zorder=9)
        
        # 找到最近物体
        closest_obj = None
        min_dist = float('inf')
        for obj in surrounding_objects:
            dist = math.sqrt((obj.x - ego_center_x)**2 + (obj.y - ego_center_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        
        # 绘制周围物体
        for obj in surrounding_objects:
            is_closest = (obj == closest_obj)
            
            if 'pedestrian' in obj.object_type.lower():
                color = 'red' if is_closest else 'orange'
                size = 150 if is_closest else 100
                ax.scatter(obj.x, obj.y, c=color, s=size, marker='o', 
                        edgecolors='black', linewidth=2 if is_closest else 1,
                        alpha=0.8, zorder=5)
            else:  # vehicle
                color = 'darkred' if is_closest else 'gray'
                obj_rect = patches.Rectangle(
                    (obj.x - obj.length/2, obj.y - obj.width/2),
                    obj.length, obj.width,
                    angle=math.degrees(obj.heading),
                    facecolor=color, alpha=0.7 if is_closest else 0.5,
                    edgecolor='black', linewidth=2 if is_closest else 1
                )
                ax.add_patch(obj_rect)
                ax.scatter(obj.x, obj.y, c='black', s=20, marker='+', zorder=6)
        
        # 修复轨迹绘制 - 确保起点与ego中心重合
        if trajectory1:
            # 创建完整轨迹：ego中心 + 轨迹点
            traj1_x = [ego_center_x] + [p[0] for p in trajectory1]
            traj1_y = [ego_center_y] + [p[1] for p in trajectory1]
            
            # 检查轨迹第一个点是否与ego中心重合（允许小误差）
            if trajectory1:
                first_point = trajectory1[0]
                dist_to_ego = math.sqrt((first_point[0] - ego_center_x)**2 + (first_point[1] - ego_center_y)**2)
                
                if dist_to_ego > 0.5:  # 如果距离超过0.5米，说明轨迹起点不在ego中心
                    # 使用纯轨迹点，不添加ego中心
                    traj1_x = [p[0] for p in trajectory1]
                    traj1_y = [p[1] for p in trajectory1]
                    
                    # 绘制从ego中心到轨迹起点的连接线
                    ax.plot([ego_center_x, first_point[0]], [ego_center_y, first_point[1]], 
                        color='orange', linestyle=':', linewidth=2, alpha=0.7, zorder=3)
            
            if selected_index == 0:
                ax.plot(traj1_x, traj1_y, 'o-', color='orange', linewidth=3, 
                    markersize=6, label='Open Planner (Winner)', zorder=4)
                ax.scatter(ego_center_x, ego_center_y, c='orange', s=100, marker='*', 
                        edgecolors='darkorange', linewidth=2, zorder=8)
            else:
                ax.plot(traj1_x, traj1_y, '--', color='orange', linewidth=2, 
                    alpha=0.6, label='Open Planner', zorder=3)
        
        if trajectory2:
            # 创建完整轨迹：ego中心 + 轨迹点
            traj2_x = [ego_center_x] + [p[0] for p in trajectory2]
            traj2_y = [ego_center_y] + [p[1] for p in trajectory2]
            
            # 检查轨迹第一个点是否与ego中心重合（允许小误差）
            if trajectory2:
                first_point = trajectory2[0]
                dist_to_ego = math.sqrt((first_point[0] - ego_center_x)**2 + (first_point[1] - ego_center_y)**2)
                
                if dist_to_ego > 0.5:  # 如果距离超过0.5米，说明轨迹起点不在ego中心
                    # 使用纯轨迹点，不添加ego中心
                    traj2_x = [p[0] for p in trajectory2]
                    traj2_y = [p[1] for p in trajectory2]
                    
                    # 绘制从ego中心到轨迹起点的连接线
                    ax.plot([ego_center_x, first_point[0]], [ego_center_y, first_point[1]], 
                        color='green', linestyle=':', linewidth=2, alpha=0.7, zorder=3)
            
            if selected_index == 1:
                ax.plot(traj2_x, traj2_y, 's-', color='green', linewidth=3, 
                    markersize=6, label='Closed Planner (Winner)', zorder=4)
                ax.scatter(ego_center_x, ego_center_y, c='green', s=100, marker='*', 
                        edgecolors='darkgreen', linewidth=2, zorder=8)
            else:
                ax.plot(traj2_x, traj2_y, '--', color='green', linewidth=2, 
                    alpha=0.6, label='Closed Planner', zorder=3)
        
        # 高亮最近物体的连线
        if closest_obj:
            ax.plot([ego_center_x, closest_obj.x], [ego_center_y, closest_obj.y],
                    'r--', linewidth=2, alpha=0.7, label=f'Closest: {min_dist:.1f}m', zorder=7)
            
            mid_x = (ego_center_x + closest_obj.x) / 2
            mid_y = (ego_center_y + closest_obj.y) / 2
            ax.text(mid_x, mid_y, f'{min_dist:.1f}m', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                ha='center', va='center', zorder=9)
        
        # 绘制安全距离圆圈
        safety_radius = 5.0
        safety_circle = patches.Circle((ego_center_x, ego_center_y), safety_radius, 
                                    fill=False, edgecolor='red', linestyle=':', 
                                    alpha=0.5, linewidth=1.5, zorder=2)
        ax.add_patch(safety_circle)
        
        ax.legend(loc='upper right', fontsize=8)
        
        # 添加轨迹起点检查信息
        info_text = f"Ego Center: ({ego_center_x:.1f}, {ego_center_y:.1f})\n"
        info_text += f"Speed: {ego_state.velocity:.1f} m/s\n"
        info_text += f"Heading: {math.degrees(ego_state.heading):.1f} degrees\n"
        info_text += f"Objects: {len(surrounding_objects)}\n"
        
        # 检查轨迹起点与ego中心的距离
        if trajectory1 and trajectory1:
            dist1 = math.sqrt((trajectory1[0][0] - ego_center_x)**2 + (trajectory1[0][1] - ego_center_y)**2)
            info_text += f"Open traj start dist: {dist1:.2f}m\n"
        
        if trajectory2 and trajectory2:
            dist2 = math.sqrt((trajectory2[0][0] - ego_center_x)**2 + (trajectory2[0][1] - ego_center_y)**2)
            info_text += f"Closed traj start dist: {dist2:.2f}m\n"
        
        if closest_obj:
            info_text += f"Closest: {closest_obj.object_type} ({min_dist:.1f}m)"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) 
     
    def _plot_metrics_radar(self, ax, frame_data):
        """第五个位置：关键指标概览"""
        ax.axis('off')
        ax.set_title('Key Metrics Overview', fontsize=16, fontweight='bold')
        
        # 创建更清晰的指标显示
        metrics_data = [
            ("EGO STATE", [
                f"Position: ({frame_data['ego_x']:.1f}, {frame_data['ego_y']:.1f})",
                f"Velocity: {frame_data['ego_velocity']:.1f} m/s", 
                f"Objects nearby: {frame_data['surrounding_count']}"
            ]),
            ("SAFETY SCORES", [
                f"Open Planner: {frame_data['open_safety']:.1f}",
                f"Closed Planner: {frame_data['closed_safety']:.1f}",
                f"Min Distance (Open): {frame_data['min_distance_open']:.1f}m",
                f"Min Distance (Closed): {frame_data['min_distance_closed']:.1f}m"
            ]),
            ("COLLISION RISK", [
                f"Open Planner: {'YES' if frame_data['collision_risk_open'] else 'NO'}",
                f"Closed Planner: {'YES' if frame_data['collision_risk_closed'] else 'NO'}"
            ]),
            ("TOTAL SCORES", [
                f"Open: {frame_data['open_total']:.1f} (Safety: {frame_data['open_safety']:.1f}, Comfort: {frame_data['open_comfort']:.1f}, Efficiency: {frame_data['open_efficiency']:.1f})",
                f"Closed: {frame_data['closed_total']:.1f} (Safety: {frame_data['closed_safety']:.1f}, Comfort: {frame_data['closed_comfort']:.1f}, Efficiency: {frame_data['closed_efficiency']:.1f})"
            ])
        ]
        
        y_start = 0.95
        y_spacing = 0.2
        
        for i, (category, items) in enumerate(metrics_data):
            y_pos = y_start - i * y_spacing
            
            # 类别标题
            ax.text(0.05, y_pos, category, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', color='darkblue')
            
            # 类别内容
            for j, item in enumerate(items):
                ax.text(0.08, y_pos - (j + 1) * 0.04, f"• {item}", 
                    transform=ax.transAxes, fontsize=12, 
                    fontfamily='monospace')
        
        # 添加获胜者高亮
        winner = frame_data['selected']
        winner_color = '#FF8C00' if winner == 'Open' else '#32CD32'
        
        ax.text(0.7, 0.5, f"SELECTED:\n{winner}\nPlanner", 
            transform=ax.transAxes, fontsize=18, fontweight='bold',
            ha='center', va='center', color=winner_color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=winner_color, alpha=0.2))
    
    def _plot_distance_analysis(self, ax, frame_data):
        """第六个位置：距离分析视图（和以前一样）"""
        if 'surrounding_objects' not in frame_data:
            ax.text(0.5, 0.5, 'Distance data not available', ha='center', va='center')
            ax.set_title('Distance Analysis')
            return
        
        ego_state = frame_data['ego_state']
        surrounding_objects = frame_data['surrounding_objects']
        trajectory1 = frame_data['trajectory1']
        trajectory2 = frame_data['trajectory2']
        
        ax.set_title('Trajectory Distance Analysis')
        ax.set_xlabel('Trajectory Point')
        ax.set_ylabel('Distance to Closest Object (m)')
        
        # 计算两条轨迹各点到最近物体的距离
        if trajectory1 and surrounding_objects:
            distances1 = []
            for point in trajectory1[:15]:
                min_dist = float('inf')
                for obj in surrounding_objects:
                    dist = math.sqrt((point[0] - obj.x)**2 + (point[1] - obj.y)**2)
                    min_dist = min(min_dist, dist)
                distances1.append(min_dist)
            
            ax.plot(range(len(distances1)), distances1, 'o-', color='orange', 
                linewidth=2, label='Open Planner', markersize=4)
        
        if trajectory2 and surrounding_objects:
            distances2 = []
            for point in trajectory2[:15]:
                min_dist = float('inf')
                for obj in surrounding_objects:
                    dist = math.sqrt((point[0] - obj.x)**2 + (point[1] - obj.y)**2)
                    min_dist = min(min_dist, dist)
                distances2.append(min_dist)
            
            ax.plot(range(len(distances2)), distances2, 's-', color='green', 
                linewidth=2, label='Closed Planner', markersize=4)
        
        # 添加安全距离线
        safety_distance = 5.0
        ax.axhline(y=safety_distance, color='red', linestyle='--', alpha=0.7, 
                label=f'Safety Threshold ({safety_distance}m)')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(20, ax.get_ylim()[1]))
    
    def _draw_road_network(self, ax, ego_state, plot_range):
        """绘制道路网格（辅助函数）"""
        road_spacing = 10
        
        min_x = ego_state.x - plot_range
        max_x = ego_state.x + plot_range
        min_y = ego_state.y - plot_range
        max_y = ego_state.y + plot_range
        
        # 绘制垂直和水平道路线
        for x in range(int(min_x // road_spacing) * road_spacing, 
                    int(max_x // road_spacing + 1) * road_spacing, 
                    road_spacing):
            ax.plot([x, x], [min_y, max_y], 'k-', alpha=0.3, linewidth=1, zorder=1)
        
        for y in range(int(min_y // road_spacing) * road_spacing, 
                    int(max_y // road_spacing + 1) * road_spacing, 
                    road_spacing):
            ax.plot([min_x, max_x], [y, y], 'k-', alpha=0.3, linewidth=1, zorder=1)
        
        # 绘制主要道路
        ego_road_y = round(ego_state.y / road_spacing) * road_spacing
        ax.plot([min_x, max_x], [ego_road_y, ego_road_y], 'yellow', alpha=0.6, linewidth=3, zorder=1)
        
        ego_road_x = round(ego_state.x / road_spacing) * road_spacing
        ax.plot([ego_road_x, ego_road_x], [min_y, max_y], 'yellow', alpha=0.6, linewidth=3, zorder=1)
        
        # 绘制车道分隔线
        lane_width = 3.5
        for offset in [-lane_width/2, lane_width/2]:
            ax.plot([min_x, max_x], [ego_road_y + offset, ego_road_y + offset], 
                'white', alpha=0.5, linewidth=1, linestyle='--', zorder=1)
            ax.plot([ego_road_x + offset, ego_road_x + offset], [min_y, max_y], 
                'white', alpha=0.5, linewidth=1, linestyle='--', zorder=1)
    
    # 其余方法保持不变...
    def generate_video_from_images(self, fps=8, output_name="trajectory_evaluation.mp4"):
        """Generate video from saved images"""
        if not self.enable_images:
            print("Image generation was disabled")
            return False
        
        image_files = sorted(list(self.output_dir.glob("frame_*.png")))
        if not image_files:
            print("No images found for video generation")
            return False
        
        output_path = self.output_dir.parent / output_name
        
        print(f"Generating video from {len(image_files)} images...")
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(self.output_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"Video created: {output_path}")
                print(f"Duration: {len(image_files) / fps:.1f} seconds")
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Video generation failed: {e}")
            return False
    
    def save_score_data_csv(self, filename="trajectory_scores.csv"):
        """Save score data as CSV for analysis"""
        if not self.score_data:
            print("No score data to save")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.score_data)
        csv_path = self.output_dir.parent / filename
        df.to_csv(csv_path, index=False)
        print(f"Score data saved: {csv_path}")

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
        
    def __init__(self, parameters: Optional[Parameters] = None, 
                enable_video: bool = True, detailed_logging: bool = True) -> None:
        if parameters is None:
            parameters = EgoAgent.Parameters()
        self.parameters: EgoAgent.Parameters = parameters
        self.environment_model: EnvironmentModel = EnvironmentModel(
            EnvironmentModel.Parameters(self.parameters.trajectory_sampling)
        )
        
        # 🔧 修复：直接实例化规划器，不使用包装器
        print("🔄 Initializing Open Planner...")
        try:
            open_planner_cfg = OmegaConf.load('/workspace/tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_open_planner.yaml')
            self.open = cast(AbstractPlanner, instantiate(open_planner_cfg.pdm_open_planner))
            print("✅ Open Planner initialized successfully")
        except Exception as e:
            print(f"❌ Open Planner initialization failed: {e}")
            # 创建一个简单的备用规划器
            self.open = None
        
        print("🔄 Initializing Closed Planner...")
        try:
            close_planner_cfg = OmegaConf.load('/workspace/tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_closed_planner.yaml')
            self.close = cast(AbstractPlanner, instantiate(close_planner_cfg.pdm_closed_planner))
            print("✅ Closed Planner initialized successfully")
        except Exception as e:
            print(f"❌ Closed Planner initialization failed: {e}")
            # 创建一个简单的备用规划器
            self.close = None
        
        # Use improved evaluator
        self.trajectory_evaluator = ImprovedTrajectoryEvaluator()
        self.video_generator = SafeVideoGenerator(enable_images=enable_video)
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
        
        print(f"✅ EgoAgent initialization complete. Available planners: "
            f"Open={'✓' if self.open else '✗'} Closed={'✓' if self.close else '✗'}")

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
                velocity=getattr(current_ego.dynamic_car_state.rear_axle_velocity_2d, 'magnitude', lambda: 0.0)()
            )
        else:
            ego_state = VehicleState(x=0.0, y=0.0, heading=0.0)
        
        # Extract surrounding objects
        surrounding_objects = []
        observations = current_input.history.observations
        if observations:
            latest_observation = observations[-1]
            if hasattr(latest_observation, 'tracked_objects'):
                for tracked_obj in latest_observation.tracked_objects.tracked_objects:
                    surrounding_objects.append(SurroundingObject(
                        x=tracked_obj.center.x,
                        y=tracked_obj.center.y,
                        heading=tracked_obj.center.heading,
                        length=tracked_obj.box.length,
                        width=tracked_obj.box.width,
                        object_type=getattr(tracked_obj.tracked_object_type, 'name', 'vehicle').lower()
                    ))
        
        return ego_state, surrounding_objects

    def _extract_trajectory_points(self, trajectory):
        """Extract coordinate points from trajectory object"""
        if trajectory is None:
            return []
        
        try:
            if hasattr(trajectory, 'get_sampled_trajectory'):
                sampled_traj = trajectory.get_sampled_trajectory()
                return [(state.rear_axle.x, state.rear_axle.y) for state in sampled_traj]
            elif hasattr(trajectory, 'trajectory'):
                return [(state.rear_axle.x, state.rear_axle.y) for state in trajectory.trajectory]
            else:
                return []
        except:
            return []

    def _log_detailed_decision(self, score1: TrajectoryScore, score2: TrajectoryScore, 
                             selected_index: int, decision_text: str, 
                             ego_state: VehicleState, surrounding_objects: List[SurroundingObject]):
        """Enhanced logging for decision details"""
        if not self.detailed_logging:
            return
        
        winner = "Open" if selected_index == 0 else "Closed"
        winner_score = score1 if selected_index == 0 else score2
        loser_score = score2 if selected_index == 0 else score1
        
        frame = self.trajectory_evaluator.evaluation_count
        for part in decision_text.split('|'):
            print(f"   • {part.strip()}")
        
        # Show score statistics every 50 frames
        if frame % 50 == 0:
            stats = self.trajectory_evaluator.get_score_statistics()
            if stats:
                print(f"\nScore Statistics (last {len(self.trajectory_evaluator.score_history)} frames):")
                for metric, stat in stats.items():
                    print(f"   {metric.capitalize()}: mean={stat['mean']:.1f} std={stat['std']:.1f} range=[{stat['min']:.1f}, {stat['max']:.1f}]")
        
        print(f"{'='*100}\n")

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
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
        selected_index, decision_text, score1, score2, detailed1, detailed2 = self.trajectory_evaluator.select_better_trajectory(
            ego_state, surrounding_objects, traj1_points, traj2_points)
        
        # 调试信息：检查detailed字典
        print(f"DEBUG MAIN: detailed1 type: {type(detailed1)}, keys: {list(detailed1.keys()) if detailed1 else 'None'}")
        print(f"DEBUG MAIN: detailed2 type: {type(detailed2)}, keys: {list(detailed2.keys()) if detailed2 else 'None'}")
        
        # Safe video generation with detailed scores
        self.video_generator.save_frame_data(ego_state, surrounding_objects, traj1_points, traj2_points, 
                                        score1, score2, selected_index, decision_text, detailed1, detailed2)
        
        # Enhanced logging
        self._log_detailed_decision(score1, score2, selected_index, decision_text, ego_state, surrounding_objects)
        
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
        
        # Generate video if enabled
        success = self.video_generator.generate_video_from_images()
        
        # Save score data
        self.video_generator.save_score_data_csv()
        
        # Print summary statistics
        if self.trajectory_evaluator.score_history:
            stats = self.trajectory_evaluator.get_score_statistics()
            print(f"Final Score Statistics:")
            for metric, stat in stats.items():
                print(f"  {metric.capitalize()}: mean={stat['mean']:.1f} std={stat['std']:.1f} range=[{stat['min']:.1f}, {stat['max']:.1f}]")
        
        return success

    def __getstate__(self):
        state = self.__dict__.copy()
        state["video_generator"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.video_generator = SafeVideoGenerator(enable_images=False)
    
    def __del__(self):
        try:
            if hasattr(self, 'video_generator'):
                self.finalize_evaluation()
        except:
            pass

print("Improved normalized trajectory evaluation system loaded successfully!")
print("Configured with proper metric normalization and weighted scoring")
print("All metrics will be distributed uniformly between 0.000 and 100.000")
print("Usage: planner = EgoAgent(enable_video=True, detailed_logging=True)")
import hydra

# Location of paths with all simulation configs
CONFIG_PATH = '../nuplan/planning/script/config/simulation'
CONFIG_NAME = 'default_simulation'

# Create a temporary directory to store the simulation artifacts
SAVE_DIR = '/workspace/nuplan-devkit'

# Select simulation parameters
CHALLENGE = 'closed_loop_reactive_agents' # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
# OBSERVATION = 'idm_agents_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration  
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'group={SAVE_DIR}',
    f'experiment_name=planner_tutorial', 
    f'job_name=planner_tutorial',
    'experiment=${experiment_name}/${job_name}',
    'output_dir=${group}/${experiment}',
    f'+simulation={CHALLENGE}',
    # f'observation={OBSERVATION}',
    'scenario_filter=val14_split',
    'scenario_builder=nuplan',
    # 'worker=sequential',
    'hydra.searchpath=[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]',
    
])
from nuplan.planning.script.run_simulation import run_simulation as main_simulation
import nest_asyncio

nest_asyncio.apply()

planner = EgoAgent()  # 构造 planner

main_simulation(cfg, planners=planner)
simulation_folder = cfg.output_dir

# Print the simulation folder path
print(f"Simulation results are saved in: {simulation_folder}")

# Location of paths with all nuBoard configs
CONFIG_PATH = '../nuplan/planning/script/config/nuboard'
CONFIG_NAME = 'default_nuboard'

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
    f'simulation_path=/workspace/nuplan-devkit/planner_tutorial/planner_tutorial',  # nuboard file path, if left empty the user can open the file inside nuBoard
])
import pandas as pd
from pathlib import Path
from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame

# Step 1: 设置仿真输出目录
output_dir = Path("/workspace/nuplan-devkit/planner_tutorial/planner_tutorial")

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
        metric_statistic_name=metric_name,
        metric_statistics_dataframe=df
    )

# Step 3: 构建 aggregator。表示你希望对所有指标赋等权（或手动设定某些指标高权重），聚合每个 scenario 的多个指标，计算评分。
aggregator = WeightedAverageMetricAggregator(
    name="default_aggregator",
    metric_weights={"default": 1.0},
    file_name="aggregator_metric.parquet",
    aggregator_save_path=output_dir / "aggregator_metric",
    multiple_metrics=[],
    challenge_name=None
)

# Step 4: 运行聚合
aggregator(metric_dataframes)

print("✅ 成功生成 aggregator_metric.parquet！你可以刷新 nuBoard 看 Evaluation Score 了。")
# 修复不连续编号的图片文件并生成视频
import subprocess
from pathlib import Path
import shutil

def fix_video_generation_for_sparse_frames():
    """修复稀疏帧编号并生成视频"""
    
    image_dir = Path("/workspace/nuplan-devkit/visualizations")
    image_files = sorted(list(image_dir.glob("frame_*.png")))
    
    if not image_files:
        print("❌ No frame images found")
        return False
    
    print(f"📁 Found {len(image_files)} images with sparse numbering")
    print(f"📄 First few files: {[f.name for f in image_files[:5]]}")
    
    # 创建临时目录存放重新编号的图片
    temp_dir = image_dir / "temp_renamed"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"🔄 Creating continuous numbering in {temp_dir}")
    
    # 重新编号图片文件
    for i, img_file in enumerate(image_files):
        new_name = f"frame_{i+1:06d}.png"
        new_path = temp_dir / new_name
        shutil.copy2(img_file, new_path)
        
        if i % 50 == 0:
            print(f"📋 Copied {i+1}/{len(image_files)} files")
    
    print(f"✅ Created {len(image_files)} continuously numbered files")
    
    # 生成视频的多种方法
    output_path = image_dir.parent / "trajectory_evaluation_fixed.mp4"
    
    # 方法1: 使用重新编号的文件
    cmd1 = [
        "ffmpeg", "-y",
        "-framerate", "8",
        "-i", str(temp_dir / "frame_%06d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 确保偶数尺寸
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(output_path)
    ]
    
    # 方法2: 使用 glob 模式处理原始文件
    cmd2 = [
        "ffmpeg", "-y",
        "-framerate", "6",
        "-pattern_type", "glob",
        "-i", str(image_dir / "frame_*.png"),
        "-vf", "scale=1588:1180",  # 固定尺寸
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    # 方法3: 创建文件列表
    filelist_path = temp_dir / "filelist.txt"
    with open(filelist_path, 'w') as f:
        for img_file in image_files:
            f.write(f"file '{img_file}'\n")
            f.write("duration 0.125\n")  # 8fps = 1/8 = 0.125秒
    
    cmd3 = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(filelist_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    commands = [
        ("Continuous numbering", cmd1),
        ("Glob pattern", cmd2), 
        ("File list concat", cmd3)
    ]
    
    for method_name, cmd in commands:
        print(f"\n🔄 Trying {method_name}...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                file_size = output_path.stat().st_size / (1024 * 1024)
                duration = len(image_files) / 8
                
                print(f"✅ Video created successfully!")
                print(f"📹 File: {output_path}")
                print(f"📊 Size: {file_size:.1f} MB")
                print(f"⏱️  Duration: ~{duration:.1f} seconds")
                print(f"🎞️  Frames: {len(image_files)}")
                
                # 清理临时文件
                shutil.rmtree(temp_dir, ignore_errors=True)
                return True
            else:
                print(f"❌ {method_name} failed")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                
        except Exception as e:
            print(f"❌ {method_name} error: {e}")
    
    # 清理临时文件
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n❌ All video generation methods failed")
    return False

def create_simple_slideshow():
    """创建简单的幻灯片视频"""
    image_dir = Path("/workspace/nuplan-devkit/visualizations")
    image_files = sorted(list(image_dir.glob("frame_*.png")))
    
    if not image_files:
        return False
    
    output_path = image_dir.parent / "trajectory_slideshow.mp4"
    
    print(f"\n🎬 Creating slideshow from {len(image_files)} images...")
    
    # 最简单的方法：每帧显示0.5秒
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "2",  # 2fps = 每帧0.5秒
        "-pattern_type", "glob",
        "-i", str(image_dir / "frame_*.png"),
        "-c:v", "libx264",
        "-vf", "scale=1600:1200,fps=2",  # 固定尺寸和帧率
        "-pix_fmt", "yuv420p",
        "-t", "30",  # 限制30秒
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Slideshow created: {output_path}")
            print(f"📊 Size: {file_size:.1f} MB")
            return True
        else:
            print(f"❌ Slideshow failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Slideshow error: {e}")
    
    return False

def main_fixed_video_generation():
    """主修复函数"""
    print("🔧 FIXING SPARSE FRAME NUMBERING FOR VIDEO GENERATION")
    print("=" * 70)
    
    # 1. 尝试修复编号并生成视频
    success = fix_video_generation_for_sparse_frames()
    
    # 2. 如果失败，尝试简单幻灯片
    if not success:
        print("\n🔄 Trying slideshow approach...")
        success = create_simple_slideshow()
    
    # 3. 显示结果
    if success:
        print("\n🎉 SUCCESS! Video generated successfully!")
        
        # 列出生成的视频文件
        video_files = list(Path("/workspace/nuplan-devkit").glob("trajectory_*.mp4"))
        for video_file in video_files:
            if video_file.stat().st_size > 1000:
                size_mb = video_file.stat().st_size / (1024 * 1024)
                print(f"📹 {video_file.name}: {size_mb:.1f} MB")
                
        print("\n💡 To download the video:")
        print("   1. Right-click the .mp4 file in VS Code Explorer")
        print("   2. Select 'Download...'")
        print("   3. The video should play in any media player")
    else:
        print("\n❌ All video generation attempts failed")
        print("\n🔍 Debug info:")
        
        # 检查ffmpeg版本
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
            print(f"📋 FFmpeg version: {result.stdout.split()[2]}")
        except:
            print("❌ FFmpeg version check failed")
        
        # 检查第一张图片的详细信息
        image_files = list(Path("/workspace/nuplan-devkit/visualizations").glob("frame_*.png"))
        if image_files:
            first_img = image_files[0]
            print(f"📷 Sample image: {first_img.name}")
            print(f"📊 Size: {first_img.stat().st_size / 1024:.1f} KB")

# 运行修复版本
main_fixed_video_generation()
from nuplan.planning.script.run_nuboard import main as main_nuboard

# Run nuBoard
main_nuboard(cfg)
