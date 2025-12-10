import math
from typing import Dict, List, Tuple

import numpy as np

from common.types import SurroundingObject, TrajectoryScore, VehicleState


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
