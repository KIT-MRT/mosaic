from dataclasses import dataclass


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
