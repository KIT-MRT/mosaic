from dataclasses import dataclass
from typing import Optional


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
