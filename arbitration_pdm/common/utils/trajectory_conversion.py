def trajectory_to_points(trajectory):
    """
    TODO: Use nuplan's built-in trajectory types instead of this conversion
    """
    if trajectory is None:
        return []

    try:
        if hasattr(trajectory, "get_sampled_trajectory"):
            sampled_traj = trajectory.get_sampled_trajectory()
            return [(state.rear_axle.x, state.rear_axle.y) for state in sampled_traj]
        elif hasattr(trajectory, "trajectory"):
            return [
                (state.rear_axle.x, state.rear_axle.y)
                for state in trajectory.trajectory
            ]
        else:
            return []
    except:
        return []
