from datetime import timedelta

from nuplan.common.actor_state.state_representation import TimePoint


def to_timedelta(time_point: TimePoint) -> timedelta:
    """
    Converts a TimePoint to a datetime object.

    :param time_point: The TimePoint to convert.
    :return: Corresponding timedelta object.
    """
    return timedelta(microseconds=time_point.time_us)
