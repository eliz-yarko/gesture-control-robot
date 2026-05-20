"""ROS command sender placeholder for robot integration."""

from __future__ import annotations

import logging

from src.config import SenderConfig
from src.domain import CommandEvent, RobotCommand
from src.transmission.base_sender import CommandSender

logger = logging.getLogger(__name__)


class RosCommandSender(CommandSender):
    """Publish movement commands to a ROS ``/cmd_vel`` topic."""

    def __init__(self, config: SenderConfig | None = None) -> None:
        """Initialize ROS sender and lazily import ROS packages."""

        self._config = config or SenderConfig()
        self._publisher: object | None = None
        self._twist_type: type[object] | None = None

    def open(self) -> None:
        """Create ROS publisher.

        Raises:
            RuntimeError: If rospy or geometry_msgs is not available.
        """

        try:
            import rospy  # type: ignore[import-not-found]
            from geometry_msgs.msg import Twist  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("rospy and geometry_msgs are required for RosCommandSender.") from exc

        self._twist_type = Twist
        self._publisher = rospy.Publisher(self._config.ros_topic, Twist, queue_size=10)
        logger.info("Opened ROS publisher for topic %s", self._config.ros_topic)

    def send(self, event: CommandEvent) -> None:
        """Publish a ROS Twist message for movement-related commands."""

        if self._publisher is None or self._twist_type is None:
            raise RuntimeError("ROS publisher is not open.")

        twist = self._twist_type()
        _apply_command_to_twist(event.command, twist)
        publish_method = getattr(self._publisher, "publish")
        publish_method(twist)
        logger.info("Published ROS command: %s", event.command.value)


def _apply_command_to_twist(command: RobotCommand, twist: object) -> None:
    linear = getattr(twist, "linear")
    angular = getattr(twist, "angular")

    if command == RobotCommand.FORWARD:
        setattr(linear, "x", 0.2)
    elif command == RobotCommand.TURN_LEFT:
        setattr(angular, "z", 0.6)
    elif command == RobotCommand.TURN_RIGHT:
        setattr(angular, "z", -0.6)
    elif command == RobotCommand.ROTATE_360:
        setattr(angular, "z", 0.8)
    elif command in {RobotCommand.STOP, RobotCommand.EMERGENCY_STOP}:
        setattr(linear, "x", 0.0)
        setattr(angular, "z", 0.0)
