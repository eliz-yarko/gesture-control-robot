"""ROS command sender for robot integration."""

from __future__ import annotations

import logging
from typing import Any

from src.config import SenderConfig
from src.domain import CommandEvent, RobotCommand
from src.transmission.base_sender import CommandSender

logger = logging.getLogger(__name__)


class RosCommandSender(CommandSender):
    """Publish movement commands to a ROS ``/cmd_vel`` topic."""

    def __init__(self, config: SenderConfig | None = None) -> None:
        """Initialize ROS sender and lazily import ROS packages."""

        self._config = config or SenderConfig()
        self._publisher: Any | None = None
        self._twist_type: Any | None = None

    def open(self) -> None:
        """Create ROS publisher.

        Raises:
            RuntimeError: If rospy or geometry_msgs is not available.
        """

        try:
            import rospy  # type: ignore[import-not-found]
            from geometry_msgs.msg import Twist  # type: ignore[import-not-found]
        except ImportError as exc:
            message = "rospy and geometry_msgs are required for RosCommandSender."
            raise RuntimeError(message) from exc

        self._twist_type = Twist
        self._publisher = rospy.Publisher(self._config.ros_topic, Twist, queue_size=10)
        logger.info("Opened ROS publisher for topic %s", self._config.ros_topic)

    def send(self, event: CommandEvent) -> None:
        """Publish a ROS Twist message for movement-related commands."""

        if self._publisher is None or self._twist_type is None:
            raise RuntimeError("ROS publisher is not open.")

        twist = self._twist_type()
        _apply_command_to_twist(event.command, twist)
        self._publisher.publish(twist)
        logger.info("Published ROS command: %s", event.command.value)


def _apply_command_to_twist(command: RobotCommand, twist: Any) -> None:
    linear = twist.linear
    angular = twist.angular

    if command == RobotCommand.FORWARD:
        linear.x = 0.2
    elif command == RobotCommand.TURN_LEFT:
        angular.z = 0.6
    elif command == RobotCommand.TURN_RIGHT:
        angular.z = -0.6
    elif command == RobotCommand.ROTATE_360:
        angular.z = 0.8
    elif command in {RobotCommand.STOP, RobotCommand.EMERGENCY_STOP}:
        linear.x = 0.0
        angular.z = 0.0
