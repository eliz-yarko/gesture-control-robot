"""Base interface for robot command senders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain import CommandEvent


class CommandSender(ABC):
    """Abstract transport for confirmed robot commands."""

    @abstractmethod
    def send(self, event: CommandEvent) -> None:
        """Send a confirmed command event to a robot or simulator."""
