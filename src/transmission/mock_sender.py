"""In-memory sender used by tests and offline demos."""

from __future__ import annotations

from src.domain import CommandEvent
from src.transmission.base_sender import CommandSender


class MockCommandSender(CommandSender):
    """Collect commands without touching physical robot hardware."""

    def __init__(self) -> None:
        """Initialize empty command history."""

        self.events: list[CommandEvent] = []

    def send(self, event: CommandEvent) -> None:
        """Store a command event in memory."""

        self.events.append(event)

    @property
    def last_event(self) -> CommandEvent | None:
        """Return the most recently sent event."""

        if not self.events:
            return None
        return self.events[-1]
