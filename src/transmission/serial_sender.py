"""UART command sender for microcontroller-based robot platforms."""

from __future__ import annotations

import logging
from types import TracebackType

from src.config import SenderConfig
from src.domain import CommandEvent
from src.transmission.base_sender import CommandSender

logger = logging.getLogger(__name__)


class SerialCommandSender(CommandSender):
    """Send commands through a serial UART connection."""

    def __init__(self, config: SenderConfig | None = None) -> None:
        """Initialize sender configuration without opening the port."""

        self._config = config or SenderConfig()
        self._serial_connection: object | None = None

    def __enter__(self) -> "SerialCommandSender":
        """Open serial connection when entering a context manager."""

        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close serial connection when leaving a context manager."""

        self.close()

    def open(self) -> None:
        """Open configured serial port.

        Raises:
            RuntimeError: If pySerial is not installed.
        """

        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("pySerial is required for SerialCommandSender.") from exc

        self._serial_connection = serial.Serial(
            port=self._config.serial_port,
            baudrate=self._config.baudrate,
            timeout=self._config.timeout_seconds,
        )
        logger.info("Opened serial port %s", self._config.serial_port)

    def close(self) -> None:
        """Close serial port if it is open."""

        if self._serial_connection is None:
            return
        close_method = getattr(self._serial_connection, "close")
        close_method()
        logger.info("Closed serial port %s", self._config.serial_port)
        self._serial_connection = None

    def send(self, event: CommandEvent) -> None:
        """Write a command event as a newline-terminated ASCII packet."""

        if self._serial_connection is None:
            raise RuntimeError("Serial port is not open.")
        packet = f"{event.command.value};{int(event.gesture_id)};{event.confidence:.3f}\n"
        write_method = getattr(self._serial_connection, "write")
        write_method(packet.encode("ascii"))
        logger.info("Sent command over UART: %s", event.command.value)
