"""Shared keyboard listener utilities for Robo-MLT robot scripts.

Provides a thin pynput-based keyboard listener that queues key names into a
:class:`queue.Queue`.  Both ``robot_inference.py`` (deployment keyboard override)
and ``collect_dagger_dataset.py`` (DAgger recording controls) import from here.

If pynput is unavailable (headless server, missing X display, etc.) the listener
degrades gracefully: :func:`start_keyboard_listener` returns ``None`` and a
warning is logged, while :func:`drain_keys` returns an empty list.
"""

from __future__ import annotations

import logging
import queue
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["start_keyboard_listener", "drain_keys"]


def start_keyboard_listener(event_queue: "queue.Queue") -> Optional[object]:
    """Start a background pynput keyboard listener.

    Pressed keys are pushed onto *event_queue* as strings: ``key.char`` for
    printable characters (e.g. ``"c"``, ``"1"``), otherwise ``key.name``
    (e.g. ``"esc"``, ``"space"``).

    Args:
        event_queue: A :class:`queue.Queue` instance that receives key names.

    Returns:
        The running :class:`pynput.keyboard.Listener`, or ``None`` if pynput is
        not available or fails to start.
    """
    try:
        from pynput import keyboard
    except Exception as exc:  # noqa: BLE001
        logger.warning("pynput unavailable (%s); keyboard controls disabled.", exc)
        return None

    def on_press(key):
        try:
            char = (key.char if (hasattr(key, "char") and key.char is not None)
                    else key.name)
            event_queue.put(char)
        except Exception:  # noqa: BLE001
            pass

    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        return listener
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to start keyboard listener (%s); keyboard controls disabled.", exc
        )
        return None


def drain_keys(event_queue: "queue.Queue") -> list[str]:
    """Drain all pending key events without blocking.

    Args:
        event_queue: The same queue passed to :func:`start_keyboard_listener`.

    Returns:
        A list of key name strings in arrival order (may be empty).
    """
    keys: list[str] = []
    while True:
        try:
            keys.append(event_queue.get_nowait())
        except queue.Empty:
            break
    return keys
