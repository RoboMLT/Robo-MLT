"""Keyframe recorder for a single Piper arm's 7 joint states.

Read-only CAN tool: it opens the arm's CAN port, keeps polling the joint /
gripper feedback, and appends the *current* state to a JSON file every time the
``c`` key is pressed.  Because it never sends an enable / motion command, the
arm can stay in teaching (drag) mode the whole time — you hand-guide it to a
pose, press ``c``, guide it to the next pose, press ``c`` again, …  All
keyframes of one trajectory land in the same JSON file.

Usage (from the repo root) — every argument has a default, so a bare run works::

    python -m low_level_model.utils.record_piper_joints
    # -> data/waypoints/piper_traj_<timestamp>.json, port can_left

    python -m low_level_model.utils.record_piper_joints \\
        --out data/waypoints/tubesort_traj_a.json --can-port can_right

Keys (terminal must stay focused):
    c        capture the current 7-joint state
    u        undo (drop) the last captured point
    s        save now (auto-save is on by default anyway)
    l        list all captured points
    q / ESC  save and quit  (Ctrl-C does the same)

CAN ports are the ones brought up by ``can_config.sh``; ``find_all_can_port.sh``
prints the interfaces currently available (typically ``can_left`` /
``can_right``).  The joint units match the rest of the repo: ``joint_1..6`` in
radians, ``joint_7`` (gripper) in metres.

Output JSON::

    {
      "robot": "piper",
      "can_port": "can_left",
      "joint_names": ["joint_1", ..., "joint_7"],
      "units": {"joint_1": "rad", ..., "joint_7": "m"},
      "created_at": "...", "updated_at": "...",
      "points": [
        {"index": 0, "t_rel_s": 0.0, "timestamp": "...",
         "joints": [...7 floats...],
         "joints_named": {"joint_1": ..., ...}}
      ]
    }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["PiperJointReader", "JointRecording", "main"]

# ------------------------------------------------------------------- defaults
# Site defaults for this workstation: both CAN ports are up at 1 Mbit/s
# (`ip -details -br link show type can`), the left arm sits on `can_left`.
DEFAULT_CAN_PORT = "can_left"
# Recordings land under a gitignored data/ dir at the repo root, so they are never
# accidentally committed. Override with --out for a different location.
DEFAULT_OUT_DIR = Path("data/waypoints")

JOINT_NAMES: list[str] = [f"joint_{i + 1}" for i in range(7)]
JOINT_UNITS: dict[str, str] = {name: "rad" for name in JOINT_NAMES[:6]}
JOINT_UNITS[JOINT_NAMES[6]] = "m"

# SDK scaling, mirrored from lerobot.motors.piper.piper.PiperMotorsBus so the
# direct-SDK fallback produces byte-identical values to the lerobot path.
_JOINT_FACTOR = 57324.840764  # 0.001 deg -> rad
_GRIPPER_FACTOR = 1_000_000.0  # 0.001 mm -> m


# --------------------------------------------------------------------- reader
class PiperJointReader:
    """Read-only view of one Piper arm's 7 joint states over CAN.

    Prefers lerobot's ``PiperMotorsBus`` (single source of truth for the unit
    conversions); falls back to ``piper_sdk`` directly when lerobot is not
    importable.  Neither path enables the motors, so teaching mode is safe.
    """

    def __init__(self, can_port: str) -> None:
        self.can_port = can_port
        self._bus = None
        self._piper = None

        try:
            from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig

            self._bus = PiperMotorsBus(
                PiperMotorsBusConfig(
                    can_name=can_port,
                    motors={name: (i + 1, "agilex_piper") for i, name in enumerate(JOINT_NAMES)},
                )
            )
            logger.info("Piper reader on '%s' via lerobot PiperMotorsBus.", can_port)
        except Exception as exc:  # noqa: BLE001
            logger.info("lerobot PiperMotorsBus unavailable (%s); using piper_sdk directly.", exc)
            from piper_sdk import C_PiperInterface_V2

            self._piper = C_PiperInterface_V2(can_port)
            self._piper.ConnectPort()
            logger.info("Piper reader on '%s' via piper_sdk.", can_port)

    def read(self) -> list[float]:
        """Return the current ``[joint_1..joint_6 (rad), joint_7 (m)]``."""
        if self._bus is not None:
            state = self._bus.read()
            return [float(state[name]) for name in JOINT_NAMES]

        joint_state = self._piper.GetArmJointMsgs().joint_state
        gripper_state = self._piper.GetArmGripperMsgs().gripper_state
        return [
            joint_state.joint_1 / _JOINT_FACTOR,
            joint_state.joint_2 / _JOINT_FACTOR,
            joint_state.joint_3 / _JOINT_FACTOR,
            joint_state.joint_4 / _JOINT_FACTOR,
            joint_state.joint_5 / _JOINT_FACTOR,
            joint_state.joint_6 / _JOINT_FACTOR,
            gripper_state.grippers_angle / _GRIPPER_FACTOR,
        ]


# ------------------------------------------------------------------ recording
class JointRecording:
    """The in-memory trajectory plus its (atomic) JSON serialisation."""

    def __init__(self, out_path: Path, can_port: str, decimals: int = 6,
                 with_degrees: bool = False) -> None:
        self.path = out_path
        self.can_port = can_port
        self.decimals = decimals
        self.with_degrees = with_degrees
        self.points: list[dict] = []
        self.created_at = _now_iso()
        self._t0 = time.time()

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, out_path: Path, can_port: str, **kwargs) -> "JointRecording":
        """Load an existing file so new points are appended to its trajectory."""
        rec = cls(out_path, can_port, **kwargs)
        with open(out_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        names = data.get("joint_names", JOINT_NAMES)
        if list(names) != JOINT_NAMES:
            raise ValueError(f"{out_path}: joint_names {names} != {JOINT_NAMES}")
        if data.get("can_port") and data["can_port"] != can_port:
            logger.warning(
                "Appending to a file recorded on CAN port '%s' while reading '%s'.",
                data["can_port"], can_port,
            )
        rec.points = list(data.get("points", []))
        rec.created_at = data.get("created_at", rec.created_at)
        return rec

    # -- mutation ----------------------------------------------------------
    def add(self, joints: list[float]) -> dict:
        rounded = [round(float(v), self.decimals) for v in joints]
        point = {
            "index": len(self.points),
            "t_rel_s": round(time.time() - self._t0, 3),
            "timestamp": _now_iso(),
            "joints": rounded,
            "joints_named": dict(zip(JOINT_NAMES, rounded)),
        }
        if self.with_degrees:
            point["joints_deg"] = [round(math.degrees(v), 4) for v in rounded[:6]]
        self.points.append(point)
        return point

    def undo(self) -> Optional[dict]:
        if not self.points:
            return None
        dropped = self.points.pop()
        for i, point in enumerate(self.points):  # keep indices contiguous
            point["index"] = i
        return dropped

    # -- io ----------------------------------------------------------------
    def save(self) -> None:
        payload = {
            "robot": "piper",
            "can_port": self.can_port,
            "joint_names": JOINT_NAMES,
            "units": JOINT_UNITS,
            "created_at": self.created_at,
            "updated_at": _now_iso(),
            "num_points": len(self.points),
            "points": self.points,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, self.path)

    def reset_clock(self) -> None:
        """Zero the ``t_rel_s`` clock (called once the CAN link is up)."""
        self._t0 = time.time()


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------- keyboard
class _KeyReader:
    """Non-blocking single-key reader.

    Uses cbreak-mode stdin when running on a TTY (works over SSH, no X needed);
    falls back to the repo's pynput listener otherwise.
    """

    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._saved = None
        self._queue = None
        self._listener = None

    def __enter__(self) -> "_KeyReader":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # cbreak, not raw: Ctrl-C still raises SIGINT
        else:
            import queue

            from low_level_model.robot.keyboard_control import start_keyboard_listener

            self._queue = queue.Queue()
            self._listener = start_keyboard_listener(self._queue)
            if self._listener is None:
                raise RuntimeError(
                    "stdin is not a TTY and pynput is unavailable — no way to read keys."
                )
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        if self._listener is not None:
            self._listener.stop()

    def poll(self) -> list[str]:
        """Return the keys pressed since the last call (may be empty)."""
        if self._fd is not None:
            keys: list[str] = []
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                keys.append("esc" if ch == "\x1b" else ch.lower())
            return keys

        from low_level_model.robot.keyboard_control import drain_keys

        return [k.lower() for k in drain_keys(self._queue)]


# ------------------------------------------------------------------------ cli
def _fmt(joints: list[float]) -> str:
    return " ".join(f"{v:+7.4f}" for v in joints)


def _print(msg: str = "") -> None:
    """Print a line, clearing the in-place ``[live]`` status line first."""
    sys.stdout.write("\r\033[K" + msg + "\n")
    sys.stdout.flush()


def _default_out() -> Path:
    """Timestamped default under `DEFAULT_OUT_DIR` so runs never collide."""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUT_DIR / f"piper_traj_{stamp}.json"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Piper 7-joint keyframes to JSON (press 'c' to capture).",
    )
    parser.add_argument("--out", "-o", type=Path, default=None,
                        help=f"output JSON path (default: {DEFAULT_OUT_DIR}/piper_traj_<timestamp>.json)")
    parser.add_argument("--can-port", default=DEFAULT_CAN_PORT,
                        help=f"CAN interface name (default: {DEFAULT_CAN_PORT}; "
                             f"see find_all_can_port.sh)")
    parser.add_argument("--append", action="store_true",
                        help="append to --out if it already exists (default: refuse)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite --out if it already exists")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="live-display / key-polling rate (default: 20)")
    parser.add_argument("--decimals", type=int, default=6,
                        help="rounding applied to stored joint values (default: 6)")
    parser.add_argument("--with-degrees", action="store_true",
                        help="also store joint_1..6 in degrees under 'joints_deg'")
    parser.add_argument("--no-autosave", action="store_true",
                        help="only write the file on 's' / quit (default: save on every capture)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    out: Path = (args.out or _default_out()).expanduser()
    if out.exists() and not (args.append or args.overwrite):
        _print(f"[error] {out} already exists — pass --append or --overwrite.")
        return 1

    kwargs = {"decimals": args.decimals, "with_degrees": args.with_degrees}
    if out.exists() and args.append:
        rec = JointRecording.load(out, args.can_port, **kwargs)
        _print(f"[info] appending to {out} ({len(rec.points)} existing points)")
    else:
        rec = JointRecording(out, args.can_port, **kwargs)

    reader = PiperJointReader(args.can_port)
    rec.reset_clock()

    # Sanity check: an all-zero, never-changing feedback usually means the CAN
    # port is wrong or the arm is powered off.
    time.sleep(0.3)
    probe = reader.read()
    if all(abs(v) < 1e-9 for v in probe):
        _print(f"[warn] all joints read 0.0 on '{args.can_port}' — check the CAN port "
               f"(./find_all_can_port.sh) and that the arm is powered on.")

    _print("")
    _print(f"CAN port : {args.can_port}")
    _print(f"Output   : {out}")
    _print("Keys     : [c] capture  [u] undo  [s] save  [l] list  [q/ESC] save & quit")
    _print("Keep the arm in teaching mode — this tool never sends motion commands.")
    _print("")

    period = 1.0 / max(args.hz, 1.0)
    quit_requested = False
    try:
        with _KeyReader() as keys:
            while not quit_requested:
                joints = reader.read()
                for key in keys.poll():
                    if key == "c":
                        point = rec.add(joints)
                        if not args.no_autosave:
                            rec.save()
                        _print(f"[capture #{point['index']:03d}] {_fmt(joints)}")
                    elif key == "u":
                        dropped = rec.undo()
                        if dropped is None:
                            _print("[undo] nothing to undo")
                        else:
                            if not args.no_autosave:
                                rec.save()
                            _print(f"[undo] dropped point #{dropped['index']}")
                    elif key == "s":
                        rec.save()
                        _print(f"[save] {len(rec.points)} point(s) -> {out}")
                    elif key == "l":
                        _print(f"[list] {len(rec.points)} point(s):")
                        for point in rec.points:
                            _print(f"  #{point['index']:03d} t={point['t_rel_s']:7.2f}s "
                                   f"{_fmt(point['joints'])}")
                    elif key in ("q", "esc"):
                        quit_requested = True

                sys.stdout.write(f"\r[live] {_fmt(joints)} | captured={len(rec.points)}  ")
                sys.stdout.flush()
                time.sleep(period)
    except KeyboardInterrupt:
        pass

    rec.save()
    _print("")
    _print(f"[done] {len(rec.points)} point(s) saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())