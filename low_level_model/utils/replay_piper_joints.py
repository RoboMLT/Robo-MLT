"""Replay a recorded Piper keyframe trajectory with smooth interpolation.

Reads the JSON written by :mod:`low_level_model.utils.record_piper_joints` and
drives the arm through its ``points`` **in order**, one keyframe at a time, using
lerobot's ``PiperMotorsBus.move_to_joint_smoothly`` (cccccccccccssmoothstep ease-in/out with
a per-step velocity cap) — the same routine the deployment stack uses for its
home reset.

Usage (from the repo root)::

    python -m low_level_model.utils.replay_piper_joints \\
        --in data/waypoints/traj_a.json [--can-port can_left] \\
        [--duration 3.0] [--dwell 0.5] [--step] [--loops 2]

Safety notes — this script **moves the arm**:
    * It enables the motors (``bus.connect(enable=True)``), which takes the arm
      out of teaching/drag mode.  Clear the workspace first.
    * The very first segment travels from wherever the arm currently is to
      keyframe #0; that leg uses ``--first-duration`` (default 5 s) so it is
      slower than the rest.
    * ``--step`` waits for Enter before each keyframe; ``--dry-run`` sends
      nothing at all and just prints the plan.
    * Ctrl-C stops sending commands immediately; the arm stays enabled and holds
      its position.  ``--safe-disconnect`` instead retreats to the bus's safe
      pose and disables the motors before exiting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from low_level_model.utils.record_piper_joints import JOINT_NAMES

logger = logging.getLogger(__name__)

__all__ = ["load_points", "check_limits", "main"]

# Per-joint software limits, converted from the range documented on
# ``PiperMotorsBus.write`` (raw counts / 57324.840764).  joint_7 is the gripper
# opening in metres.  Used for warnings only, unless --strict-limits is passed.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-1.605, 1.605),
    "joint_2": (-0.042, 2.093),
    "joint_3": (-1.919, 0.052),
    "joint_4": (-1.570, 1.570),
    "joint_5": (-1.396, 1.396),
    "joint_6": (-1.570, 1.570),
    "joint_7": (0.0, 0.08),
}


# ------------------------------------------------------------------- loading
def load_points(path: Path) -> list[list[float]]:
    """Load the ordered 7-DoF keyframes from a recorder JSON file.

    Accepts the recorder's schema (``{"points": [{"joints": [...]}, ...]}``) and,
    for convenience, a bare ``[[...7 floats...], ...]`` list.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        names = data.get("joint_names", JOINT_NAMES)
        if list(names) != JOINT_NAMES:
            raise ValueError(f"{path}: joint_names {names} != {JOINT_NAMES}")
        raw = data.get("points", [])
        points = [p["joints"] if isinstance(p, dict) else p for p in raw]
    else:
        points = list(data)

    out: list[list[float]] = []
    for i, point in enumerate(points):
        vals = [float(v) for v in point]
        if len(vals) != 7:
            raise ValueError(f"{path}: point #{i} has {len(vals)} values, expected 7")
        out.append(vals)
    if not out:
        raise ValueError(f"{path}: no points to replay")
    return out


def check_limits(points: list[list[float]]) -> list[str]:
    """Return a message per keyframe value that falls outside `JOINT_LIMITS`."""
    problems: list[str] = []
    for i, point in enumerate(points):
        for name, value in zip(JOINT_NAMES, point):
            lo, hi = JOINT_LIMITS[name]
            if not (lo <= value <= hi):
                problems.append(
                    f"point #{i} {name}={value:+.4f} outside [{lo:+.3f}, {hi:+.3f}]"
                )
    return problems


def _fmt(joints: list[float]) -> str:
    return " ".join(f"{v:+7.4f}" for v in joints)


# ------------------------------------------------------------------------ cli
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded Piper keyframes with smooth interpolation.",
    )
    parser.add_argument("--in", "-i", dest="in_path", required=True, type=Path,
                        help="input JSON produced by record_piper_joints.py")
    parser.add_argument("--can-port", default="can_left",
                        help="CAN interface name (default: can_left)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="seconds per keyframe segment (default: 3.0)")
    parser.add_argument("--first-duration", type=float, default=5.0,
                        help="seconds for the initial move to keyframe #0 (default: 5.0)")
    parser.add_argument("--hz", type=float, default=100.0,
                        help="interpolation command rate (default: 100)")
    parser.add_argument("--max-step", type=float, default=0.01,
                        help="max per-command joint increment in rad (default: 0.01)")
    parser.add_argument("--dwell", type=float, default=0.5,
                        help="seconds to hold at each keyframe (default: 0.5)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="first keyframe index to replay (default: 0)")
    parser.add_argument("--end-index", type=int, default=None,
                        help="last keyframe index to replay, inclusive (default: last)")
    parser.add_argument("--loops", type=int, default=1,
                        help="how many times to run the sequence (default: 1)")
    parser.add_argument("--step", action="store_true",
                        help="wait for Enter before each keyframe")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without connecting or moving")
    parser.add_argument("--strict-limits", action="store_true",
                        help="abort (instead of warning) when a value exceeds JOINT_LIMITS")
    parser.add_argument("--safe-disconnect", action="store_true",
                        help="on exit, retreat to the bus safe pose and disable the motors")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="skip the interactive 'the arm will move' confirmation")
    return parser.parse_args(argv)


def _make_bus(args: argparse.Namespace):
    """Build a `PiperMotorsBus` carrying the CLI interpolation parameters."""
    from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig

    return PiperMotorsBus(
        PiperMotorsBusConfig(
            can_name=args.can_port,
            motors={name: (i + 1, "agilex_piper") for i, name in enumerate(JOINT_NAMES)},
            reset_hz=args.hz,
            reset_duration_s=args.duration,
            max_joint_step_rad=args.max_step,
        )
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    points = load_points(args.in_path.expanduser())
    end = len(points) - 1 if args.end_index is None else args.end_index
    if not (0 <= args.start_index <= end < len(points)):
        print(f"[error] invalid index range [{args.start_index}, {end}] "
              f"for {len(points)} point(s).")
        return 1
    segment = points[args.start_index:end + 1]

    problems = check_limits(segment)
    for msg in problems:
        print(f"[{'error' if args.strict_limits else 'warn'}] {msg}")
    if problems and args.strict_limits:
        return 1

    print()
    print(f"Input     : {args.in_path}")
    print(f"CAN port  : {args.can_port}")
    print(f"Keyframes : {len(segment)} (index {args.start_index}..{end}), loops={args.loops}")
    print(f"Motion    : {args.duration}s/segment ({args.first_duration}s for the first), "
          f"{args.hz} Hz, max_step={args.max_step} rad, dwell={args.dwell}s")
    for i, point in enumerate(segment, start=args.start_index):
        print(f"  #{i:03d} {_fmt(point)}")
    print()

    if args.dry_run:
        print("[dry-run] nothing sent.")
        return 0

    if not args.yes:
        print("The arm WILL MOVE and the motors will be enabled (teaching mode ends).")
        try:
            reply = input("Clear the workspace, then type 'go' to start: ").strip().lower()
        except EOFError:
            reply = ""
        if reply != "go":
            print("[abort] cancelled.")
            return 1

    bus = _make_bus(args)
    if not bus.connect(enable=True):
        print("[error] failed to enable the arm (timeout) — check power and CAN wiring.")
        return 1

    interrupted = False
    try:
        for loop in range(args.loops):
            for i, point in enumerate(segment, start=args.start_index):
                if args.step:
                    input(f"[loop {loop + 1}/{args.loops}] Enter -> move to #{i:03d} ")
                first = (loop == 0 and i == args.start_index)
                duration = args.first_duration if first else args.duration
                print(f"[loop {loop + 1}/{args.loops}] -> #{i:03d} {_fmt(point)} "
                      f"({duration:.1f}s)")
                bus.move_to_joint_smoothly(
                    point,
                    duration_s=duration,
                    hz=args.hz,
                    max_joint_step_rad=args.max_step,
                )
                if args.dwell > 0:
                    time.sleep(args.dwell)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[stop] interrupted — no further commands sent; the arm holds position.")

    if args.safe_disconnect:
        print("[exit] moving to the safe pose, then disabling the motors...")
        bus.safe_disconnect()
        time.sleep(1.0)
        bus.connect(enable=False)
    else:
        print("[exit] motors left enabled and holding.")

    print("[done] replay finished." if not interrupted else "[done] replay aborted.")
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
