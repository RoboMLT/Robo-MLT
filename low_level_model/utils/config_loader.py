"""YAML-first CLI config loading for the robot scripts.

Mirrors ``train_system1.py``'s configuration style — all parameters live in a
YAML file, and individual fields can be overridden on the command line with
``key=value`` syntax (dot-notation for nested keys)::

    python -m low_level_model.robot.robot_inference \\
        configs/system1/inference/inference.yaml \\
        robot.port=/dev/ttyACM1 overlap_steps=6

Unlike the training script, the target configs here contain draccus
ChoiceRegistry fields (``robot.type`` / ``teleop.type`` select a lerobot config
subclass), so the merged dict is decoded with :func:`draccus.decode` instead of
hand-rolled dataclass filling.

For backward compatibility, when the first CLI argument is *not* a YAML path
the arguments are parsed as a plain draccus dotlist
(``--robot.type=so101_follower ...``), exactly as before.
"""

from __future__ import annotations

import sys
from typing import Optional, Type, TypeVar

import draccus

from robomlt.config import apply_overrides, load_yaml_with_overrides

T = TypeVar("T")

__all__ = ["apply_overrides", "load_yaml_with_overrides", "parse_with_yaml"]


def parse_with_yaml(config_cls: Type[T], argv: Optional[list[str]] = None) -> T:
    """Parse *config_cls* from ``<config.yaml> [key=value ...]`` or a draccus dotlist.

    Mode selection on the first argument:
        - ends with ``.yaml``/``.yml`` and doesn't start with ``-``
          → YAML mode: load the file, apply ``key=value`` overrides, decode the
          merged dict via :func:`draccus.decode` (ChoiceRegistry fields such as
          ``robot.type`` resolve to the registered lerobot config subclass).
        - otherwise → plain draccus CLI (``--robot.type=... --fps=20``),
          identical to the previous behaviour (including ``--help``).
    """
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    if argv and not argv[0].startswith("-") and argv[0].endswith((".yaml", ".yml")):
        raw = load_yaml_with_overrides(argv[0], argv[1:])
        return draccus.decode(config_cls, raw)

    return draccus.parse(config_cls, args=argv)
