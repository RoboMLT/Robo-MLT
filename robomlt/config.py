"""Shared YAML configuration helpers with dot-notation CLI overrides."""

from __future__ import annotations

from typing import Any

import yaml

__all__ = ["apply_overrides", "load_yaml_with_overrides"]


def _set_nested(raw: dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    target = raw
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot apply override {key_path!r}: {key!r} is not a mapping"
            )
        target = child
    target[keys[-1]] = value


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply YAML-decoded ``key=value`` overrides to *raw* and return it."""
    for override in overrides:
        key, separator, value_text = override.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(
                f"Invalid override {override!r}: expected key=value "
                "(dot-notation for nested keys, e.g. robot.port=/dev/ttyACM1)"
            )
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError:
            value = value_text
        _set_nested(raw, key, value)
    return raw


def load_yaml_with_overrides(yaml_path: str, overrides: list[str]) -> dict[str, Any]:
    """Load a mapping from YAML and apply dot-notation CLI overrides."""
    with open(yaml_path, encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Invalid config {yaml_path!r}: expected a YAML mapping, got {type(raw).__name__}"
        )
    return apply_overrides(raw, overrides)
