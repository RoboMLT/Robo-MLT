"""Zero-shot VLM execution monitor for the real-robot tasks (blood gas analysis, tube sorting).

The "why not just ask a VLM?" baseline. It replaces only the learned monitor: plan pointer,
protocol, K-step cadence, debounce and replanner stay those of Robo-MLT. At every System-2 tick a
vision-language model sees the global camera at the start and at the end of the M-frame window plus
the latest wrist views, the active skill instruction and a per-skill success / precondition text,
and answers ``{"completed": bool, "feasible": bool}``; that maps to (gamma, beta) in {0, 1}.

The prompt is a YAML file next to the planner prompts (``configs/skill_library/prompts/``)::

    cameras:          # images sent with every query, in this order; {span} = window length in s
      - {key: observation.images.cam_top, frame: first, caption: "top camera, {span:.1f} s ago"}
      - {key: observation.images.cam_top, frame: last,  caption: "top camera, now"}
    image_max_side: 448   # optional downscale before sending (fewer image tokens, lower latency)
    skills:           # one entry per skill_id of the paired skill library
      <skill_id>: {completed: "...", not_yet: "...", needs: "..."}
    prompt: |         # {images} {text} {completed} {not_yet} {needs}
      ...

:class:`VLMGate` exposes the learned gate's ``predict(images, texts)`` interface on raw [0, 1]
frames (build the controller with ``imagenet_normalize=False``); offline replays call
:meth:`VLMGate.query` with the two window ends directly
(``high_level_model.eval.eval_vlm_monitor_offline``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["MonitorCamera", "MonitorPrompt", "VLMGate", "parse_reply"]

_SKILL_FIELDS = ("completed", "not_yet", "needs")


@dataclass(frozen=True)
class MonitorCamera:
    key: str            # dataset / observation camera key
    frame: str          # "first" or "last" frame of the window
    caption: str        # may use {span}


@dataclass
class MonitorPrompt:
    """Template, camera list and per-skill ``completed`` / ``not_yet`` / ``needs`` text of one task."""

    template: str
    cameras: list[MonitorCamera]
    skills: dict[str, dict[str, str]]
    image_max_side: int | None = None

    @classmethod
    def from_file(cls, path: str) -> MonitorPrompt:
        import yaml  # local import keeps planning yaml-free at import time

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cameras = [MonitorCamera(c["key"], c.get("frame", "last"), c["caption"]) for c in data.get("cameras", [])]
        bad = [c.frame for c in cameras if c.frame not in ("first", "last")]
        if not cameras or bad or not data.get("prompt"):
            raise ValueError(f"{path}: needs a 'prompt' and 'cameras' with frame first|last (got {bad or cameras})")
        skills = {}
        for sid, s in (data.get("skills") or {}).items():
            missing = [k for k in _SKILL_FIELDS if not s.get(k)]
            if missing:
                raise ValueError(f"{path}: skill '{sid}' lacks {missing}")
            skills[sid] = {k: str(s[k]).strip().rstrip(".") for k in _SKILL_FIELDS}
        return cls(template=data["prompt"], cameras=cameras, skills=skills,
                   image_max_side=data.get("image_max_side"))

    def check(self, skill_ids) -> None:
        """Every skill the plan can hold needs its own success / precondition text."""
        missing = [s for s in skill_ids if s not in self.skills]
        if missing:
            raise KeyError(f"monitor prompt has no completed/not_yet/needs text for skills {missing}")

    def render(self, skill_id: str, text: str, span_s: float) -> str:
        images = "\n".join(f"Image {i}: {c.caption.format(span=span_s)}." for i, c in enumerate(self.cameras, 1))
        return self.template.format(images=images, text=text, **self.skills[skill_id])


def _to_uint8(frame, max_side: int | None = None) -> np.ndarray:
    """[C, H, W] float in [0, 1] -> HWC uint8 RGB, longer side at most ``max_side``."""
    arr = frame.detach().cpu().float().clamp(0, 1).numpy() if hasattr(frame, "detach") else np.asarray(frame)
    img = np.ascontiguousarray((arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8))
    if max_side and max(img.shape[:2]) > max_side:
        from PIL import Image

        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        img = np.asarray(Image.fromarray(img).resize((round(w * scale), round(h * scale)), Image.BILINEAR))
    return img


def parse_reply(reply: str) -> tuple[bool, bool] | None:
    """``(completed, feasible)`` from the model's reply, or None if it holds no such JSON object."""
    match = re.search(r"\{.*?\}", reply or "", flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data.get("completed"), bool) or not isinstance(data.get("feasible"), bool):
        return None
    return data["completed"], data["feasible"]


class VLMGate:
    """Zero-shot monitor with the learned gate's ``predict`` interface (see module docstring).

    ``camera_names`` is the camera order of the frames it is given (the controller's / dataset's);
    ``span_s`` the time between the first and the last frame of the window.
    """

    def __init__(self, llm_fn, prompt: MonitorPrompt, library, camera_names, span_s: float):
        self.llm_fn, self.prompt, self.span_s = llm_fn, prompt, float(span_s)
        self._cam = {cam: i for i, cam in enumerate(camera_names)}
        missing = [c.key for c in prompt.cameras if c.key not in self._cam]
        if missing:
            raise ValueError(f"monitor prompt uses cameras {missing} that are not among {list(camera_names)}")
        self._skill_of = {library.instruction_of(sid): sid for sid in library.ids()}
        prompt.check(library.ids())
        self.calls: list[dict] = []

    def query(self, first, last, text: str) -> tuple[float, float]:
        """(gamma, beta) for one window; ``first`` / ``last`` are its end frames, [Cams, C, H, W]
        in [0, 1]. An unparsable reply or an API error is "no decision" (0, 0): stay."""
        skill = self._skill_of[text]
        shots = [_to_uint8((first if c.frame == "first" else last)[self._cam[c.key]], self.prompt.image_max_side)
                 for c in self.prompt.cameras]
        t0 = time.time()
        try:
            reply, error = self.llm_fn(self.prompt.render(skill, text, self.span_s), images=shots), None
        except Exception as exc:  # noqa: BLE001 - an API error must not crash the episode
            reply, error = "", repr(exc)
        parsed = parse_reply(reply)
        self.calls.append({"skill": skill, "latency_s": round(time.time() - t0, 3), "reply": reply[:200],
                           "parsed": parsed is not None, "error": error,
                           "usage": getattr(self.llm_fn, "last_usage", None)})
        if parsed is None:
            return 0.0, 0.0
        completed, feasible = parsed
        return float(completed), float(not feasible)

    def predict(self, images, texts):
        """Learned-gate interface: ``images`` [B, M, Cams, C, H, W] in [0, 1], one text per row
        (a single text is broadcast)."""
        import torch

        out = [self.query(images[b, 0], images[b, -1], texts[b] if len(texts) > 1 else texts[0])
               for b in range(images.shape[0])]
        return torch.tensor([c for c, _ in out]), torch.tensor([r for _, r in out])

    def summary(self) -> dict:
        lat = [c["latency_s"] for c in self.calls]
        tokens = [c["usage"].get("total_tokens", 0) for c in self.calls if c.get("usage")]
        return {"calls": len(self.calls), "latency_median_s": float(np.median(lat)) if lat else None,
                "tokens": int(sum(tokens)), "unparsed": sum(not c["parsed"] for c in self.calls),
                "errors": sum(c["error"] is not None for c in self.calls)}
