"""System 2 half of the asynchronous dual-system runtime (paper Algorithm 1, Sec. III-D).

The control loop calls :meth:`System2Controller.observe` once per control step (20 Hz). The
controller

    - plans the episode from the first observation I_0 (Eq. 2);
    - appends one multiview frame to the history H every K control steps, keeping the last M;
    - once |H| = M, dispatches a snapshot of H to a background worker (job J2) whenever no job is
      in flight, so the monitor re-evaluates a sliding window every K steps (1.33 Hz at K = 15);
    - applies the returned decision: ``advance`` / ``recover`` / ``done`` update the active skill,
      raise a switch flag for the control loop (which then discards the queued action chunk and
      the pending System-1 job), and start a cooldown of Delta_cd control steps without sampling.

A ``recover`` runs the replanner inside J2. :meth:`is_replanning` is set meanwhile so the control
loop can hold the arm, and :meth:`current_task` / :meth:`status` read a cached snapshot so they
never block the control loop on the replanner.

This module has no LeRobot dependency; the robot entry point is ``robot_inference.py``.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

__all__ = ["GATE_SIDECAR_NAME", "resolve_gate_ckpt", "System2Controller"]

GATE_SIDECAR_NAME = "completion_gate_config.yaml"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# A normal monitor update holds the lock for milliseconds; a replan holds it for seconds.
_OVERRIDE_LOCK_TIMEOUT_S = 0.05


def resolve_gate_ckpt(gate_ckpt: str) -> tuple[str, dict]:
    """Resolve ``gate_ckpt`` (a run folder or a ``.pth``) to ``(pth_path, sidecar_cfg)``.

    A folder resolves to ``completion_gate_best.pth`` (else the newest ``*.pth``). The sidecar
    YAML written by training is the source of truth for the monitor architecture, history length
    and camera order; checkpoints without one fall back to the config embedded in the ``.pth``.
    """
    if os.path.isdir(gate_ckpt):
        pth = os.path.join(gate_ckpt, "completion_gate_best.pth")
        if not os.path.isfile(pth):
            cands = sorted(glob.glob(os.path.join(gate_ckpt, "*.pth")), key=os.path.getmtime)
            if not cands:
                raise FileNotFoundError(f"No .pth checkpoint found in folder {gate_ckpt!r}")
            pth = cands[-1]
        sidecar = os.path.join(gate_ckpt, GATE_SIDECAR_NAME)
    else:
        pth = gate_ckpt
        sidecar = os.path.join(os.path.dirname(pth), GATE_SIDECAR_NAME)

    if os.path.isfile(sidecar):
        with open(sidecar, encoding="utf-8") as f:
            return pth, (yaml.safe_load(f) or {})

    try:
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        cfg = ck.get("config") if isinstance(ck, dict) else None
    except Exception:  # noqa: BLE001 - an unreadable embedded config just means "use defaults"
        cfg = None
    if isinstance(cfg, dict):
        ds = cfg.get("dataset", {})
        return pth, {
            "camera_names": ds.get("camera_names"),
            "history_len": ds.get("history_len"),
            "history_skip_frame": ds.get("history_skip_frame"),
            "model": cfg.get("model", {}),
        }
    return pth, {}


def _load_gate(pth: str, gate_cfg: dict, history_len: int, device: str):
    from high_level_model.models.completion_gate import CompletionGate, gate_kwargs_from_config
    from high_level_model.models.siglip_encoder import build_siglip_encoder

    backbone = build_siglip_encoder(device, history_len, freeze_siglip=True).to(device)
    gate = CompletionGate(backbone, freeze_siglip=True,
                          **gate_kwargs_from_config(gate_cfg.get("model"))).to(device)
    ck = torch.load(pth, map_location=device, weights_only=False)
    sd_ck = ck.get("state_dict", ck)
    sd_model = gate.state_dict()
    filtered = {k: v for k, v in sd_ck.items() if k in sd_model and sd_model[k].shape == v.shape}
    missing = sorted(set(sd_model) - set(filtered) - {k for k in sd_model if k.startswith("backbone.")})
    if missing:
        logger.warning("Monitor checkpoint is missing %d trainable tensors (e.g. %s) — check that "
                       "the sidecar describes the trained architecture.", len(missing), missing[:3])
    gate.load_state_dict(filtered, strict=False)
    gate.eval()
    return gate


class System2Controller:
    """Background System-2 worker driven by the control loop (see the module docstring).

    Args:
        pipeline: a :class:`~high_level_model.planning.system2_pipeline.System2Pipeline` whose
            ``gate`` is set; its ``on_replan`` hook is taken over by the controller.
        task_instruction: the high-level task handed to the planner.
        camera_names: observation keys fed to the monitor, in the trained order.
        history_len: M, the number of frames in the monitor window.
        sampling_interval: K, control steps between two sampled frames.
        cooldown_steps: Delta_cd, control steps without sampling after a switch.
        planner_camera: observation key of the global view sent to the planner (I_0, I_t).
        allow_inadmissible_plan: start even if no initial plan satisfies the protocol.
    """

    def __init__(
        self,
        pipeline,
        task_instruction: str,
        camera_names: list[str],
        device: str = "cuda",
        history_len: int = 6,
        sampling_interval: int = 15,
        cooldown_steps: int = 10,
        planner_camera: Optional[str] = "observation.images.cam_top",
        allow_inadmissible_plan: bool = False,
        imagenet_normalize: bool = True,
    ) -> None:
        if pipeline.gate is None:
            raise ValueError("System2Controller needs a pipeline with a monitor (pipeline.gate).")
        self.pipeline = pipeline
        self.pipeline.on_replan = self._on_replan
        self.task_instruction = task_instruction
        self.camera_names = list(camera_names)
        self.device = device
        self.history_len = max(1, int(history_len))
        self.sampling_interval = max(1, int(sampling_interval))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.planner_camera = planner_camera
        self.allow_inadmissible_plan = bool(allow_inadmissible_plan)

        # Must match the monitor's training transform (sidecar ``imagenet_normalize``).
        self._mean = torch.tensor(_IMAGENET_MEAN if imagenet_normalize else (0.0, 0.0, 0.0)).view(3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD if imagenet_normalize else (1.0, 1.0, 1.0)).view(3, 1, 1)
        self._lock = threading.Lock()
        self._replanning = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        # Bumped by reset() and manual overrides; a J2 job launched under an older generation
        # returns without stepping the pipeline, so it cannot act on a plan it was not run for.
        self._generation = 0
        self._reset_loop_state()

    @classmethod
    def from_checkpoint(
        cls,
        gate_ckpt: str,
        skill_library: str,
        task_instruction: str,
        camera_names: list[str],
        device: str = "cuda",
        llm_fn=None,
        prompt_path: str = "",
        sampling_interval: int = 15,
        cooldown_steps: int = 10,
        fps: float = 20.0,
        planner_camera: Optional[str] = "observation.images.cam_top",
        allow_inadmissible_plan: bool = False,
        **pipeline_kwargs,
    ) -> System2Controller:
        """Build planner, monitor and pipeline from a trained checkpoint and a skill library."""
        from high_level_model.planning.planner import Planner
        from high_level_model.planning.prompt_templates import load_prompt_set
        from high_level_model.planning.skill_library import SkillLibrary
        from high_level_model.planning.system2_pipeline import System2Pipeline

        pth, gate_cfg = resolve_gate_ckpt(gate_ckpt)
        history_len = int(gate_cfg.get("history_len") or 1)
        trained_cams = gate_cfg.get("camera_names")
        if trained_cams and list(trained_cams) != list(camera_names):
            logger.warning("System-2 camera_names %s differ from the trained order %s — using the "
                           "trained order.", camera_names, trained_cams)
            camera_names = list(trained_cams)
        _check_history_spacing(gate_cfg, sampling_interval, fps)

        library = SkillLibrary.from_file(skill_library)
        planner = Planner(library, llm_fn=llm_fn, prompt_set=load_prompt_set(prompt_path))
        gate = _load_gate(pth, gate_cfg, history_len, device)
        unseen = sorted(set(library.instructions()) - set(gate_cfg.get("skill_texts") or []))
        if gate_cfg.get("skill_texts") and unseen:
            logger.warning("Skill instructions not seen during monitor training: %s (check %s "
                           "for rewording).", unseen, skill_library)
        pipeline = System2Pipeline(library, planner, completion_gate=gate, **pipeline_kwargs)
        logger.info("System 2: M=%d, K=%d, cooldown=%d steps, trunk=%s, %s trainable params.",
                    history_len, sampling_interval, cooldown_steps,
                    gate.fusion, gate_cfg.get("trainable_params", "?"))
        return cls(pipeline, task_instruction, camera_names, device=device,
                   history_len=history_len, sampling_interval=sampling_interval,
                   cooldown_steps=cooldown_steps, planner_camera=planner_camera,
                   allow_inadmissible_plan=allow_inadmissible_plan,
                   imagenet_normalize=bool(gate_cfg.get("imagenet_normalize", True)))

    # ------------------------------------------------------------------ public API
    def current_task(self) -> str:
        """Active skill instruction l_curr (the high-level task until the first plan exists)."""
        return self._skill_text

    def is_done(self) -> bool:
        return self._done

    def is_replanning(self) -> bool:
        return self._replanning.is_set()

    def consume_switch(self) -> bool:
        """True once after every advance / recover / override (flush System 1 when it is)."""
        switched, self._switched = self._switched, False
        return switched

    def status(self) -> dict:
        """Snapshot for the status view; never waits for a running replan."""
        if not self._lock.acquire(blocking=False):
            return dict(self._status_cache, replanning=True)
        try:
            p = self.pipeline
            info = self._last_info or {}
            self._status_cache = {
                "task": p.task_instruction or self.task_instruction,
                "plan": list(p.plan),
                "pointer": p.pointer,
                "skill_id": p.current_skill_id() if p.plan else None,
                "skill_text": self._skill_text,
                "back_prob": info.get("back_prob"),
                "completion_prob": info.get("completion_prob"),
                "action": info.get("action"),
                "replan_budget": p.replan_budget,
                "is_done": self._done,
                "replanning": self.is_replanning(),
            }
        finally:
            self._lock.release()
        return dict(self._status_cache)

    def reset(self, task_instruction: str) -> None:
        """Start a new episode; the plan is made from the next observation (I_0). A monitor job
        still in flight is dropped (its result is never applied)."""
        with self._lock:
            self._generation += 1
        self._future = None
        self.task_instruction = task_instruction
        self._reset_loop_state()

    def observe(self, raw_observation: dict) -> None:
        """One control step of Algorithm 1 (System-2 branch). Non-blocking except at the very
        first call, which plans the episode synchronously before any action is executed."""
        if not self._planned:
            self._plan_initial(raw_observation)

        # A switch starts the cooldown; a ``stay`` keeps the K-step sampling cadence unbroken, so
        # the window is re-evaluated every K steps (1.33 Hz at K = 15, f_ctrl = 20 Hz).
        if self._future is not None and self._future.done() and self._collect_result():
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            return

        self._t += 1
        if self._t % self.sampling_interval != 0:
            return
        frame = self._extract_cameras(raw_observation)
        if frame is None:
            return
        self._history.append(frame)
        self._scene = self._planner_images(raw_observation)
        if len(self._history) == self.history_len and self._future is None:
            window = torch.stack(list(self._history), dim=0).unsqueeze(0)   # [1, M, Cams, C, H, W]
            self._future = self._executor.submit(self._infer, window, self._scene,
                                                 self._generation)

    def manual_override(self, digit: int, manager) -> None:
        """Keyboard override (not part of the method): jump the pointer to a skill by digit;
        digit 0 only logs the current plan."""
        if digit == 0:
            logger.info("System 2 status: pointer=%d/%d plan=%s",
                        self.pipeline.pointer, len(self.pipeline.plan), self.pipeline.plan)
            return
        skill_id = manager.skill_for_digit(digit)
        if skill_id is None:
            logger.warning("Keyboard override: digit %d has no skill; ignored.", digit)
            return
        # A replan holds the lock for up to B_replan LLM calls; never stall the control loop on it.
        if not self._lock.acquire(timeout=_OVERRIDE_LOCK_TIMEOUT_S):
            logger.warning("Keyboard override ignored: System 2 is replanning; press again later.")
            return
        try:
            self.pipeline.jump_to_skill(skill_id, source="keyboard")
            self._generation += 1
            self._refresh_cache()
        finally:
            self._lock.release()
        self._history.clear()
        self._switched = True
        self._cooldown = self.cooldown_steps
        logger.info("Keyboard override -> skill '%s' (%s)", skill_id, self._skill_text)

    def shutdown(self) -> None:
        # Non-blocking: a wedged monitor job must not stall the cleanup path.
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ internals
    def _reset_loop_state(self) -> None:
        self._history: deque = deque(maxlen=self.history_len)
        self._t = 0
        self._cooldown = 0
        self._planned = False
        self._switched = False
        self._done = False
        self._scene = None
        self._last_info: Optional[dict] = None
        self._skill_text = self.task_instruction
        self._status_cache = {"task": self.task_instruction, "skill_text": self.task_instruction}

    def _on_replan(self, active: bool) -> None:
        if active:
            self._replanning.set()
            logger.info("System 2: replanning — holding the arm.")
        else:
            self._replanning.clear()

    def _refresh_cache(self) -> None:
        """Called with the lock held after anything that may move the pointer."""
        p = self.pipeline
        self._skill_text = p.current_skill_text() if p.plan else self.task_instruction
        self._done = p.is_done()

    def _plan_initial(self, raw_observation: dict) -> None:
        scene = self._planner_images(raw_observation)
        with self._lock:
            self.pipeline.reset(self.task_instruction, images=scene)
            if not self.pipeline.plan_admissible and not self.allow_inadmissible_plan:
                raise RuntimeError(
                    "No initial plan satisfies the protocol constraints (see the log). Fix the "
                    "planner prompt / scene, or set allow_inadmissible_plan=true to run anyway.")
            self._refresh_cache()
        self._planned = True
        logger.info("System 2 initial plan (%d steps): %s", len(self.pipeline.plan), self.pipeline.plan)

    def _collect_result(self) -> bool:
        """Apply the finished J2 decision; True when it switched the active skill."""
        future, self._future = self._future, None
        try:
            info = future.result()
        except Exception as exc:  # noqa: BLE001
            logger.error("System-2 job failed: %s", exc, exc_info=True)
            return False
        if info.get("stale"):
            return False
        self._last_info = info
        if info.get("action", "stay") == "stay":
            return False
        self._switched = True
        self._cooldown = self.cooldown_steps
        logger.info("System 2: %s -> skill '%s'", info["action"], self._skill_text)
        return True

    def _infer(self, window: torch.Tensor, scene, generation: int) -> dict:
        with self._lock:
            # The latest scene doubles as I_t should this update trigger a replan.
            self.pipeline.set_planner_images(scene)
            skill_text = self.pipeline.current_skill_text()
        with torch.inference_mode():
            comp_t, rec_t = self.pipeline.gate.predict(window.to(self.device), [skill_text])
        with self._lock:
            if generation != self._generation:   # reset / override since launch
                return {"action": "stay", "stale": True}
            info = self.pipeline.step(back_prob=float(rec_t.reshape(-1)[0]),
                                      completion_prob=float(comp_t.reshape(-1)[0]))
            self._refresh_cache()
        return info

    def _camera(self, raw_observation: dict, cam: str):
        key = cam if cam in raw_observation else cam.removeprefix("observation.images.")
        return raw_observation.get(key)

    def _extract_cameras(self, raw_observation: dict) -> Optional[torch.Tensor]:
        """Multiview frame ``[Cams, C, H, W]``, ImageNet-normalised as in training."""
        tensors = []
        for cam in self.camera_names:
            img = self._camera(raw_observation, cam)
            if img is None:
                logger.warning("Camera '%s' missing from observation; skipping System-2 sample", cam)
                return None
            img = img if isinstance(img, torch.Tensor) else torch.from_numpy(np.asarray(img))
            if img.dim() == 3 and img.shape[-1] == 3:     # HWC -> CHW
                img = img.permute(2, 0, 1)
            img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
            tensors.append((img - self._mean) / self._std)
        return torch.stack(tensors, dim=0)

    def _planner_images(self, raw_observation: dict):
        """The global view as a list with one HWC uint8 RGB array (None if unavailable)."""
        if not self.planner_camera:
            return None
        img = self._camera(raw_observation, self.planner_camera)
        if img is None:
            logger.warning("Planner camera '%s' missing from observation.", self.planner_camera)
            return None
        arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:   # CHW -> HWC
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return [arr.copy()]


def _check_history_spacing(gate_cfg: dict, sampling_interval: int, fps: float) -> None:
    """Warn when the deployed frame spacing (K / f_ctrl) differs from the trained one."""
    skip, data_fps = gate_cfg.get("history_skip_frame"), gate_cfg.get("dataset_fps")
    if not skip or not data_fps or int(gate_cfg.get("history_len") or 1) <= 1:
        return
    trained_s, deployed_s = float(skip) / float(data_fps), float(sampling_interval) / float(fps)
    if abs(trained_s - deployed_s) > 1.0 / float(data_fps):
        logger.warning("Monitor trained with %.2f s between history frames but deployed with "
                       "%.2f s (K=%d at %.0f Hz).", trained_s, deployed_s, sampling_interval, fps)
