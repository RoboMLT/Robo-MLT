"""System 2 execution monitor: plan pointer + decision heads + bounded replanning (paper Sec. III-B).

Each update reads the completion probability gamma_t and the recovery probability beta_t of the
active skill and returns one of {stay, advance, recover} (plus ``done`` after the last skill):

    - advance  when gamma_t >= tau_done holds for k_a consecutive updates;
    - recover  when beta_t  >= tau_rec  holds for k_b consecutive updates, or when the pointer has
               rested on one skill for longer than T_max (stall timeout); recover has priority;
    - stay     otherwise.

A recover invokes the replanner (Eq. 5) on the completed prefix C_t, the attempt record R_t and the
current image I_t. Every candidate is checked against the hard protocol-ordering constraints of the
skill library; a rejected candidate is discarded and the replanner is queried again while the
per-episode budget B_replan lasts. When no admissible plan is obtained the pointer stays put.

The ablation variants of Sec. IV-A-4 are switches on the same object (see :data:`VARIANTS`).
The control logic is torch-free and unit-testable with explicit probabilities (test_system2.py).
"""

import logging
import time
from typing import Callable, Dict, List, Optional

try:  # torch is only needed for the real gate; keep logic importable without it
    import torch
except Exception:  # noqa: BLE001 - DLL/init failures included
    torch = None

from high_level_model.planning.planner import Planner
from high_level_model.planning.protocol import ProtocolConstraints
from high_level_model.planning.signal_filters import SignalFilter
from high_level_model.planning.skill_library import SkillLibrary

logger = logging.getLogger(__name__)

__all__ = ["System2Pipeline", "VARIANTS", "variant_flags"]

#: Monitor-side switches of the evaluated variants (paper Sec. IV-A-4). ``single_frame`` differs
#: from ``ours`` only in the monitor checkpoint (trained and deployed with M = 1).
VARIANTS: Dict[str, Dict[str, bool]] = {
    "ours":         {"enable_recovery_head": True,  "enable_stall_timeout": True},
    "single_frame": {"enable_recovery_head": True,  "enable_stall_timeout": True},
    "no_recovery":  {"enable_recovery_head": False, "enable_stall_timeout": False},
    "timeout_only": {"enable_recovery_head": False, "enable_stall_timeout": True},
}

# Old keyword names accepted for backwards compatibility with existing scripts/configs.
_LEGACY_KWARGS = {"tau_back": "tau_rec", "back_ema_alpha": "rec_ema_alpha"}


def variant_flags(variant: str) -> Dict[str, bool]:
    """Monitor switches for a named variant; raises on an unknown name."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'; expected one of {sorted(VARIANTS)}")
    return dict(VARIANTS[variant])


def _scalar(x) -> float:
    """Coerce a float / 0-d or 1-d tensor / list into a python float (first element)."""
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.reshape(-1)[0].item())
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


class System2Pipeline:
    """Plan pointer driven by the completion / recovery heads, with bounded replanning.

    Args:
        tau_done, k_a: advance threshold / debounce (Table I: 0.6 / 2).
        tau_rec, k_b: recover threshold / debounce (Table I: 0.8 / 3).
        t_max_s: per-skill stall timeout T_max in seconds (Table I: 30); ``None``/0 disables it.
        replan_budget: B_replan, the number of replanner calls allowed per episode (Table I: 3).
        cooldown: refractory period in *monitor updates* after a switch. Deployment applies the
            paper's switch cooldown Delta_cd in control steps at the control loop instead, so the
            deployed configs leave this at 0; offline replays can use it.
        enable_recovery_head / enable_stall_timeout: variant switches (see :data:`VARIANTS`).
        rec_ema_alpha, completion_median_window, completion_ema_alpha: optional pre-threshold
            filters; the defaults (1.0 / 1 / 1.0) disable them, matching the paper.
        protocol: ordering constraints; defaults to ``skill_library.protocol``.
        max_plan_attempts: planner queries for the initial plan before giving up (not counted
            against ``replan_budget``).
        clock: monotonic time source in seconds, used for the stall timeout.
        on_replan: optional callback, called with True before and False after a replan, so the
            control loop can hold the arm while the replanner runs.
    """

    def __init__(
        self,
        skill_library: SkillLibrary,
        planner: Planner,
        completion_gate=None,
        task_instruction: Optional[str] = None,
        tau_done: float = 0.6,
        k_a: int = 2,
        tau_rec: float = 0.8,
        k_b: int = 3,
        t_max_s: Optional[float] = 30.0,
        replan_budget: int = 3,
        cooldown: int = 0,
        enable_recovery_head: bool = True,
        enable_stall_timeout: bool = True,
        rec_ema_alpha: float = 1.0,
        completion_median_window: int = 1,
        completion_ema_alpha: float = 1.0,
        protocol: Optional[ProtocolConstraints] = None,
        max_plan_attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
        on_replan: Optional[Callable[[bool], None]] = None,
        **legacy_kwargs,
    ) -> None:
        for old, new in _LEGACY_KWARGS.items():
            if old in legacy_kwargs:
                logger.warning("System2Pipeline: '%s' is deprecated; use '%s'.", old, new)
                value = legacy_kwargs.pop(old)
                if new == "tau_rec":
                    tau_rec = value
                else:
                    rec_ema_alpha = value
        if legacy_kwargs:
            raise TypeError(f"Unexpected System2Pipeline arguments: {sorted(legacy_kwargs)}")

        self.library = skill_library
        self.planner = planner
        self.gate = completion_gate
        self.task_instruction = task_instruction
        self.protocol = protocol if protocol is not None else skill_library.protocol

        self.tau_done, self.k_a = tau_done, k_a
        self.tau_rec, self.k_b = tau_rec, k_b
        self.t_max_s = float(t_max_s) if t_max_s else None
        self.replan_budget_max = replan_budget
        self.cooldown_max = cooldown
        self.enable_recovery_head = bool(enable_recovery_head)
        self.enable_stall_timeout = bool(enable_stall_timeout)
        self.max_plan_attempts = max(1, int(max_plan_attempts))
        self.clock = clock
        self.on_replan = on_replan

        self._rec_filter = SignalFilter(median_window=1, ema_alpha=rec_ema_alpha)
        self._comp_filter = SignalFilter(median_window=completion_median_window,
                                         ema_alpha=completion_ema_alpha)

        self.plan: List[str] = []
        self.pointer: int = 0
        self.memory: List = []           # R_t: (skill_id, outcome) record of attempted skills
        self.plan_admissible: bool = True
        # Scene images handed to the planner: I_0 at reset, the latest frame I_t at a recover.
        self.planner_images = None
        # Optional hook: how many work items (samples, tubes) remain; sizes the offline fallback.
        self.remaining_cycles_fn: Optional[Callable[[], int]] = None
        self._reset_runtime_state(self.clock())

    def _reset_runtime_state(self, now: float) -> None:
        self._adv_count = 0
        self._rec_count = 0
        self._rec_latched = False
        self._cooldown = 0
        self._skill_t0 = now
        self.replan_budget = self.replan_budget_max
        self._done = False
        self._rec_filter.reset()
        self._comp_filter.reset()

    def set_planner_images(self, images) -> None:
        """Cache the latest scene images; the next plan/replan call sends them to the planner."""
        self.planner_images = images

    # ------------------------------------------------------------------ episode
    def reset(self, task_text: str, images=None, now: Optional[float] = None) -> List[str]:
        """Plan a new episode from the task and the initial observation I_0 (Eq. 2)."""
        self.task_instruction = task_text
        if images is not None:
            self.planner_images = images
        candidate: List[str] = []
        violations: List[str] = ["no plan returned"]
        for attempt in range(1, self.max_plan_attempts + 1):
            candidate = self.planner.plan(task_text, images=self.planner_images)
            violations = self.protocol.violations(candidate) if candidate else ["empty plan"]
            if not violations:
                break
            logger.warning("Initial plan rejected (attempt %d/%d): %s",
                           attempt, self.max_plan_attempts, "; ".join(violations[:3]))
            if self.planner.llm_fn is None:
                break  # the offline fallback is deterministic; re-querying cannot help
        self.plan_admissible = not violations
        if not self.plan_admissible:
            logger.error("No admissible initial plan; keeping the last candidate (%d steps).",
                         len(candidate))
        self.plan = list(candidate)
        self.pointer = 0
        self.memory = []
        self._reset_runtime_state(self.clock() if now is None else now)
        logger.info("Planned %d steps: %s", len(self.plan), self.plan)
        return self.plan

    def current_skill_id(self) -> str:
        return self.plan[self.pointer]

    def current_skill_text(self) -> str:
        return self.library.instruction_of(self.current_skill_id())

    def skill_elapsed_s(self, now: Optional[float] = None) -> float:
        return (self.clock() if now is None else now) - self._skill_t0

    # ------------------------------------------------------------------ monitor update
    def step(self, images=None, back_prob=None, completion_prob=None,
             now: Optional[float] = None) -> dict:
        """One monitor update. Provide ``images`` (real gate) or explicit probabilities (tests).

        ``back_prob`` is the recovery probability beta_t (the name follows the dataset column
        ``back_event``). ``now`` overrides the clock, e.g. ``frame_index / fps`` in offline replay.

        Returns {action in {stay, advance, recover, done}, skill_id, back_prob, completion_prob,
        replanned, failure_type?}.
        """
        if not self.plan:
            raise RuntimeError("Pipeline not reset; call reset(task_text) first.")
        replay_now = now  # None = live clock, re-read after a (blocking) replan
        now = self.clock() if now is None else now

        if back_prob is None or completion_prob is None:
            if self.gate is None:
                raise RuntimeError(
                    "No completion_gate set and no explicit back_prob/completion_prob given.")
            comp_t, rec_t = self.gate.predict(images, [self.current_skill_text()])
            completion_prob, back_prob = _scalar(comp_t), _scalar(rec_t)
        back_prob = self._rec_filter.update(float(back_prob))
        completion_prob = self._comp_filter.update(float(completion_prob))

        if self._cooldown > 0:
            self._cooldown -= 1

        info = {"action": "stay", "skill_id": self.current_skill_id(),
                "back_prob": back_prob, "completion_prob": completion_prob, "replanned": False}

        # 1) stall timeout: overrides the heads, so the system cannot rest indefinitely on a
        #    skill the completion head will never confirm (Algorithm 1, line 5).
        if (self.enable_stall_timeout and self.t_max_s is not None
                and now - self._skill_t0 > self.t_max_s):
            return self._recover(info, "stall_timeout", replay_now)

        # 2) recovery head. One sustained disturbance keeps beta_t high for many updates; the
        #    latch makes it count once and re-arms only after beta_t drops below tau_rec, so a
        #    single event cannot burn the whole replan budget on identical replans.
        if self.enable_recovery_head:
            if self._rec_latched:
                if back_prob < self.tau_rec and self._cooldown == 0:
                    self._rec_latched = False
                self._rec_count = 0
            else:
                self._rec_count = (self._rec_count + 1
                                   if back_prob >= self.tau_rec and self._cooldown == 0 else 0)
            if self._rec_count >= self.k_b:
                info = self._recover(info, "recovery_head", replay_now)
                self._rec_latched = True
                return info

        # 3) completion head -> advance (or done after the last skill)
        if completion_prob >= self.tau_done and self._cooldown == 0:
            self._adv_count += 1
        elif completion_prob < self.tau_done:
            self._adv_count = 0
        if self._adv_count >= self.k_a:
            self.memory.append((info["skill_id"], "done"))
            advanced = self.advance()
            info["action"] = "advance" if advanced else "done"
            self._done = not advanced
            self._reset_counts_after_switch(now)
            return info

        return info

    def _recover(self, info: dict, failure_type: str, now: Optional[float]) -> dict:
        info["action"] = "recover"
        info["failure_type"] = failure_type
        info["replanned"] = self._do_recover(failure_type)
        info["skill_id"] = self.current_skill_id()
        # Restart the per-skill timer even when replanning failed, so an exhausted budget is
        # re-attempted at most once per T_max instead of on every update. The timer starts after
        # the replan: the replanner can block for several calls, so the live clock is re-read;
        # an explicit (replay) time is kept as given.
        self._reset_counts_after_switch(self.clock() if now is None else now)
        return info

    def _reset_counts_after_switch(self, now: float) -> None:
        self._adv_count = 0
        self._rec_count = 0
        self._cooldown = self.cooldown_max
        self._skill_t0 = now
        self._rec_filter.reset()
        self._comp_filter.reset()

    # ------------------------------------------------------------------ pointer / recover
    def advance(self) -> bool:
        """Advance the pointer; returns False if already at the last step (episode done)."""
        if self.pointer >= len(self.plan) - 1:
            return False
        self.pointer += 1
        return True

    def _do_recover(self, failure_type: str) -> bool:
        """Revise the remaining plan (Eq. 5) within the replan budget. Returns True on success."""
        if self.replan_budget <= 0:
            logger.warning("Replan budget exhausted; holding the current skill.")
            self.memory.append((self.current_skill_id(), f"recover:{failure_type}:no_budget"))
            return False
        completed = self.plan[: self.pointer]
        skill = self.current_skill_id()
        facts = {
            "infeasible_skill": skill,
            "precondition": self.library.get(skill).precondition_text,
            "postcondition_expected": self.library.get(skill).postcondition_text,
        }
        cycles = 1
        if self.remaining_cycles_fn is not None:
            try:
                cycles = max(1, int(self.remaining_cycles_fn()))
            except Exception as exc:  # noqa: BLE001 - a bad hook must not abort the recovery
                logger.warning("remaining_cycles_fn raised (%s); assuming 1 remaining cycle.", exc)

        accepted: Optional[List[str]] = None
        outcome = "rejected"
        if self.on_replan is not None:
            self.on_replan(True)
        try:
            while self.replan_budget > 0:
                self.replan_budget -= 1
                try:
                    remaining = list(self.planner.replan(
                        self.task_instruction, facts=facts, failure_type=failure_type,
                        completed_ids=completed, memory=self.memory, images=self.planner_images,
                        repeat_cycles=cycles,
                    ))
                except Exception as exc:  # noqa: BLE001 - backend errors depend on the llm_fn
                    # The failed call still counts against the budget. Hold the current skill and
                    # let the latch / T_max decide when to retry instead of re-querying at once.
                    logger.error("Replanner call failed (%d calls left): %s",
                                 self.replan_budget, exc)
                    outcome = "error"
                    break
                candidate = completed + remaining
                violations = self.protocol.violations(candidate) if remaining else ["empty plan"]
                if not violations:
                    accepted = candidate
                    break
                logger.warning("Replan candidate rejected (%d calls left): %s",
                               self.replan_budget, "; ".join(violations[:3]))
                facts["rejected_previous_candidate"] = "; ".join(violations[:3])
                if self.planner.llm_fn is None:
                    break  # deterministic fallback: another query returns the same candidate
        finally:
            if self.on_replan is not None:
                self.on_replan(False)

        if accepted is None:
            self.memory.append((skill, f"recover:{failure_type}:{outcome}"))
            return False
        self.plan = accepted
        self.pointer = len(completed)
        self.memory.append((skill, f"recover:{failure_type}"))
        logger.info("Recovered (%s) -> %s (pointer=%d)", failure_type, self.plan, self.pointer)
        return True

    # ------------------------------------------------------------------ manual override
    def jump_to_skill(self, skill_id: str, source: str = "manual") -> dict:
        """Manually move the pointer to *skill_id* (keyboard override). Not part of the paper
        method — an operational safety affordance for deployment.

        If the skill occurs in the current plan, the pointer jumps to the occurrence nearest to
        the current pointer; otherwise the skill is inserted at the pointer. Counters, cooldown
        and the skill timer reset, and the episode ``done`` flag is cleared.

        Raises:
            KeyError: if *skill_id* is not in the skill library.
        """
        self.library.get(skill_id)  # raises KeyError on unknown skill
        if not self.plan:
            raise RuntimeError("Pipeline not reset; call reset(task_text) first.")

        occurrences = [i for i, sid in enumerate(self.plan) if sid == skill_id]
        if occurrences:
            self.pointer = min(occurrences, key=lambda i: abs(i - self.pointer))
        else:
            self.plan.insert(self.pointer, skill_id)  # pointer now points at it
        self._done = False
        self._reset_counts_after_switch(self.clock())
        self.memory.append((skill_id, f"override:{source}"))
        logger.info("Manual override (%s) -> skill '%s' (pointer=%d)", source, skill_id, self.pointer)
        return {"action": "override", "skill_id": skill_id, "pointer": self.pointer}

    def is_done(self) -> bool:
        """True only after :meth:`step` has emitted ``done`` for the final skill."""
        return self._done
