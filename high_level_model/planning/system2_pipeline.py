"""System 2 runtime: planner + pointer controller (completion / back) + replanning.

Decoupled modules:
    - Module A (planner): frozen LLM emits an ordered skill plan from the task.
    - Module B (pointer controller): the current skill is ``plan[pointer]``; the completion gate's
      completion head decides stay/advance (a task-aligned binary "is this subtask done?" signal),
      the back head decides recover. Aliasing is resolved structurally — the controller only
      decides *direction*, never the skill.
    - Module C (replanner): on recover the planner re-routes from structured pre/postcondition
      facts; "back" is just a replan that re-inserts a previous skill (no backward-motion data needed).

Adding a skill needs no System-2 retraining: the planner (frozen LLM) and the text-conditioned gate
both consume the skill *text*; the controller's output space {stay, advance, recover} is fixed.

The control logic is torch-free and unit-testable with a stub gate (see ``__main__`` / test_system2.py).
"""

import logging
from typing import Callable, List, Optional

try:  # torch is only needed for the real gate; keep logic importable without it
    import torch
except Exception:  # noqa: BLE001 - DLL/init failures included
    torch = None

from high_level_model.planning.planner import Planner
from high_level_model.planning.signal_filters import ProgressSignalFilter
from high_level_model.planning.skill_library import SkillLibrary

logger = logging.getLogger(__name__)

__all__ = ["System2Pipeline"]


def _scalar(x) -> float:
    """Coerce a float / 0-d or 1-d tensor / list into a python float (first element)."""
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.reshape(-1)[0].item())
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)

class System2Pipeline:
    """Plan -> pointer controller (gate-driven) -> replanning, with guards."""

    def __init__(
        self,
        skill_library: SkillLibrary,
        planner: Planner,
        completion_gate=None,
        task_instruction: Optional[str] = None,
        protocol_constraints: Optional[Callable[[List[str]], List[str]]] = None,
        tau_done: float = 0.6,
        k_a: int = 2,
        cooldown: int = 10,
        tau_back: float = 0.8,
        k_b: int = 3,
        replan_budget: int = 3,
        back_ema_alpha: float = 0.5,
        completion_median_window: int = 3,
        completion_ema_alpha: float = 0.5,
    ) -> None:
        self.library = skill_library
        self.planner = planner
        self.gate = completion_gate
        self.task_instruction = task_instruction
        self.protocol_constraints = protocol_constraints

        # hyperparameters (paper Table tab:hyperparams)
        self.k_a, self.cooldown_max = k_a, cooldown
        self.tau_done = tau_done
        self.tau_back, self.k_b = tau_back, k_b
        self.replan_budget_max = replan_budget

        # output-side de-jitter of the raw gate signals (built before _reset_runtime_state, which
        # resets their temporal state). Applied before the k-of-k debounce; set ema_alpha=1.0 and
        # median_window=1 to disable and recover the exact Eq. 4-5 behaviour. Back must be able to
        # drop, so no median/clamp — only light EMA over the k_b debouncing. Completion is a "is the
        # subtask done?" probability that spikes near the end; it must be able to drop (no monotone
        # clamp), but a small median kills single-frame spikes before the k_a debounce.
        self._back_filter = ProgressSignalFilter(
            median_window=1, ema_alpha=back_ema_alpha, monotone=False)
        self._comp_filter = ProgressSignalFilter(
            median_window=completion_median_window, ema_alpha=completion_ema_alpha, monotone=False)

        # episode state
        self.plan: List[str] = []
        self.pointer: int = 0
        self.memory: List = []
        # Reserved: optional scene images forwarded to the planner (plan/replan) when the
        # active PromptSet is multimodal. Set via reset(images=...) or set_planner_images().
        self.planner_images = None
        # Optional hook: returns how many work items (samples on the bench, tubes in the rack) are
        # still to be processed. Whoever owns the scene understanding sets it; the pipeline itself
        # stays free of perception. Feeds Planner.replan(repeat_cycles=...) so a recover re-plans
        # the *right amount* of remaining work instead of a single generic cycle.
        self.remaining_cycles_fn: Optional[Callable[[], int]] = None
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._adv_count = 0
        self._back_count = 0
        self._back_latched = False
        self._cooldown = 0
        self.replan_budget = self.replan_budget_max
        self._done = False
        self._back_filter.reset()
        self._comp_filter.reset()

    def set_planner_images(self, images) -> None:
        """Cache the latest scene images for the planner (reserved multimodal hook)."""
        self.planner_images = images

    # ------------------------------------------------------------------ episode
    def reset(self, task_text: str, images=None) -> List[str]:
        self.task_instruction = task_text
        if images is not None:
            self.planner_images = images
        plan = self.planner.plan(task_text, images=self.planner_images)
        self.plan = self._apply_constraints(plan)
        self.pointer = 0
        self.memory = []
        self._reset_runtime_state()
        logger.info("Planned %d steps: %s", len(self.plan), self.plan)
        return self.plan

    def current_skill_id(self) -> str:
        return self.plan[self.pointer]

    def current_skill_text(self) -> str:
        return self.library.instruction_of(self.current_skill_id())

    # ------------------------------------------------------------------ control tick
    def step(self, images=None, back_prob=None, completion_prob=None) -> dict:
        """One control tick. Provide ``images`` (real gate) OR explicit
        ``back_prob``/``completion_prob`` (tests).

        Returns a dict: {action in {stay,advance,recover,done}, skill_id, back_prob,
        completion_prob, replanned}.
        """
        if not self.plan:
            raise RuntimeError("Pipeline not reset; call reset(task_text) first.")

        if back_prob is None or completion_prob is None:
            if self.gate is None:
                raise RuntimeError(
                    "No completion_gate set and no explicit back_prob/completion_prob given.")
            comp_t, back_t = self.gate.predict(images, [self.current_skill_text()])
            completion_prob, back_prob = _scalar(comp_t), _scalar(back_t)
        # de-jitter the raw gate values before any thresholding
        back_prob = self._back_filter.update(float(back_prob))
        completion_prob = self._comp_filter.update(float(completion_prob))

        if self._cooldown > 0:
            self._cooldown -= 1

        skill_id = self.current_skill_id()
        info = {"action": "stay", "skill_id": skill_id,
                "back_prob": back_prob, "completion_prob": completion_prob, "replanned": False}

        # 1) recover (back) — highest priority.
        # Edge-triggered, not level-triggered. A precondition violation (an operator's hand in the
        # workspace, a sample taken off the bench) keeps the back head above tau_back for as long as
        # the scene still shows it — many seconds. Treating that as fresh evidence makes one physical
        # event fire a recover every k_b ticks, each one re-planning the same thing, until the replan
        # budget is gone. So a recover *latches* the signal: it counts as handled, and only re-arms
        # once back has actually fallen back below tau_back **and** the post-switch refractory period
        # (shared with advance, below) has elapsed.
        if self._back_latched:
            if back_prob < self.tau_back and self._cooldown == 0:
                self._back_latched = False
            self._back_count = 0
        else:
            self._back_count = (self._back_count + 1
                                if back_prob >= self.tau_back and self._cooldown == 0 else 0)
        if self._back_count >= self.k_b:
            info["action"] = "recover"
            info["failure_type"] = "precondition_violation"
            info["replanned"] = self._do_recover("precondition_violation")
            self._reset_counts_after_switch()
            self._back_latched = True
            return info

        # 2) advance — driven by the completion head (task-aligned "is this subtask done?")
        if completion_prob >= self.tau_done and self._cooldown == 0:
            self._adv_count += 1
        elif completion_prob < self.tau_done:
            self._adv_count = 0
        if self._adv_count >= self.k_a:
            self.memory.append((skill_id, "done"))
            advanced = self.advance()
            info["action"] = "done" if not advanced else "advance"
            if not advanced:
                self._done = True
            self._reset_counts_after_switch()
            return info

        # 3) stay (default)
        return info

    def _reset_counts_after_switch(self) -> None:
        self._adv_count = 0
        self._back_count = 0
        self._cooldown = self.cooldown_max
        self._back_filter.reset()
        self._comp_filter.reset()

    # ------------------------------------------------------------------ pointer / recover
    def advance(self) -> bool:
        """Advance the pointer; returns False if already at the last step (episode done)."""
        if self.pointer >= len(self.plan) - 1:
            return False
        self.pointer += 1
        return True

    def _do_recover(self, failure_type: str) -> bool:
        """Trigger an LLM replan (this is how 'back' is realized). Returns True if replanned."""
        if self.replan_budget <= 0:
            logger.warning("Replan budget exhausted; staying on current skill (consider safe reset).")
            return False
        completed = self.plan[: self.pointer]
        skill = self.current_skill_id()
        facts = {
            "violated_skill": skill,
            "precondition": self.library.get(skill).precondition_text,
            "postcondition_expected": self.library.get(skill).postcondition_text,
        }
        cycles = 1
        if self.remaining_cycles_fn is not None:
            try:
                cycles = max(1, int(self.remaining_cycles_fn()))
            except Exception as exc:  # noqa: BLE001 - a bad hook must not abort the recovery
                logger.warning("remaining_cycles_fn raised (%s); assuming 1 remaining cycle.", exc)
        remaining = self.planner.replan(
            self.task_instruction, facts=facts, failure_type=failure_type,
            completed_ids=completed, memory=self.memory, images=self.planner_images,
            repeat_cycles=cycles,
        )
        # Concatenate as-is. Filtering out ids already in ``completed`` (the old behaviour) made a
        # second pass over the bench unrepresentable: after one finished cycle every skill is in
        # ``completed``, so ``remaining`` was emptied down to the terminal skill and the pointer
        # declared the episode over while the robot still had samples to process.
        new_plan = completed + list(remaining)
        self.plan = self._apply_constraints(new_plan)
        self.pointer = min(len(completed), max(0, len(self.plan) - 1))
        self.memory.append((skill, f"recover:{failure_type}"))
        self.replan_budget -= 1
        logger.info("Recovered (%s) -> %s (pointer=%d)", failure_type, self.plan, self.pointer)
        return True

    def _apply_constraints(self, plan: List[str]) -> List[str]:
        if self.protocol_constraints is None:
            return plan
        try:
            return self.protocol_constraints(plan)
        except Exception as exc:  # noqa: BLE001
            logger.warning("protocol_constraints raised (%s); keeping unconstrained plan.", exc)
            return plan

    # ------------------------------------------------------------------ manual override
    def jump_to_skill(self, skill_id: str, source: str = "manual") -> dict:
        """Manually move the pointer to *skill_id* (e.g. keyboard override). Not part of the paper
        method — an operational safety affordance for deployment.

        If the skill occurs in the current plan, the pointer jumps to the
        occurrence nearest to the current pointer; otherwise the skill is
        inserted at the current pointer position. Gate counters/cooldown are
        reset so the new skill is judged from fresh evidence, and the episode
        ``done`` flag is cleared (an override can resume a finished episode).

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
        self._reset_counts_after_switch()
        self.memory.append((skill_id, f"override:{source}"))
        logger.info("Manual override (%s) -> skill '%s' (pointer=%d)", source, skill_id, self.pointer)
        return {"action": "override", "skill_id": skill_id, "pointer": self.pointer}

    def is_done(self) -> bool:
        """True only after :meth:`step` has emitted ``done`` for the final skill.

        (The previous ``pointer >= len(plan) - 1`` check returned True as soon
        as the pointer *reached* the last skill, so callers stopped before the
        last skill was ever executed.)
        """
        return self._done


if __name__ == "__main__":
    # Offline smoke test of the control logic with a stub gate (no torch / no model needed).
    logging.basicConfig(level=logging.INFO)
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(here, "..", "..")  # high_level_model/planning -> repo root
    lib = SkillLibrary.from_file(os.path.join(repo_root, "configs", "skill_library", "bloodgas.yaml"))
    planner = Planner(lib, llm_fn=None)  # declaration-order fallback

    pipe = System2Pipeline(lib, planner, tau_done=0.5, k_a=2, cooldown=0, tau_back=0.8, k_b=2)
    pipe.reset("Pick up the green tube, analyze it, and return it to the rack")

    # feed scripted signals: two high-completion ticks per skill -> advance
    n_advances = 0
    for _ in range(100):
        assert not pipe.is_done(), "is_done() must stay False until 'done' is emitted"
        info = pipe.step(back_prob=0.0, completion_prob=0.95)
        if info["action"] == "advance":
            n_advances += 1
        if info["action"] == "done":
            break
    print(f"advances={n_advances}, final pointer={pipe.pointer}, plan_len={len(pipe.plan)}")
    assert n_advances == len(pipe.plan) - 1, "should advance through the whole plan"
    assert pipe.is_done(), "is_done() must be True after 'done' is emitted"

    # manual override can resume a finished episode
    pipe.jump_to_skill(pipe.plan[0], source="test")
    assert not pipe.is_done() and pipe.pointer == 0
    print("Control-logic smoke test OK.")
