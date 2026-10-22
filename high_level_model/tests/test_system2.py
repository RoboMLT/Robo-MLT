"""Tests for System 2 (planner, protocol constraints, monitor network, pipeline, controller).

Run:  python -m high_level_model.tests.test_system2   (from the repo root)

Pure-logic tests need only PyYAML; the monitor-network and controller tests need torch (CPU is
enough) and use a mock SigLIP backbone, so nothing is downloaded. Torch tests skip gracefully.

    T1  planner: declaration-order fallback, protocol/image forwarding, optional live Qwen call
    T2  monitor network: shapes for the three trunks, single-skill broadcast, parameter count
    T3  recovery head overfits a tiny synthetic set
    T4  pipeline: advance / debounce / recover / budget / latch / repeated cycles
    T5  manual override
    T6  protocol constraints
    T7  stall timeout T_max
    T8  ablation variants (no_recovery, timeout_only)
    T9  replan candidates rejected by the protocol, within the replan budget
    T10 recover has priority over advance
    T11 controller cadence (Algorithm 1): K-step sampling, cooldown, switch flag, replanning hold
    T12 recover robustness: timer after a slow replan, replanner exceptions, sorting fallback
    T13 controller: jobs from before a reset are dropped, override never waits for a replan
"""

import logging
import os
from concurrent.futures import Future

logging.basicConfig(level=logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "..", "..", "configs", "skill_library")
LIB_PATH = os.path.join(LIB_DIR, "bloodgas.yaml")
SORT_LIB_PATH = os.path.join(LIB_DIR, "bloodtube_sorting.yaml")
TASK = "Analyse every blood gas sample on the rack."


def _load_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] torch unavailable: {exc}")
        return None


def _library(path=LIB_PATH):
    from high_level_model.planning.skill_library import SkillLibrary
    return SkillLibrary.from_file(path)


def _pipeline(lib=None, planner=None, **kw):
    """Pipeline with a frozen clock (no stall timeout unless a test moves ``now``)."""
    from high_level_model.planning.planner import Planner
    from high_level_model.planning.system2_pipeline import System2Pipeline
    lib = lib or _library()
    kw.setdefault("clock", lambda: 0.0)
    pipe = System2Pipeline(lib, planner or Planner(lib, llm_fn=None), **kw)
    pipe.reset(TASK)
    return pipe


class _ScriptedPlanner:
    """Planner stand-in whose replanner returns pre-scripted candidates."""

    def __init__(self, lib, candidates, on_call=None):
        self.library, self.candidates, self.on_call = lib, list(candidates), on_call
        self.llm_fn = object()            # "has a model": the pipeline may re-query
        self.calls = 0

    def plan(self, task_text, images=None):
        return self.library.ids()

    def replan(self, *args, **kwargs):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        return self.candidates.pop(0) if self.candidates else []


# ----------------------------------------------------------------------- T1
def test_planner() -> bool:
    from high_level_model.planning.planner import Planner

    lib = _library()
    plan = Planner(lib, llm_fn=None).plan(TASK)
    assert plan == lib.ids(), "fallback should return declaration order"
    print(f"   fallback plan OK ({len(plan)} skills)")

    seen = {}

    def fake_llm(prompt, images=None):
        seen["prompt"], seen["images"] = prompt, images
        return '["grasp_routine_tube", "remove_cap", "remove_cap", "unknown_skill", "return_home"]'

    plan = Planner(lib, llm_fn=fake_llm).plan(TASK, images=["I0"])
    assert lib.protocol.text.split()[0] in seen["prompt"], "protocol text must reach the prompt"
    assert seen["images"] == ["I0"], "I_0 must be forwarded to a multimodal backend"
    assert plan == ["grasp_routine_tube", "remove_cap", "return_home"], plan
    print("   protocol + I_0 forwarded, stutter / unknown ids dropped")

    if os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"):
        from high_level_model.planning.llm_backends import qwen_llm_fn
        live = Planner(lib, qwen_llm_fn()).plan(TASK)
        assert live and all(s in lib for s in live), "Qwen plan must be valid skill_ids"
        print(f"   Qwen plan: {live}")
    else:
        print("   [skip] DASHSCOPE_API_KEY not set -> skipping live Qwen call")
    return True


# ----------------------------------------------------------------------- T2/T3 helpers
def _mock_backbone(torch):
    import torch.nn as nn

    class _MockSigLIP:
        def encode_text(self, tokens):
            return torch.randn(tokens.shape[0], 768)

    class _MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_output_dim = 768
            self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.siglip_model = _MockSigLIP()
            self.tokenizer = lambda texts: torch.zeros(len(texts), 8, dtype=torch.long)

        def set_siglip_trainable(self, trainable):
            pass

        def encode_siglip_frames(self, images):
            b, t, cams = images.shape[0], images.shape[1], images.shape[2]
            return torch.randn(b, t, cams, 768), 0

    return _MockBackbone()


# ----------------------------------------------------------------------- T2
def test_gate_shapes() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    from high_level_model.models.completion_gate import CompletionGate, gate_kwargs_from_config

    images = torch.randn(4, 6, 3, 3, 32, 32)
    for fusion in ("cross_attention", "concat", "legacy_cross_attention"):
        gate = CompletionGate(_mock_backbone(torch), fusion=fusion)
        comp, rec = gate(images, ["insert the tube"] * 4)
        assert comp.shape == (4,) and rec.shape == (4,), f"{fusion}: head outputs must be [B]"
        c1, _ = gate(images, ["insert the tube"])
        assert c1.shape == (4,), f"{fusion}: single-skill broadcast must yield [B]"
        print(f"   {fusion:24s} OK, trainable params = {gate.num_trainable_parameters():,}")

    gate = CompletionGate(_mock_backbone(torch))
    assert gate.fusion == "cross_attention" and len(gate.cross_blocks) == 2, "paper trunk is default"
    assert gate.completion_head[1].in_features == 768, "heads must read v_t alone"
    assert gate_kwargs_from_config({})["fusion"] == "concat", "sidecar-less ckpt = oldest trunk"
    assert gate_kwargs_from_config({"use_cross_attention": True})["fusion"] == "legacy_cross_attention"
    print("   default trunk = 2 causal + 2 cross-attention blocks; legacy sidecars map correctly")
    return True


# ----------------------------------------------------------------------- T3
def test_recovery_head_learns() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    import torch.nn as nn

    from high_level_model.models.completion_gate import CompletionGate

    torch.manual_seed(0)
    gate = CompletionGate(_mock_backbone(torch))
    x = torch.randn(64, gate.back_head[0].normalized_shape[0])
    targets = (torch.rand(64) > 0.7).float()
    opt = torch.optim.Adam(gate.back_head.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()
    start = float(bce(gate.back_head(x).squeeze(-1), targets))
    for _ in range(100):
        opt.zero_grad()
        loss = bce(gate.back_head(x).squeeze(-1), targets)
        loss.backward()
        opt.step()
    end = float(bce(gate.back_head(x).squeeze(-1), targets))
    assert end < start, f"recovery BCE should decrease ({start:.4f} -> {end:.4f})"
    print(f"   recovery head learns: BCE {start:.4f} -> {end:.4f}")
    return True


# ----------------------------------------------------------------------- T4
def test_pipeline_logic() -> bool:
    from high_level_model.planning.planner import Planner

    lib = _library()
    planner = Planner(lib, llm_fn=None)

    pipe = _pipeline(lib, planner, tau_done=0.5, k_a=2, tau_rec=0.9, k_b=2)
    advances = 0
    for _ in range(200):
        assert not pipe.is_done(), "is_done() must be False before 'done' is emitted"
        info = pipe.step(back_prob=0.0, completion_prob=0.95)
        advances += info["action"] == "advance"
        if info["action"] == "done":
            break
    assert advances == len(pipe.plan) - 1 and pipe.is_done()
    print(f"   advance path OK ({advances} advances, is_done gating OK)")

    pipe = _pipeline(lib, planner, tau_done=0.5, k_a=2)
    assert pipe.step(back_prob=0.0, completion_prob=0.95)["action"] == "stay"
    print("   k_a debounce OK (single update does not advance)")

    pipe = _pipeline(lib, planner, tau_rec=0.8, k_b=2, replan_budget=2)
    pipe.pointer = 4
    pipe.step(back_prob=0.95, completion_prob=0.0)
    info = pipe.step(back_prob=0.95, completion_prob=0.0)
    assert info["action"] == "recover" and info["replanned"] and info["failure_type"] == "recovery_head"
    assert pipe.replan_budget == 1 and pipe.memory[-1][1] == "recover:recovery_head"
    print(f"   recover path OK (budget {pipe.replan_budget}, memory={pipe.memory[-1]})")

    pipe = _pipeline(lib, planner, tau_rec=0.8, k_b=1, replan_budget=1)
    assert pipe.step(back_prob=0.95, completion_prob=0.0)["replanned"]
    pipe.step(back_prob=0.0, completion_prob=0.0)             # disturbance clears -> latch re-arms
    later = [pipe.step(back_prob=0.95, completion_prob=0.0) for _ in range(6)]
    recovers = [r for r in later if r["action"] == "recover"]
    assert len(recovers) == 1 and not recovers[0]["replanned"], "budget must block the 2nd replan"
    print("   budget guard OK")

    pipe = _pipeline(lib, planner, tau_rec=0.8, k_b=3, cooldown=10, replan_budget=5)
    pipe.pointer = 1
    n1 = sum(pipe.step(back_prob=0.95, completion_prob=0.0)["action"] == "recover" for _ in range(60))
    assert n1 == 1 and pipe.replan_budget == 4, "one sustained disturbance = one recover"
    for _ in range(12):
        pipe.step(back_prob=0.0, completion_prob=0.0)
    n2 = sum(pipe.step(back_prob=0.95, completion_prob=0.0)["action"] == "recover" for _ in range(30))
    assert n2 == 1, "a new disturbance must re-arm the recover"
    print("   recover latch OK (1 recover per sustained event, re-arms after it clears)")

    pipe = _pipeline(lib, planner, tau_rec=0.8, k_b=1, replan_budget=1)
    pipe.pointer = 2
    pipe.remaining_cycles_fn = lambda: 3
    pipe.step(back_prob=0.95, completion_prob=0.0)
    assert pipe.plan[2:] == planner.cycle_ids() * 3 + planner.terminal_ids()
    print(f"   repeated-cycle replan OK ({len(pipe.plan)} steps for 3 remaining samples)")
    return True


# ----------------------------------------------------------------------- T5
def test_manual_override() -> bool:
    pipe = _pipeline(tau_done=0.5, k_a=2, tau_rec=0.9, k_b=2)
    pipe._adv_count = 1
    info = pipe.jump_to_skill(pipe.plan[3], source="keyboard")
    assert info["action"] == "override" and pipe.pointer == 3 and pipe._adv_count == 0
    assert pipe.memory[-1][1] == "override:keyboard"
    print("   mid-plan jump OK")

    for _ in range(200):
        if pipe.step(back_prob=0.0, completion_prob=0.95)["action"] == "done":
            break
    assert pipe.is_done()
    pipe.jump_to_skill(pipe.plan[0], source="keyboard")
    assert not pipe.is_done(), "override must clear the done flag"
    print("   resume-after-done OK")

    pipe.reset(TASK)
    removed = pipe.plan.pop()
    n = len(pipe.plan)
    pipe.jump_to_skill(removed, source="keyboard")
    assert len(pipe.plan) == n + 1 and pipe.current_skill_id() == removed
    print("   insert-missing-skill OK")

    try:
        pipe.jump_to_skill("no_such_skill")
        raise AssertionError("unknown skill_id should raise KeyError")
    except KeyError:
        print("   unknown-skill guard OK")
    return True


# ----------------------------------------------------------------------- T6
def test_protocol_constraints() -> bool:
    from high_level_model.planning.skill_library import SkillLibrary

    lib = _library()
    proto = lib.protocol
    assert proto.is_valid(lib.ids()), proto.violations(lib.ids())
    no_withdraw = [s for s in lib.ids() if s != "withdraw_tube"]
    assert any("discard_sample" in v for v in proto.violations(no_withdraw)), \
        "a tube cannot be discarded before it is withdrawn"
    assert not proto.is_valid(["return_home", "grasp_routine_tube"]), "return_home is terminal"
    two_cycles = lib.ids()[:-1] * 2 + ["return_home"]
    assert proto.is_valid(two_cycles), "repeated assay cycles are admissible"
    print("   blood gas rules OK (per-cycle ordering, terminal return_home)")

    sort_lib = _library(SORT_LIB_PATH)
    ok = ["grasp_routine_blood_tube", "place_to_blue_plate", "grasp_sodium_citrate_tube",
          "place_to_orange_plate", "return_home"]
    bad = ["grasp_routine_blood_tube", "place_to_blue_plate", "place_to_orange_plate", "return_home"]
    assert sort_lib.protocol.is_valid(ok) and not sort_lib.protocol.is_valid(bad)
    print("   tube sorting rules OK (a placement needs a grasp since the last placement)")

    try:
        SkillLibrary.from_dicts([{"skill_id": "a", "canonical_instruction": "a."}],
                                {"rules": [{"skill": "a", "after": ["b"]}]})
        raise AssertionError("rules over unknown skills must be rejected")
    except ValueError:
        print("   unknown-skill rule guard OK")
    return True


# ----------------------------------------------------------------------- T7
def test_stall_timeout() -> bool:
    now = [0.0]
    pipe = _pipeline(t_max_s=30.0, clock=lambda: now[0])
    pipe.pointer = 3
    assert pipe.step(back_prob=0.0, completion_prob=0.0, now=29.0)["action"] == "stay"
    info = pipe.step(back_prob=0.0, completion_prob=0.0, now=31.0)
    assert info["action"] == "recover" and info["failure_type"] == "stall_timeout"
    assert info["replanned"] and pipe.replan_budget == 2
    assert pipe.step(back_prob=0.0, completion_prob=0.0, now=40.0)["action"] == "stay", \
        "the skill timer restarts after a recover"
    print("   T_max fires regardless of beta_t and restarts the per-skill timer")
    return True


# ----------------------------------------------------------------------- T8
def test_variants() -> bool:
    from high_level_model.planning.system2_pipeline import variant_flags

    pipe = _pipeline(**variant_flags("no_recovery"), k_b=1)
    acts = [pipe.step(back_prob=0.99, completion_prob=0.0, now=float(t))["action"]
            for t in range(0, 200, 10)]
    assert "recover" not in acts, "No-Recovery: neither the head nor the timeout may fire"
    assert pipe.step(back_prob=0.99, completion_prob=0.99)["action"] == "stay"
    assert pipe.step(back_prob=0.99, completion_prob=0.99)["action"] == "advance"
    print("   no_recovery OK (pointer advances on completion only)")

    pipe = _pipeline(**variant_flags("timeout_only"), k_b=1)
    assert pipe.step(back_prob=0.99, completion_prob=0.0, now=5.0)["action"] == "stay"
    info = pipe.step(back_prob=0.99, completion_prob=0.0, now=31.0)
    assert info["action"] == "recover" and info["failure_type"] == "stall_timeout"
    print("   timeout_only OK (recovery head ignored, stall timeout kept)")

    try:
        variant_flags("nope")
        raise AssertionError("unknown variant must raise")
    except ValueError:
        pass
    return True


# ----------------------------------------------------------------------- T9
def test_replan_rejection() -> bool:
    lib = _library()
    bad = ["dock_to_analyzer", "insert_to_probe", "discard_sample", "return_home"]
    good = lib.ids()

    planner = _ScriptedPlanner(lib, [bad, good])
    pipe = _pipeline(lib, planner, k_b=1, replan_budget=3)
    pipe.pointer = 1
    info = pipe.step(back_prob=0.95, completion_prob=0.0)
    assert info["replanned"] and planner.calls == 2 and pipe.replan_budget == 1
    assert pipe.plan == lib.ids()[:1] + good and pipe.pointer == 1
    print("   rejected candidate re-queried; each call spends budget")

    planner = _ScriptedPlanner(lib, [bad, bad, bad])
    pipe = _pipeline(lib, planner, k_b=1, replan_budget=3)
    pipe.pointer = 2
    before = list(pipe.plan)
    info = pipe.step(back_prob=0.95, completion_prob=0.0)
    assert not info["replanned"] and pipe.replan_budget == 0 and planner.calls == 3
    assert pipe.plan == before and pipe.pointer == 2, "no admissible plan -> hold the skill"
    print("   budget exhausted by rejections -> pointer holds")
    return True


# ----------------------------------------------------------------------- T10
def test_recover_priority() -> bool:
    pipe = _pipeline(tau_done=0.5, k_a=1, tau_rec=0.5, k_b=1)
    pipe.pointer = 3
    assert pipe.step(back_prob=0.9, completion_prob=0.9)["action"] == "recover"
    print("   both heads fire -> recover wins")
    return True


# ----------------------------------------------------------------------- T11
class _SyncExecutor:
    """Runs submitted jobs immediately (deterministic cadence in tests)."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *args):
        self.submitted += 1
        fut = Future()
        fut.set_result(fn(*args))
        return fut

    def shutdown(self, **_):
        pass


def test_controller_cadence() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    import numpy as np

    from high_level_model.planning.planner import Planner
    from high_level_model.planning.system2_pipeline import System2Pipeline
    from low_level_model.robot.system2_controller import System2Controller

    probs = {"comp": 0.0, "rec": 0.0}

    class _StubGate:
        def predict(self, images, texts):
            assert images.shape[:2] == (1, 6), images.shape
            return torch.tensor([probs["comp"]]), torch.tensor([probs["rec"]])

    lib = _library()
    obs = {"cam_top": np.zeros((24, 32, 3), np.uint8)}

    def make(planner=None):
        pipe = System2Pipeline(lib, planner or Planner(lib, llm_fn=None), completion_gate=_StubGate(),
                               clock=lambda: 0.0, k_a=2, k_b=3)
        ctl = System2Controller(pipe, TASK, ["observation.images.cam_top"], device="cpu",
                                history_len=6, sampling_interval=15, cooldown_steps=10)
        ctl._executor = _SyncExecutor()
        return ctl

    ctl = make()
    for _ in range(300):
        ctl.observe(obs)
    # samples at t = 15, 30, ..., 300; the window is full from the 6th sample on
    assert ctl._executor.submitted == 15, ctl._executor.submitted
    assert ctl.current_task() == lib.instruction_of(lib.ids()[0])
    print("   K-step cadence OK: 15 monitor updates in 300 steps (20 Hz / 15 = 1.33 Hz)")

    probs["comp"] = 0.95
    switches, steps_to_next = 0, None
    for step in range(60):
        n_before = ctl._executor.submitted
        ctl.observe(obs)
        if ctl.consume_switch():
            switches += 1
            switch_step = step
        if switches and steps_to_next is None and ctl._executor.submitted > n_before \
                and step > switch_step:
            steps_to_next = step - switch_step
    assert switches >= 1 and ctl.current_task() != lib.instruction_of(lib.ids()[0])
    assert steps_to_next is not None and steps_to_next > 10, "cooldown must pause sampling"
    print(f"   advance -> switch flag + {ctl.cooldown_steps}-step cooldown OK "
          f"(next update {steps_to_next} steps later)")

    holding = []
    planner = _ScriptedPlanner(lib, [lib.ids()], on_call=lambda: holding.append(ctl2.is_replanning()))
    ctl2 = make(planner)
    probs.update(comp=0.0, rec=0.95)
    for _ in range(200):
        ctl2.observe(obs)
    assert holding == [True] and not ctl2.is_replanning(), holding
    print("   recover -> is_replanning() held during the replanner call")
    return True


# ----------------------------------------------------------------------- T12
def test_recover_robustness() -> bool:
    from high_level_model.planning.planner import Planner
    from high_level_model.planning.system2_pipeline import System2Pipeline

    lib = _library()
    now = [0.0]

    def slow_call():
        now[0] += 31.0                    # the replanner blocks longer than T_max

    pipe = _pipeline(lib, _ScriptedPlanner(lib, [lib.ids()], on_call=slow_call),
                     t_max_s=30.0, clock=lambda: now[0])
    now[0] = 31.0
    assert pipe.step(back_prob=0.0, completion_prob=0.0)["failure_type"] == "stall_timeout"
    assert pipe.step(back_prob=0.0, completion_prob=0.0)["action"] == "stay", \
        "the skill timer must start after the replan, not before it"
    assert pipe.replan_budget == 2
    print("   stall timer restarts after a slow replan")

    def api_down():
        raise RuntimeError("API down")

    pipe = _pipeline(lib, _ScriptedPlanner(lib, [], on_call=api_down), k_b=3, replan_budget=3)
    pipe.pointer = 2
    acts = [pipe.step(back_prob=0.95, completion_prob=0.0)["action"] for _ in range(12)]
    assert acts.count("recover") == 1 and pipe.replan_budget == 2, (acts, pipe.replan_budget)
    assert pipe.pointer == 2 and pipe.memory[-1] == ("discard_cap", "recover:recovery_head:error")
    print("   replanner exception spends one call, latches, holds the skill")

    sort_lib = _library(SORT_LIB_PATH)
    planner = Planner(sort_lib, llm_fn=None)
    for plan in (planner.plan("sort"), planner.replan("sort", repeat_cycles=2)):
        assert sort_lib.protocol.is_valid(plan), (plan, sort_lib.protocol.violations(plan))
    pipe = System2Pipeline(sort_lib, planner, clock=lambda: 0.0)
    pipe.reset("sort")
    assert pipe.plan_admissible
    print("   tube-sorting offline fallback plan and replan are admissible")
    return True


# ----------------------------------------------------------------------- T13
class _DeferredExecutor:
    """Holds submitted jobs until the test runs them (a job in flight across a reset)."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        fut = Future()
        self.jobs.append((fut, fn, args))
        return fut

    def run_all(self):
        for fut, fn, args in self.jobs:
            fut.set_result(fn(*args))
        self.jobs.clear()

    def shutdown(self, **_):
        pass


def test_controller_stale_jobs() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    import threading
    import time

    import numpy as np

    from high_level_model.planning.planner import Planner
    from high_level_model.planning.system2_pipeline import System2Pipeline
    from low_level_model.robot.system2_controller import System2Controller

    class _StubGate:
        def predict(self, images, texts):
            return torch.tensor([0.95]), torch.tensor([0.0])

    lib = _library()
    pipe = System2Pipeline(lib, Planner(lib, llm_fn=None), completion_gate=_StubGate(),
                           clock=lambda: 0.0, k_a=1)
    ctl = System2Controller(pipe, TASK, ["observation.images.cam_top"], device="cpu",
                            history_len=1, sampling_interval=1, cooldown_steps=0)
    ctl._executor = _DeferredExecutor()
    obs = {"cam_top": np.zeros((24, 32, 3), np.uint8)}
    ctl.observe(obs)
    assert len(ctl._executor.jobs) == 1
    ctl.reset(TASK)
    ctl._executor.run_all()
    assert pipe.pointer == 0 and not pipe.memory, "a job from before reset() must not step the plan"
    print("   job launched before reset() is dropped")

    class _Manager:
        def skill_for_digit(self, digit):
            return lib.ids()[digit - 1]

    ctl.observe(obs)                     # re-plans on I_0, launches a fresh job
    ctl._executor.jobs.clear()
    held = threading.Event()

    def hold_lock():
        with ctl._lock:                  # a replan in progress
            held.set()
            time.sleep(0.5)

    worker = threading.Thread(target=hold_lock)
    worker.start()
    held.wait()
    t0 = time.perf_counter()
    ctl.manual_override(3, _Manager())
    waited = time.perf_counter() - t0
    worker.join()
    assert waited < 0.3 and pipe.pointer == 0, (waited, pipe.pointer)
    ctl.manual_override(3, _Manager())
    assert pipe.current_skill_id() == lib.ids()[2]
    print(f"   override during a replan returns after {waited * 1000:.0f} ms; works once free")
    return True


# ----------------------------------------------------------------------- runner
def main():
    tests = [
        ("T1 planner", test_planner),
        ("T2 monitor network", test_gate_shapes),
        ("T3 recovery head learns", test_recovery_head_learns),
        ("T4 pipeline logic", test_pipeline_logic),
        ("T5 manual override", test_manual_override),
        ("T6 protocol constraints", test_protocol_constraints),
        ("T7 stall timeout", test_stall_timeout),
        ("T8 ablation variants", test_variants),
        ("T9 replan rejection", test_replan_rejection),
        ("T10 recover priority", test_recover_priority),
        ("T11 controller cadence", test_controller_cadence),
        ("T12 recover robustness", test_recover_robustness),
        ("T13 controller stale jobs", test_controller_stale_jobs),
    ]
    passed = skipped = failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            result = fn()
            if result is None:
                skipped += 1
                print(f"--- {name}: SKIPPED")
            else:
                passed += 1
                print(f"--- {name}: PASS")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"--- {name}: FAIL ({exc})")
    print(f"\n==== summary: {passed} passed, {skipped} skipped, {failed} failed ====")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
