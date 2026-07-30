"""Tests for System 2 (planner + completion gate + pipeline controller).

Run:  python -m high_level_model.tests.test_system2   (from the repo root)

- T1 Planner: declaration-order fallback + (if DASHSCOPE_API_KEY set) a real Qwen plan.
- T2 CompletionGate shapes: forward returns a back logit and a completion logit (paper Eq. 4-5;
  uses a mock backbone, so no SigLIP/LeRobot download needed; requires torch).
- T3 Back head learns: the back head overfits a tiny synthetic set -> BCE decreases (requires torch).
- T4 Pipeline control logic: completion-driven advance / recover / budget / memory (pure logic, no torch).

Torch-dependent tests (T2/T3) are skipped gracefully if torch is unavailable.
"""

import os
import logging

logging.basicConfig(level=logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
# high_level_model/eval -> repo root -> top-level configs/
LIB_PATH = os.path.join(HERE, "..", "..", "configs", "skill_library", "bloodgas.yaml")
TASK = "Pick up the green tube, analyze it in the blood gas analyzer, and return it to the rack."


def _load_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] torch unavailable: {exc}")
        return None


# ----------------------------------------------------------------------- T1
def test_planner() -> bool:
    from high_level_model.planning.skill_library import SkillLibrary
    from high_level_model.planning.planner import Planner

    lib = SkillLibrary.from_file(LIB_PATH)

    # fallback (offline)
    plan = Planner(lib, llm_fn=None).plan(TASK)
    assert plan == lib.ids(), "fallback should return declaration order"
    print(f"   fallback plan OK ({len(plan)} skills)")

    # Qwen (optional)
    if os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"):
        from high_level_model.planning.llm_backends import qwen_llm_fn
        p = Planner(lib, qwen_llm_fn())
        plan2 = p.plan(TASK)
        assert len(plan2) > 0 and all(s in lib for s in plan2), "Qwen plan must be valid skill_ids"
        print(f"   Qwen plan: {plan2}")
        # replan smoke test with structured facts
        facts = {"violated_skill": plan2[min(2, len(plan2) - 1)], "precondition": "tube in gripper"}
        rem = p.replan(TASK, facts=facts, failure_type="precondition_violation",
                       completed_ids=plan2[:1], memory=[(plan2[0], "done")])
        assert all(s in lib for s in rem), "replan must be valid skill_ids"
        print(f"   Qwen replan: {rem}")
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
            self._dummy = nn.Parameter(torch.zeros(1))
            self.siglip_model = _MockSigLIP()
            self.tokenizer = lambda texts: torch.zeros(len(texts), 8, dtype=torch.long)

        def set_siglip_trainable(self, trainable):
            pass

        def encode_siglip_frames(self, images):
            # [B, T, Cams, C, H, W] -> (feats [B, T, Cams, D], num_pad_frames)
            b, t, cams = images.shape[0], images.shape[1], images.shape[2]
            return torch.randn(b, t, cams, 768), 0

    return _MockBackbone()


# ----------------------------------------------------------------------- T2
def test_gate_shapes() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    from high_level_model.models.completion_gate import CompletionGate

    gate = CompletionGate(_mock_backbone(torch))
    images = torch.randn(4, 6, 3, 3, 224, 224)
    comp_logit, back_logit = gate(images, ["insert the tube"] * 4)
    assert comp_logit.shape == (4,) and back_logit.shape == (4,), "head outputs must be [B]"
    print(f"   shapes OK: completion_logit{tuple(comp_logit.shape)}, "
          f"back_logit{tuple(back_logit.shape)}")
    return True


# ----------------------------------------------------------------------- T2b
def test_gate_shapes_cross_attention() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    from high_level_model.models.completion_gate import CompletionGate

    gate = CompletionGate(_mock_backbone(torch), use_cross_attention=True)
    images = torch.randn(4, 6, 3, 3, 224, 224)
    comp_logit, back_logit = gate(images, ["insert the tube"] * 4)
    assert comp_logit.shape == (4,) and back_logit.shape == (4,), "head outputs must be [B]"
    # broadcast a single skill string over a batch
    c1, _ = gate(images, ["insert the tube"])
    assert c1.shape == (4,), "single-skill broadcast must yield [B]"
    print(f"   cross-attn shapes OK: completion_logit{tuple(comp_logit.shape)}, "
          f"back_logit{tuple(back_logit.shape)}")
    return True


# ----------------------------------------------------------------------- T3
def test_back_head_learns() -> bool:
    """The back head (a small MLP) overfits a tiny synthetic set -> BCE decreases. Exercises the
    head module directly (not the full backbone forward) to keep the test fast and deterministic."""
    torch = _load_torch()
    if torch is None:
        return None
    import torch.nn as nn
    from high_level_model.models.completion_gate import CompletionGate

    torch.manual_seed(0)
    gate = CompletionGate(_mock_backbone(torch))
    n = 64
    in_dim = gate.back_head[0].normalized_shape[0]
    x = torch.randn(n, in_dim)
    targets = (torch.rand(n) > 0.7).float()  # ~30% positives
    opt = torch.optim.Adam(gate.back_head.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()

    def loss_now():
        back_logit = gate.back_head(x).squeeze(-1)
        return bce(back_logit, targets)

    start = float(loss_now().item())
    for _ in range(100):
        opt.zero_grad()
        l = loss_now()
        l.backward()
        opt.step()
    end = float(loss_now().item())
    assert end < start, f"back BCE should decrease ({start:.4f} -> {end:.4f})"
    print(f"   back head learns: BCE {start:.4f} -> {end:.4f}")
    return True


# ----------------------------------------------------------------------- T4
def test_pipeline_logic() -> bool:
    from high_level_model.planning.skill_library import SkillLibrary
    from high_level_model.planning.planner import Planner
    from high_level_model.planning.system2_pipeline import System2Pipeline

    lib = SkillLibrary.from_file(LIB_PATH)
    planner = Planner(lib, llm_fn=None)  # fallback replan returns remaining declaration order

    # (a) advance through the whole plan — driven by the completion head
    pipe = System2Pipeline(lib, planner, tau_done=0.5, k_a=2, cooldown=0, tau_back=0.9, k_b=2)
    pipe.reset(TASK)
    assert not pipe.is_done(), "fresh pipeline must not be done"
    advances = 0
    for _ in range(200):
        # is_done() must stay False until 'done' is emitted — in particular while
        # the pointer is *at* the last skill but it hasn't completed yet
        # (regression test for the old `pointer >= len(plan)-1` off-by-one).
        assert not pipe.is_done(), "is_done() must be False before 'done' is emitted"
        info = pipe.step(back_prob=0.0, completion_prob=0.95)
        advances += info["action"] == "advance"
        if info["action"] == "done":
            break
    assert advances == len(pipe.plan) - 1, f"expected {len(pipe.plan)-1} advances, got {advances}"
    assert pipe.is_done(), "is_done() must be True after 'done' is emitted"
    print(f"   advance path OK ({advances} advances, is_done gating OK)")

    # (a2) advance requires k_a consecutive above-threshold ticks; a single tick must not advance
    pipe = System2Pipeline(lib, planner, tau_done=0.5, k_a=2, cooldown=0, tau_back=0.9, k_b=2)
    pipe.reset(TASK)
    info = pipe.step(back_prob=0.0, completion_prob=0.95)
    assert info["action"] == "stay", "a single above-threshold tick must not advance when k_a=2"
    print("   k_a debounce OK (single tick does not advance)")

    # (b) recover triggers replan, decrements budget, records memory
    pipe = System2Pipeline(lib, planner, tau_back=0.8, k_b=2, replan_budget=2)
    pipe.reset(TASK)
    pipe.pointer = 4  # pretend we're mid-plan
    info1 = pipe.step(back_prob=0.95, completion_prob=0.0)
    info2 = pipe.step(back_prob=0.95, completion_prob=0.0)  # k_b=2 -> triggers on 2nd
    assert info2["action"] == "recover" and info2["replanned"], "should recover+replan"
    assert pipe.replan_budget == 1, "budget should decrement"
    assert any("recover" in str(m[1]) for m in pipe.memory), "memory should record recover"
    print(f"   recover path OK (budget {pipe.replan_budget}, memory={pipe.memory[-1]})")

    # (c) budget exhaustion -> replanned False. The back signal must be allowed to fall between the
    # two events, otherwise the recover latch (below) — not the budget — is what blocks the second
    # one, and this stops testing what it claims to.
    pipe = System2Pipeline(lib, planner, tau_back=0.8, k_b=1, replan_budget=1, cooldown=0)
    pipe.reset(TASK)
    r1 = pipe.step(back_prob=0.95, completion_prob=0.0)
    pipe.step(back_prob=0.0, completion_prob=0.0)          # violation clears -> latch re-arms
    # Several ticks, not one: back is EMA-filtered, so the smoothed value needs a few high samples
    # to climb back over tau_back after that zero.
    later = [pipe.step(back_prob=0.95, completion_prob=0.0) for _ in range(6)]
    recovers = [r for r in later if r["action"] == "recover"]
    assert r1["replanned"], "first recover should replan"
    assert len(recovers) == 1, f"the new violation should recover once, got {len(recovers)}"
    assert not recovers[0]["replanned"], "second recover must be blocked by the exhausted budget"
    print("   budget guard OK")

    # (d) one sustained violation = ONE recover. The back head stays above tau_back for as long as
    # the scene shows the violation; without the latch that re-fires every k_b ticks and burns the
    # whole replan budget on identical re-plans of the same physical event.
    pipe = System2Pipeline(lib, planner, tau_back=0.8, k_b=3, cooldown=10, replan_budget=5)
    pipe.reset(TASK)
    pipe.pointer = 1
    n_recover = sum(pipe.step(back_prob=0.95, completion_prob=0.0)["action"] == "recover"
                    for _ in range(60))
    assert n_recover == 1, f"sustained back must recover exactly once, got {n_recover}"
    assert pipe.replan_budget == 4, f"only one replan should be spent, budget={pipe.replan_budget}"
    # …and a *second*, genuinely new violation must still get through once the signal has dropped.
    for _ in range(12):
        pipe.step(back_prob=0.0, completion_prob=0.0)
    n_recover2 = sum(pipe.step(back_prob=0.95, completion_prob=0.0)["action"] == "recover"
                     for _ in range(30))
    assert n_recover2 == 1, f"a new violation must re-arm the recover, got {n_recover2}"
    print("   recover latch OK (1 recover per sustained event, re-arms after it clears)")

    # (e) a replan must be able to express repeated cycles — one entry per remaining work item.
    pipe = System2Pipeline(lib, planner, tau_back=0.8, k_b=1, replan_budget=1, cooldown=0)
    pipe.reset(TASK)
    pipe.pointer = 2
    pipe.remaining_cycles_fn = lambda: 3
    pipe.step(back_prob=0.95, completion_prob=0.0)
    cycle = planner.cycle_ids()
    tail = pipe.plan[2:]
    assert tail == cycle * 3 + planner.terminal_ids(), (
        f"replan must repeat the cycle once per remaining work item; got {tail}")
    print(f"   repeated-cycle replan OK ({len(pipe.plan)} steps for 3 remaining items)")
    return True


# ----------------------------------------------------------------------- T5
def test_manual_override() -> bool:
    """jump_to_skill: keyboard-override semantics (pure logic, no torch)."""
    from high_level_model.planning.skill_library import SkillLibrary
    from high_level_model.planning.planner import Planner
    from high_level_model.planning.system2_pipeline import System2Pipeline

    lib = SkillLibrary.from_file(LIB_PATH)
    planner = Planner(lib, llm_fn=None)
    pipe = System2Pipeline(lib, planner, tau_done=0.5, k_a=2, cooldown=0, tau_back=0.9, k_b=2)
    pipe.reset(TASK)

    # (a) jump to a mid-plan skill: pointer moves, counters reset, memory records it
    pipe._adv_count = 1  # pretend partial evidence
    info = pipe.jump_to_skill(pipe.plan[3], source="keyboard")
    assert info["action"] == "override" and pipe.pointer == 3, "pointer should jump to index 3"
    assert pipe._adv_count == 0, "gate counters must reset on override"
    assert pipe.memory[-1][1] == "override:keyboard", "memory should record the override source"
    print("   mid-plan jump OK")

    # (b) after done, jumping back to an earlier skill resumes the episode
    for _ in range(200):
        if pipe.step(back_prob=0.0, completion_prob=0.95)["action"] == "done":
            break
    assert pipe.is_done()
    pipe.jump_to_skill(pipe.plan[0], source="keyboard")
    assert not pipe.is_done(), "override must clear the done flag"
    advances = 0
    for _ in range(200):
        info = pipe.step(back_prob=0.0, completion_prob=0.95)
        advances += info["action"] == "advance"
        if info["action"] == "done":
            break
    assert pipe.is_done() and advances == len(pipe.plan) - 1, "must re-run plan to completion"
    print("   resume-after-done OK")

    # (c) a library skill missing from the plan is inserted at the pointer
    pipe.reset(TASK)
    removed = pipe.plan.pop()  # drop the last skill from the plan
    plan_len = len(pipe.plan)
    pipe.jump_to_skill(removed, source="keyboard")
    assert len(pipe.plan) == plan_len + 1 and pipe.current_skill_id() == removed, \
        "missing skill must be inserted at the pointer"
    print("   insert-missing-skill OK")

    # (d) unknown skill -> KeyError
    try:
        pipe.jump_to_skill("no_such_skill")
        raise AssertionError("unknown skill_id should raise KeyError")
    except KeyError:
        print("   unknown-skill guard OK")
    return True


# ----------------------------------------------------------------------- runner
def main():
    tests = [
        # ("T1 planner", test_planner),
        ("T2 gate shapes", test_gate_shapes),
        ("T2b gate shapes (cross-attention)", test_gate_shapes_cross_attention),
        ("T3 back head learns", test_back_head_learns),
        ("T4 pipeline logic", test_pipeline_logic),
        ("T5 manual override", test_manual_override),
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
