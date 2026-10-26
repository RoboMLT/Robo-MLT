"""Tests for the zero-shot VLM monitor baseline of the real-robot tasks (no API calls).

Run:  python -m high_level_model.tests.test_vlm_monitor   (from the repo root)

    V1  monitor prompts load, cover every skill of their library and render without placeholders
    V2  reply parsing
    V3  VLMGate: image order / size, (gamma, beta) mapping, unparsable replies and API errors
    V4  offline replay: answers take effect after the latency, one query in flight, cooldown
    V5  boundary verdicts shared with eval_switch_offline
"""

import logging
import os

logging.basicConfig(level=logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "..", "configs", "skill_library")
TASKS = [("bloodgas.yaml", "bloodgas_monitor.yaml"), ("bloodtube_sorting.yaml", "bloodtube_sorting_monitor.yaml")]


def _load(lib_file, prompt_file):
    from high_level_model.planning.skill_library import SkillLibrary
    from high_level_model.planning.vlm_monitor import MonitorPrompt
    return SkillLibrary.from_file(os.path.join(CFG, lib_file)), MonitorPrompt.from_file(
        os.path.join(CFG, "prompts", prompt_file))


def test_prompts() -> bool:
    for lib_file, prompt_file in TASKS:
        lib, prompt = _load(lib_file, prompt_file)
        prompt.check(lib.ids())
        assert {c.key for c in prompt.cameras} <= {"observation.images.cam_top", "observation.images.cam_left",
                                                   "observation.images.cam_right"}, prompt.cameras
        assert prompt.cameras[0].frame == "first" and all(c.frame == "last" for c in prompt.cameras[1:])
        for sid in lib.ids():
            text = prompt.render(sid, lib.instruction_of(sid), 1.52)
            assert lib.instruction_of(sid) in text and "{" not in text.replace('{"completed"', ""), text
            assert "Image 1: top camera, 1.5 s ago." in text and ".." not in text, text
    print("   both monitor prompts cover their skill libraries")
    return True


def test_parse_reply() -> bool:
    from high_level_model.planning.vlm_monitor import parse_reply
    assert parse_reply('{"completed": true, "feasible": false}') == (True, False)
    assert parse_reply('```json\n{\n"completed": false,\n"feasible": true\n}\n```') == (False, True)
    assert parse_reply('Based on the images provided:\n\n1. **Image 1') is None      # cut-off reasoning
    assert parse_reply('{"completed": "yes", "feasible": true}') is None
    assert parse_reply("") is None and parse_reply(None) is None
    print("   replies parsed")
    return True


def test_gate() -> bool:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] torch unavailable: {exc}")
        return None
    from high_level_model.planning.vlm_monitor import VLMGate
    lib, prompt = _load(*TASKS[0])
    cams = ["observation.images.cam_top", "observation.images.cam_left", "observation.images.cam_right"]
    seen = {}

    def llm_fn(text, images=None):
        seen["text"], seen["images"] = text, images
        return seen.get("reply", '{"completed": true, "feasible": false}')

    gate = VLMGate(llm_fn, prompt, lib, cams, span_s=1.5)
    window = torch.zeros(1, 3, len(cams), 3, 480, 640)
    window[0, 0, 0] = 0.25                                  # first top frame
    window[0, -1, 0] = 0.5                                  # last top frame
    window[0, -1, 2] = 1.0                                  # last cap-arm wrist frame
    comp, back = gate.predict(window, ["remove the sample tube cap."])
    assert comp.tolist() == [1.0] and back.tolist() == [1.0]
    imgs = seen["images"]
    assert len(imgs) == 4 and all(i.shape == (336, 448, 3) for i in imgs), [i.shape for i in imgs]
    assert [int(i[0, 0, 0]) for i in imgs] == [64, 128, 0, 255]
    assert '"remove the sample tube cap."' in seen["text"] and "cap arm" in seen["text"]
    seen["reply"] = "The tube"                              # unparsable -> no decision
    assert gate.query(window[0, 0], window[0, -1], "return the robotic arm to the home position.") == (0.0, 0.0)

    def broken(text, images=None):
        raise TimeoutError("api down")

    gate2 = VLMGate(broken, prompt, lib, cams, span_s=1.5)
    assert gate2.query(window[0, 0], window[0, -1], "remove the sample tube cap.") == (0.0, 0.0)
    assert gate2.calls[-1]["error"] and gate.summary()["unparsed"] == 1 and gate.summary()["calls"] == 2
    try:
        VLMGate(llm_fn, prompt, lib, cams[:2], span_s=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("a camera missing from the frames must be refused")
    print("   gate: 4 images in prompt order at 448 px, (gamma, beta) mapped, failures stay")
    return True


def test_replay() -> bool:
    try:
        from high_level_model.eval.eval_vlm_monitor_offline import replay
        from high_level_model.eval.gate_metrics import build_eval_pipeline
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] replay needs torch / lerobot: {exc}")
        return None
    lib, _ = _load(*TASKS[1])
    plan = ["grasp_routine_blood_tube", "place_to_orange_plate", "return_home"]
    done_from = {lib.instruction_of(plan[0]): 100, lib.instruction_of(plan[1]): 300, lib.instruction_of(plan[2]): 450}

    def run(latency):
        pipe = build_eval_pipeline(lib, plan, tau_done=0.6, k_a=2, cooldown=0, clock=lambda: 0.0)
        asked = []

        def ask(f, text):
            asked.append(f)
            return float(f >= done_from[text]), 0.0

        return replay(pipe, ask, first_tick=38, n=600, tick=19, cooldown=12, latency_frames=lambda: latency), asked

    (switches, done_at, trace), asked = run(0)
    # k_a = 2: the second positive tick after the skill is done advances.
    assert switches == [133, 323] and done_at == 475, (switches, done_at)
    assert all(b - a == 19 for a, b in zip(asked, asked[1:]) if b < 133), asked
    (switches, done_at, trace), asked = run(50)
    # Every answer arrives 50 frames late and the next query waits for it, on the 19-frame grid.
    assert all(t["ready"] - t["query"] == 50 for t in trace)
    assert all(b >= a + 50 and (b - 38) % 19 == 0 for a, b in zip(asked, asked[1:])), asked
    assert switches and switches[0] > 133, switches
    print(f"   replay: instant switches {[133, 323]}, with 50-frame latency {switches} (done {done_at})")
    return True


def test_verdicts() -> bool:
    from high_level_model.eval.gate_metrics import switch_verdicts
    out = switch_verdicts([40, 200, 330], [100, 210, 300, 500], early=50, late=20)
    assert out == [("premature", -60), ("on_time", -10), ("late", 30), ("missed", None)], out
    print("   verdicts: premature / on_time / late / missed")
    return True


def main():
    results = [test_prompts(), test_parse_reply(), test_gate(), test_replay(), test_verdicts()]
    failed = sum(r is False for r in results)
    print(f"\n==== summary: {sum(r is True for r in results)} passed, "
          f"{sum(r is None for r in results)} skipped, {failed} failed ====")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
