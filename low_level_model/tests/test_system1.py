"""Tests for System 1 (Generative Executor / PI0 Flow-Matching policy).

Run:  python -m low_level_model.tests.test_system1   (from the repo root)

- T1 Async streamer logic: chunk-index / prefetch / switch / future-state-aware
  swap of System1AsyncStreamer (pure logic, no torch needed).
- T2 Flow-matching shapes (tiny real model): instantiate the REAL PI0Model on a
  downscaled config (hidden ~32, 2 layers) on CPU; assert training forward returns
  a per-element MSE loss [B, T, A] and sample_actions returns [B, chunk, A].
- T3 Loss decreases: overfit the tiny PI0Model on a fixed synthetic batch; the
  flow-matching MSE must drop.
- T4 Processor smoke: make_pi0_pre_post_processors builds (skipped if the
  PaliGemma tokenizer cannot be downloaded / loaded).

Torch / transformers / lerobot-dependent tests (T2-T4) skip gracefully if their
imports or a network-dependent step are unavailable, like test_system2.py.
"""

import numpy as np


def _load_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] torch unavailable: {exc}")
        return None


# ----------------------------------------------------------------------- T2/T3 helpers
def _tiny_config():
    """A downscaled PI0Config that instantiates the real model cheaply on CPU."""
    from low_level_model.models.pi0.configuration_pi0 import PI0Config

    cfg = PI0Config(
        dtype="bfloat16",
        chunk_size=50,
        n_action_steps=30,
        max_state_dim=7,
        max_action_dim=7,
        num_inference_steps=10,
        fuse_qkv=False,
        fuse_gate_up=False,
        compile_model=False,
        device="cpu",
    )
    # Shrink the PaliGemma (SigLIP + Gemma) backbone.
    v = cfg.vlm_config
    t = v.text_config
    t.hidden_size = 32
    t.intermediate_size = 64
    t.num_attention_heads = 2
    t.head_dim = 16
    t.num_hidden_layers = 2
    t.num_key_value_heads = 1
    t.vocab_size = 64
    v._vocab_size = 64
    v.vocab_size = 64
    v.image_token_index = 0
    vc = v.vision_config
    vc.hidden_size = 32
    vc.intermediate_size = 64
    vc.num_hidden_layers = 2
    vc.num_attention_heads = 2
    vc.patch_size = 112        # 224 / 112 -> 4 patches
    vc.image_size = 224
    vc.projection_dim = 32     # must match text hidden_size
    vc.num_channels = 3
    # Shrink the Gemma action expert (head_dim must match the text head_dim).
    ae = cfg.action_expert_config
    ae.hidden_size = 32
    ae.intermediate_size = 64
    ae.num_attention_heads = 2
    ae.head_dim = 16
    ae.num_hidden_layers = 2
    ae.num_key_value_heads = 1
    ae.vocab_size = 64
    return cfg


def _tiny_inputs(torch, cfg, bsz=2, text_len=4):
    images = [torch.rand(bsz, 3, 224, 224)]
    img_masks = [torch.ones(bsz, dtype=torch.bool)]
    tokens = torch.randint(0, cfg.vlm_config.text_config.vocab_size, (bsz, text_len))
    masks = torch.ones(bsz, text_len, dtype=torch.bool)
    state = torch.randn(bsz, cfg.max_state_dim)
    actions = torch.randn(bsz, cfg.chunk_size, cfg.max_action_dim)
    return images, img_masks, tokens, masks, state, actions


# ----------------------------------------------------------------------- T1
def test_async_streamer() -> bool:
    from low_level_model.runtime.inference_system1 import System1AsyncStreamer

    n_steps = 4
    calls = []

    def infer_fn(obs, task, future_state):
        cid = len(calls)
        calls.append((obs, task, None if future_state is None else np.asarray(future_state).copy()))
        # Row i of chunk cid is [cid, i] so we can trace provenance.
        return np.stack([np.array([cid, i], dtype=float) for i in range(n_steps)])

    streamer = System1AsyncStreamer(infer_fn, n_action_steps=n_steps, overlap_steps=1,
                                    future_state_aware=True)

    actions = [streamer.get_action({"obs": k}, "pick") for k in range(2 * n_steps)]
    streamer.shutdown()

    # First chunk (cid 0) executed for steps 0..3, second chunk (cid 1) for 4..7.
    for i in range(n_steps):
        assert actions[i][0] == 0 and actions[i][1] == i, f"step {i} should come from chunk 0 row {i}"
    for i in range(n_steps):
        assert actions[n_steps + i][0] == 1 and actions[n_steps + i][1] == i, f"step {n_steps+i} should come from chunk 1"
    # bootstrap (call 0) + a prefetch at each chunk's overlap boundary (steps 3 & 7) = 3 inferences.
    assert len(calls) == 3, f"expected 3 inferences (1 bootstrap + 2 boundary prefetch) over 8 steps, got {len(calls)}"

    # Future-state awareness: each prefetch is launched with the last action of the
    # then-current chunk (chunk 0 -> [0, 3]; chunk 1 -> [1, 3]) as its future_state.
    assert calls[0][2] is None, "bootstrap inference has no future_state"
    assert np.array_equal(calls[1][2], np.array([0.0, float(n_steps - 1)])), \
        f"1st prefetch should pass last action of chunk 0 as future_state, got {calls[1][2]}"
    assert np.array_equal(calls[2][2], np.array([1.0, float(n_steps - 1)])), \
        f"2nd prefetch should pass last action of chunk 1 as future_state, got {calls[2][2]}"
    print(f"   streamer OK: 2 chunks consumed over 8 steps, prefetch future_states="
          f"{[calls[1][2].tolist(), calls[2][2].tolist()]}")

    # Future-state disabled -> always None.
    calls.clear()
    s2 = System1AsyncStreamer(infer_fn, n_action_steps=n_steps, overlap_steps=0, future_state_aware=False)
    for k in range(n_steps + 1):
        s2.get_action({"obs": k}, "pick")
    s2.shutdown()
    assert all(c[2] is None for c in calls), "future_state must be None when disabled"

    # reset() drops in-flight state.
    s3 = System1AsyncStreamer(infer_fn, n_action_steps=n_steps)
    s3.get_action({"obs": 0}, "pick")
    s3.reset()
    assert s3.current_chunk is None and s3.chunk_index == 0 and not s3.is_running(), "reset must clear state"
    s3.shutdown()
    print("   future-state toggle + reset OK")
    return True


# ----------------------------------------------------------------------- T1b
def test_chunk_blender() -> bool:
    """TemporalChunkBlender: steady-state identity, boundary continuity, reactivity,
    and stale-prefix drop (χ₀ temporal chunk-wise smoothing). Pure numpy, no torch."""
    from low_level_model.runtime.action_smoothing import TemporalChunkBlender, interpolate_action

    # No lag in steady state: a constant stream passes through unchanged.
    b = TemporalChunkBlender(latency_k=0, min_smooth_steps=3)
    old = np.ones((3, 2)); new = np.ones((5, 2))
    out = b.merge(old, new, consumed=0, last_action=np.ones(2))
    assert np.allclose(out, 1.0), "constant input must be identity (no smoothing lag)"

    # Dtype preservation: a float32 chunk must NOT be upcast to float64, else the
    # merged buffer flows back as a float64 observation.state and the torch.compiled
    # (dynamic=False) policy recompiles every step -> looks like a hang.
    f32 = np.ones((5, 2), dtype=np.float32)
    out32 = b.merge(f32[:3], f32, consumed=0, last_action=np.ones(2, np.float32))
    assert out32.dtype == np.float32, f"blend must preserve float32, got {out32.dtype}"
    boot = b.merge(None, f32, consumed=0, last_action=None)  # bootstrap return path
    assert boot.dtype == np.float32, f"bootstrap return must preserve float32, got {boot.dtype}"

    # Continuity: blended boundary delta is strictly smaller than the hard switch.
    old = np.zeros((3, 1)); new = np.full((3, 1), 10.0)
    out = b.merge(old, new, consumed=0, last_action=np.zeros(1))
    blended_max_delta = float(np.max(np.abs(np.diff(out[:, 0]))))
    assert blended_max_delta < 10.0, f"blend should reduce the 10.0 boundary jump, got {blended_max_delta}"
    # Reactivity: a genuine step change still fully reaches the target within the window.
    assert np.isclose(out[-1, 0], 10.0), f"new target must be reached, got {out[-1,0]}"

    # Drop alignment: latency_k discards the stale leading steps.
    b2 = TemporalChunkBlender(latency_k=4, min_smooth_steps=2)
    new = np.arange(8, dtype=float).reshape(8, 1)
    out = b2.merge(None, new, consumed=4, last_action=np.array([100.0]))
    assert out.shape[0] == 4, f"expected 8-4 stale = 4 rows, got {out.shape[0]}"
    assert np.isclose(out[-1, 0], 7.0) and np.isclose(out[-2, 0], 6.0), "tail must be the un-dropped new steps"
    assert not np.any(np.isclose(out[:, 0], 0.0)), "stale leading steps (0..3) must be dropped"

    # smoothstep profile builds; invalid profile rejected.
    sm = TemporalChunkBlender(min_smooth_steps=3, weight_profile="smoothstep")
    _ = sm.merge(np.zeros((3, 1)), np.ones((3, 1)), consumed=0, last_action=np.zeros(1))
    try:
        TemporalChunkBlender(weight_profile="bogus")
        assert False, "invalid weight_profile must raise"
    except ValueError:
        pass

    # interpolate_action velocity clamp: a big jump is sub-divided, a small one is not.
    sub = interpolate_action(np.zeros(1), np.array([10.0]), max_step=2.0)
    assert len(sub) == 5 and np.isclose(sub[-1, 0], 10.0), f"expected 5 bounded sub-steps, got {len(sub)}"
    assert interpolate_action(np.zeros(1), np.array([1.0]), max_step=2.0).shape[0] == 1, "small jump = passthrough"
    print(f"   blender OK: boundary delta {blended_max_delta} < 10, drop+reactivity+clamp verified")
    return True


# ----------------------------------------------------------------------- T1c
def test_streamer_with_blender() -> bool:
    """End-to-end smoke: the async streamer's blender path runs, keeps producing
    one action/step, and triggers more than the bootstrap inference."""
    from low_level_model.runtime.action_smoothing import TemporalChunkBlender
    from low_level_model.runtime.inference_system1 import System1AsyncStreamer

    n_steps = 4
    calls = []

    def infer_fn(obs, task, future_state):
        cid = len(calls)
        calls.append(cid)
        return np.stack([np.array([cid, i], dtype=float) for i in range(n_steps)])

    blender = TemporalChunkBlender(latency_k=2, min_smooth_steps=3,weight_profile='smoothstep')
    s = System1AsyncStreamer(infer_fn, n_action_steps=n_steps, overlap_steps=2,
                             future_state_aware=True, blender=blender)
    actions = [s.get_action({"obs": k}, "pick") for k in range(12)]
    s.shutdown()

    assert len(actions) == 12, "must emit one action per step"
    assert all(np.asarray(a).shape == (2,) for a in actions), "each action is a 2-vector"
    assert len(calls) >= 2, f"blended streamer must infer beyond bootstrap, got {len(calls)}"

    # reset clears the blender seed and buffer.
    s2 = System1AsyncStreamer(infer_fn, n_action_steps=n_steps, blender=TemporalChunkBlender(weight_profile='smoothstep'))
    s2.get_action({"obs": 0}, "pick")
    s2.reset()
    assert s2.current_chunk is None and s2.chunk_index == 0 and s2._last_action is None, "reset must clear state"
    s2.shutdown()
    print(f"   streamer+blender OK: 12 steps over {len(calls)} inferences")
    return True


# ----------------------------------------------------------------------- T2
def test_flow_matching_shapes() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.modeling_pi0 import PI0Model
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import PI0Model (transformers/lerobot?): {exc}")
        return None

    torch.manual_seed(0)
    cfg = _tiny_config()
    model = PI0Model(cfg).eval()
    images, img_masks, tokens, masks, state, actions = _tiny_inputs(torch, cfg)

    # Training forward: per-element flow-matching MSE loss.
    loss = model.forward(images, img_masks, tokens, masks, state, actions)
    assert tuple(loss.shape) == (2, cfg.chunk_size, cfg.max_action_dim), \
        f"loss shape {tuple(loss.shape)} != [B, T, A]"
    assert float(loss.mean()) >= 0.0, "MSE loss must be non-negative"
    print(f"   forward loss shape {tuple(loss.shape)}, mean={float(loss.mean()):.4f}")

    # Inference: Euler integration after fusing Q/K/V + gate/up (inference path).
    sampler = PI0Model(_tiny_config()).eval()
    sampler.init_qkv_fusion_from_existing()
    sampler.init_mlp_fusion_from_existing()
    with torch.no_grad():
        chunk = sampler.sample_actions(images, img_masks, tokens, masks, state, num_steps=2)
    assert tuple(chunk.shape) == (2, cfg.chunk_size, cfg.max_action_dim), \
        f"sampled chunk {tuple(chunk.shape)} != [B, chunk, A]"
    print(f"   sample_actions shape {tuple(chunk.shape)}")
    return True


# ----------------------------------------------------------------------- T3
def test_loss_decreases() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.modeling_pi0 import PI0Model
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import PI0Model: {exc}")
        return None

    torch.manual_seed(1)
    cfg = _tiny_config()
    cfg.dtype = "float32"  # float32 for clean optimisation signal
    model = PI0Model(cfg).train()
    images, img_masks, tokens, masks, state, actions = _tiny_inputs(torch, cfg)

    noise = torch.randn(2, cfg.chunk_size, cfg.max_action_dim)
    time = torch.rand(2) * 0.8 + 0.1
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_now():
        return model.forward(images, img_masks, tokens, masks, state, actions,
                             noise=noise, time=time).mean()

    start = float(loss_now().item())
    for _ in range(40):
        opt.zero_grad()
        l = loss_now()
        l.backward()
        opt.step()
    end = float(loss_now().item())
    assert end < start, f"flow-matching MSE should decrease ({start:.4f} -> {end:.4f})"
    print(f"   overfit: MSE {start:.4f} -> {end:.4f}")
    return True


# ----------------------------------------------------------------------- T4
def test_processor_smoke() -> bool:
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.processor_pi0 import make_pi0_pre_post_processors
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import processor: {exc}")
        return None

    cfg = _tiny_config()
    cfg.validate_features()
    try:
        pre, post = make_pi0_pre_post_processors(cfg, dataset_stats=None)
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] processor build needs the PaliGemma tokenizer (offline?): {exc}")
        return None
    assert pre is not None and post is not None, "processors must build"
    print("   pre/post processors built OK")
    return True


# ----------------------------------------------------------------------- T5
class _FakeLatencyTracker:
    """Minimal stand-in for lerobot's LatencyTracker (max/add/reset), so the RTC
    streamer logic is testable without importing lerobot/torch."""

    def __init__(self, fixed_max=0.0):
        self._max = fixed_max
        self.added = []

    def add(self, v):
        self.added.append(v)
        self._max = max(self._max, v)

    def max(self):
        return self._max

    def reset(self):
        self.added.clear()
        self._max = 0.0


def test_rtc_streamer_logic() -> bool:
    """RTC path of System1AsyncStreamer (numpy-only): bootstrap is unguided, later
    inferences receive the normalised leftover as the RTC prefix, and the stale
    prefix is dropped on arrival."""
    from low_level_model.runtime.inference_system1 import System1AsyncStreamer

    n_steps = 6
    calls = []  # (rtc_kwargs) per inference

    def rtc_infer_fn(obs, task, rtc_kwargs):
        cid = len(calls)
        calls.append(rtc_kwargs)
        # denorm rows [cid, i]; model-space rows offset by +100 so we can tell which
        # space the leftover snapshot came from.
        denorm = np.stack([np.array([cid, i], dtype=float) for i in range(n_steps)])
        model_space = np.stack([np.array([cid + 100, i], dtype=float) for i in range(n_steps)])
        return denorm, model_space

    tracker = _FakeLatencyTracker(fixed_max=0.05)  # 0.05s * 20fps -> d ~= 1
    s = System1AsyncStreamer(
        rtc_infer_fn=rtc_infer_fn, infer_fn=None, n_action_steps=n_steps, overlap_steps=2,
        latency_tracker=tracker, fps=20.0, rtc_execution_horizon=4, rtc_min_delay=1,
    )
    actions = [s.get_action({"obs": k}, "pick") for k in range(2 * n_steps)]
    s.shutdown()

    assert len(actions) == 2 * n_steps, "must emit one action per step"
    assert all(np.asarray(a).shape == (2,) for a in actions), "each action is a 2-vector"
    assert len(calls) >= 2, f"RTC streamer must infer beyond bootstrap, got {len(calls)}"

    # Bootstrap inference is unguided (no predecessor).
    assert calls[0] is None, "bootstrap RTC inference must have rtc_kwargs=None"

    # At least one later inference carries a real RTC prefix, and it comes from the
    # *model-space* buffer (rows offset by +100), not the denormalised one.
    guided = [c for c in calls[1:] if c is not None]
    assert guided, "expected at least one guided (prefetch) inference"
    kw = guided[0]
    assert set(kw) == {"prev_chunk_left_over", "inference_delay", "execution_horizon"}, \
        f"unexpected RTC kwargs keys: {sorted(kw)}"
    prefix = np.asarray(kw["prev_chunk_left_over"])
    assert prefix.ndim == 2 and prefix.shape[1] == 2, f"prefix shape {prefix.shape}"
    assert np.all(prefix[:, 0] >= 100), "prefix must be the NORMALISED (model-space) leftover"
    assert 1 <= kw["inference_delay"] <= kw["execution_horizon"], \
        f"d ({kw['inference_delay']}) must be in [1, s={kw['execution_horizon']}]"
    assert kw["execution_horizon"] <= 4, "s must be clamped to rtc_execution_horizon"
    assert tracker.added, "latency must be recorded for the delay estimator"

    # Guards.
    from low_level_model.runtime.action_smoothing import TemporalChunkBlender
    try:
        System1AsyncStreamer(rtc_infer_fn=rtc_infer_fn, infer_fn=None, n_action_steps=n_steps,
                             latency_tracker=tracker, fps=20.0, blender=TemporalChunkBlender())
        raise AssertionError("RTC + blender must be rejected")
    except ValueError:
        pass
    try:
        System1AsyncStreamer(rtc_infer_fn=rtc_infer_fn, infer_fn=None, n_action_steps=n_steps)
        raise AssertionError("RTC without latency_tracker/fps must be rejected")
    except ValueError:
        pass

    # reset clears RTC state.
    s2 = System1AsyncStreamer(rtc_infer_fn=rtc_infer_fn, infer_fn=None, n_action_steps=n_steps,
                              latency_tracker=_FakeLatencyTracker(), fps=20.0)
    s2.get_action({"obs": 0}, "pick")
    s2.reset()
    assert s2.current_chunk is None and s2._model_space_chunk is None and s2.chunk_index == 0, \
        "reset must clear RTC state"
    s2.shutdown()
    print(f"   RTC streamer OK: {2*n_steps} steps over {len(calls)} inferences, "
          f"prefix from model space, d in [1,s]")
    return True


# ----------------------------------------------------------------------- T6
def test_rtc_guidance_pi0() -> bool:
    """RTC guidance on the real (tiny) PI0Model: passthrough when no prefix, and the
    frozen prefix region is pulled toward the target when a prefix is supplied."""
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.modeling_pi0 import PI0Model
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.rtc.modeling_rtc import RTCProcessor
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import PI0Model / lerobot RTC: {exc}")
        return None

    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg.dtype = "float32"  # clean grad signal for ΠGDM guidance
    model = PI0Model(cfg).eval()
    images, img_masks, tokens, masks, state, _ = _tiny_inputs(torch, cfg)

    B, T, A = 2, cfg.chunk_size, cfg.max_action_dim
    noise = torch.randn(B, T, A)

    # Unguided baseline (rtc off / no prefix), fixed noise for determinism.
    with torch.no_grad():
        base = model.sample_actions(images, img_masks, tokens, masks, state, noise=noise, num_steps=2)

    # (a) Passthrough: attach an enabled processor but pass prev_chunk_left_over=None.
    model.rtc_processor = RTCProcessor(RTCConfig(enabled=True))
    with torch.no_grad():
        same = model.sample_actions(images, img_masks, tokens, masks, state, noise=noise,
                                    num_steps=2, prev_chunk_left_over=None)
    assert torch.allclose(base, same, atol=1e-5), "RTC with no prefix must equal the unguided sample"

    # (b) Frozen-prefix pull: hard mask (ZEROS) over the first d steps toward a target.
    d = 2
    target = torch.full((B, T, A), 3.0)  # distinctive target the prefix should pull toward
    model.rtc_processor = RTCProcessor(
        RTCConfig(enabled=True, prefix_attention_schedule=RTCAttentionSchedule.EXP,
                  max_guidance_weight=10.0, execution_horizon=d)
    )
    guided = model.sample_actions(
        images, img_masks, tokens, masks, state, noise=noise, num_steps=2,
        prev_chunk_left_over=target, inference_delay=d, execution_horizon=d,
    )
    err_guided = (guided[:, :d] - target[:, :d]).abs().mean().item()
    err_base = (base[:, :d] - target[:, :d]).abs().mean().item()
    assert err_guided < err_base, \
        f"RTC guidance must pull the frozen prefix toward the target ({err_guided:.4f} !< {err_base:.4f})"
    print(f"   RTC guidance OK: passthrough exact; frozen-prefix err {err_base:.4f} -> {err_guided:.4f}")
    return True


# ----------------------------------------------------------------------- T7
def test_rtc_compile_retarget() -> bool:
    """When compile is on, PI0 compiles the full sampler by default, but switches to
    the per-step denoise_step once an RTC processor is attached (so autograd through
    the ΠGDM guidance doesn't graph-break the sampler). torch.compile is lazy, so this
    only checks the wrapper retargeting — no actual kernel compilation runs."""
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.modeling_pi0 import PI0Model
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.rtc.modeling_rtc import RTCProcessor
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import PI0Model / lerobot RTC: {exc}")
        return None

    cfg = _tiny_config()
    cfg.compile_model = True
    cfg.compile_mode = "default"
    model = PI0Model(cfg).eval()

    # No RTC -> the whole sampler is the compile target.
    assert model.sample_actions is not model._uncompiled_sample_actions, "sampler must be compiled"
    assert model.denoise_step is model._uncompiled_denoise_step, "denoise_step must stay uncompiled"

    # Attach RTC + retarget -> per-step denoise_step is compiled, sampler reverts to eager.
    model.rtc_processor = RTCProcessor(RTCConfig(enabled=True))
    model._apply_compile()
    assert model.denoise_step is not model._uncompiled_denoise_step, "denoise_step must be compiled under RTC"
    assert model.sample_actions is model._uncompiled_sample_actions, "sampler must revert to eager under RTC"

    # Idempotent + reverts cleanly when RTC is turned back off.
    model.rtc_processor = None
    model._apply_compile()
    assert model.sample_actions is not model._uncompiled_sample_actions, "sampler recompiles when RTC off"
    assert model.denoise_step is model._uncompiled_denoise_step, "denoise_step reverts when RTC off"
    print("   RTC compile retarget OK: sampler <-> denoise_step switch is clean and idempotent")
    return True


# ----------------------------------------------------------------------- T8
def test_ttrtc_suffix_embedder_per_token() -> bool:
    """PI0SuffixEmbedder accepts a per-sample scalar time [B] and a per-token time
    [B, T]; a per-token time that is a constant broadcast of the scalar must produce
    an identical suffix embedding (the TTRTC generalisation is a superset)."""
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.modeling_pi0 import PI0Model
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import PI0Model: {exc}")
        return None

    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg.dtype = "float32"
    emb = PI0Model(cfg).eval().suffix_embedder

    B, T, A = 2, cfg.chunk_size, cfg.max_action_dim
    state = torch.randn(B, cfg.max_state_dim)
    x = torch.randn(B, T, A)
    t_scalar = torch.rand(B) * 0.8 + 0.1

    out_scalar = emb(state, x, t_scalar)[0]
    out_pertoken = emb(state, x, t_scalar[:, None].expand(B, T).contiguous())[0]
    assert out_scalar.shape == out_pertoken.shape == (B, 1 + T, cfg.action_expert_config.hidden_size), \
        f"suffix embedding shape mismatch: {tuple(out_scalar.shape)}"
    assert torch.allclose(out_scalar, out_pertoken, atol=1e-5), \
        "constant per-token time must match the scalar-time embedding"
    print(f"   suffix embedder per-token OK: shapes {tuple(out_scalar.shape)}, scalar==broadcast")
    return True


# ----------------------------------------------------------------------- T9
def test_ttrtc_forward() -> bool:
    """forward_ttrtc returns (per-element MSE [B,T,A], prefix_mask [B,T]); with
    ttrtc_max_delay=0 the prefix is empty so the postfix loss equals the baseline
    flow-matching forward (same noise/time). policy-level forward_ttrtc returns a
    scalar loss."""
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.modeling_pi0 import PI0Model
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import PI0Model: {exc}")
        return None

    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg.dtype = "float32"
    cfg.ttrtc = True
    cfg.ttrtc_max_delay = 0  # force d=0 → empty prefix → equivalent to baseline forward
    model = PI0Model(cfg).eval()
    images, img_masks, tokens, masks, state, actions = _tiny_inputs(torch, cfg)

    B, T, A = actions.shape
    noise = torch.randn(B, T, A)
    time = torch.rand(B) * 0.8 + 0.1

    losses_ttrtc, prefix_mask = model.forward_ttrtc(
        images, img_masks, tokens, masks, state, actions, noise=noise, time=time
    )
    assert tuple(losses_ttrtc.shape) == (B, T, A), f"ttrtc loss shape {tuple(losses_ttrtc.shape)}"
    assert tuple(prefix_mask.shape) == (B, T), f"prefix_mask shape {tuple(prefix_mask.shape)}"
    assert not bool(prefix_mask.any()), "ttrtc_max_delay=0 must yield an empty prefix"

    losses_base = model.forward(images, img_masks, tokens, masks, state, actions, noise=noise, time=time)
    assert torch.allclose(losses_ttrtc, losses_base, atol=1e-5), \
        "with an empty prefix TTRTC loss must equal the baseline flow-matching loss"

    # Non-empty prefix: some tokens become clean (t=0) and are excluded from the loss.
    cfg.ttrtc_max_delay = min(4, T)
    torch.manual_seed(3)
    _, prefix_mask2 = model.forward_ttrtc(images, img_masks, tokens, masks, state, actions,
                                          noise=noise, time=time)
    assert prefix_mask2.shape == (B, T)
    # Prefix must be a contiguous leading run per sample (arange < d).
    for row in prefix_mask2:
        d = int(row.sum())
        assert bool(row[:d].all()) and not bool(row[d:].any()), "prefix must be the leading d tokens"
    print(f"   forward_ttrtc OK: empty-prefix == baseline; prefix is leading run (d up to {cfg.ttrtc_max_delay})")
    return True


# ----------------------------------------------------------------------- T10
def test_ttrtc_sample_clamp() -> bool:
    """sample_actions with a TTRTC checkpoint hard-clamps the first d actions to the
    previous chunk; inference_delay=None (cold start) yields d=0 and degenerates to
    plain sampling."""
    torch = _load_torch()
    if torch is None:
        return None
    try:
        from low_level_model.models.pi0.modeling_pi0 import PI0Model
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] cannot import PI0Model: {exc}")
        return None

    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg.dtype = "float32"
    cfg.ttrtc = True
    model = PI0Model(cfg).eval()
    assert model._ttrtc_enabled(), "config.ttrtc=True must enable the TTRTC sampler"
    images, img_masks, tokens, masks, state, _ = _tiny_inputs(torch, cfg)

    B, T, A = 2, cfg.chunk_size, cfg.max_action_dim
    noise = torch.randn(B, T, A)
    prev = torch.full((B, T, A), 5.0)
    d = 3

    with torch.no_grad():
        base = model.sample_actions(images, img_masks, tokens, masks, state, noise=noise, num_steps=2)
        clamped = model.sample_actions(
            images, img_masks, tokens, masks, state, noise=noise, num_steps=2,
            prev_chunk_left_over=prev, inference_delay=d, execution_horizon=T,
        )
        cold = model.sample_actions(
            images, img_masks, tokens, masks, state, noise=noise, num_steps=2,
            prev_chunk_left_over=prev, inference_delay=None, execution_horizon=T,
        )

    assert torch.allclose(clamped[:, :d], prev[:, :d], atol=1e-6), \
        "the first d actions must be hard-clamped to the previous chunk"
    assert not torch.allclose(clamped[:, d:], prev[:, d:], atol=1e-3), \
        "the postfix must not be clamped (it is freely denoised)"
    assert torch.allclose(cold, base, atol=1e-5), \
        "cold start (inference_delay=None → d=0) must equal plain sampling"
    print(f"   sample_actions TTRTC OK: first {d} steps pinned to prev; cold start == plain")
    return True


# ----------------------------------------------------------------------- runner
def main():
    tests = [
        # ("T5 RTC streamer logic", test_rtc_streamer_logic),
        # ("T6 RTC guidance (PI0)", test_rtc_guidance_pi0),
        # ("T7 RTC compile retarget", test_rtc_compile_retarget),
        ("T8 TTRTC suffix embedder per-token", test_ttrtc_suffix_embedder_per_token),
        ("T9 TTRTC training forward", test_ttrtc_forward),
        ("T10 TTRTC sample hard-clamp", test_ttrtc_sample_clamp),
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
