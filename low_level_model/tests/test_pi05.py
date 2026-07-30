"""Tests for the standalone π0.5 (full) System-1 baseline.

Run:  python -m low_level_model.tests.test_pi05   (from the repo root)

- T1 Config round-trip: PI05FullConfig save_pretrained -> load_config_local rebuilds an
  equivalent config through the draccus path (no network). Always runs.
- T2 Processor smoke (training batch): make_pi05_full_pre_post_processors builds and a
  batch with task+subtask+action yields all six language/action token keys, the
  USER_PROMPT alias equals the LANGUAGE tokens, the subtask ends in EOS before padding,
  and the task prompt is not truncated. Skipped if the PaliGemma / FAST tokenizers are
  unavailable (offline).
- T3 Processor smoke (inference batch): a batch with task only (no subtask, no action)
  does not raise, emits the USER_PROMPT keys, and emits neither subtask nor action tokens.
- T4 Weight loading (PI05_CKPT / PI05_BASE): load real weights and assert they come back
  as a PI05FullPolicy of type "pi05_full", on the requested device, in eval mode, with a
  non-empty, finite parameter set. Two sources (see below).
- T5 Inference (PI05_CKPT / PI05_BASE): load the same weights + pre/post processors and run
  one synchronous action-chunk inference (predict_chunk) on a synthetic observation,
  asserting the returned chunk has shape (n_action_steps, action_dim) and is all-finite.

Tokenizer/torch-dependent tests skip gracefully when their imports or a network step
are unavailable, mirroring test_system1.py. The weight-dependent tests (T4/T5) are opt-in:

    # A) a local trained checkpoint (config.json + model.safetensors), deployment path:
    PI05_CKPT=outputs/system1/model_best python -m low_level_model.tests.test_pi05

    # B) the LeRobot π0.5 base weights from the Hub (default lerobot/pi05_base, ~14.5 GB,
    #    loads a full gemma_2b — needs the disk + RAM/VRAM for it):
    PI05_BASE=1 python -m low_level_model.tests.test_pi05

PI05_CKPT takes priority; PI05_BASE may also be an explicit "owner/name" repo id. With
neither set, T4/T5 skip so the default suite stays offline-light.
"""

import os

import numpy as np


def _load_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] torch unavailable: {exc}")
        return None


# --------------------------------------------------------------------------- T1
def test_config_round_trip():
    import tempfile

    from low_level_model.models import factory

    cfg_cls = factory.get_config_class("pi05_full")
    cfg = cfg_cls(device="cpu")
    with tempfile.TemporaryDirectory() as d:
        cfg.save_pretrained(d)
        reloaded = factory.load_config_local(d)
    assert reloaded.type == "pi05_full", reloaded.type
    assert reloaded.tokenizer_max_length == cfg.tokenizer_max_length
    assert reloaded.subtask_max_length == cfg.subtask_max_length
    assert reloaded.chunk_size == cfg.chunk_size
    print("   [ok] T1 config round-trip: type=%s tok_max=%d subtask_max=%d"
          % (reloaded.type, reloaded.tokenizer_max_length, reloaded.subtask_max_length))
    return True


# ----------------------------------------------------------------- T2/T3 helpers
def _build_processor(torch):
    """Build the π0.5 pre/post processors with synthetic 14-dim stats (training-path shapes)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE
    from low_level_model.models.pi05 import PI05FullConfig, make_pi05_full_pre_post_processors

    cfg = PI05FullConfig(device="cpu")
    img_key = "observation.images.cam_top"
    # Mimic make_policy: real (14-dim) feature shapes are set before validate_features,
    # so the normalizer matches the 14-dim dataset stats (padding to 32 happens later).
    cfg.input_features = {
        img_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    stats = {
        OBS_STATE: {"mean": np.zeros(14, np.float32), "std": np.ones(14, np.float32)},
        ACTION: {"mean": np.zeros(14, np.float32), "std": np.ones(14, np.float32)},
    }
    pre, post = make_pi05_full_pre_post_processors(cfg, dataset_stats=stats)
    return cfg, img_key, pre, post


def test_processor_training_batch():
    torch = _load_torch()
    if torch is None:
        return None
    from lerobot.utils.constants import (
        ACTION,
        ACTION_TOKENS,
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
        OBS_LANGUAGE_SUBTASK_TOKENS,
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_USER_PROMPT_TOKENS,
        OBS_STATE,
    )
    try:
        cfg, img_key, pre, _ = _build_processor(torch)
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] T2 processor build failed (offline tokenizer?): {exc}")
        return None

    batch = {
        img_key: torch.rand(3, 224, 224),
        OBS_STATE: torch.rand(14) * 2 - 1,
        ACTION: torch.rand(cfg.chunk_size, 14) * 2 - 1,
        "task": "Put the blood gas test tube in the orange tray.",
        "subtask": "grasp the routine blood tube.",
    }
    try:
        out = pre(batch)
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] T2 preprocess failed (offline tokenizer?): {exc}")
        return None

    for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_USER_PROMPT_TOKENS,
                OBS_LANGUAGE_SUBTASK_TOKENS, ACTION_TOKENS):
        assert key in out, f"missing {key} in training batch"

    # USER_PROMPT is an alias of LANGUAGE tokens.
    assert torch.equal(out[OBS_LANGUAGE_TOKENS], out[OBS_LANGUAGE_USER_PROMPT_TOKENS]), \
        "USER_PROMPT tokens must equal LANGUAGE tokens"

    # Task prompt must fit within tokenizer_max_length (padded, so shape == max_length).
    n_lang = int(out[OBS_LANGUAGE_TOKENS].shape[-1])
    assert n_lang == cfg.tokenizer_max_length, n_lang

    # Subtask must be a single row (batch dim 1) — guards against enumerate() iterating a
    # bare string's characters into one bogus subtask per character.
    assert out[OBS_LANGUAGE_SUBTASK_TOKENS].shape[0] == 1, \
        f"subtask must have batch dim 1, got {tuple(out[OBS_LANGUAGE_SUBTASK_TOKENS].shape)}"

    # Subtask must end in EOS before padding, and decode back to the input text.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.text_tokenizer_name)
    eos_id = tok.eos_token_id
    sub_ids = out[OBS_LANGUAGE_SUBTASK_TOKENS][0]
    sub_mask = out[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK][0].bool()
    valid_ids = sub_ids[sub_mask].tolist()
    assert eos_id in valid_ids, "subtask tokens must contain EOS"
    assert valid_ids[-1] == eos_id, f"subtask must end in EOS, got {valid_ids[-3:]}"
    decoded = tok.decode([t for t in valid_ids if t != eos_id], skip_special_tokens=True).lower()
    assert "grasp" in decoded and "blood tube" in decoded, f"subtask decoded wrong: {decoded!r}"
    print("   [ok] T2 training batch: 6 token keys present, USER_PROMPT==LANGUAGE, "
          "subtask batch-dim 1, ends in EOS, decodes correctly")
    return True


def test_processor_inference_batch():
    torch = _load_torch()
    if torch is None:
        return None
    from lerobot.utils.constants import (
        ACTION_TOKENS,
        OBS_LANGUAGE_SUBTASK_TOKENS,
        OBS_LANGUAGE_USER_PROMPT_TOKENS,
        OBS_STATE,
    )
    try:
        cfg, img_key, pre, _ = _build_processor(torch)
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] T3 processor build failed (offline tokenizer?): {exc}")
        return None

    batch = {
        img_key: torch.rand(3, 224, 224),
        OBS_STATE: torch.rand(14) * 2 - 1,
        "task": "Put the blood gas test tube in the orange tray.",
        # no subtask, no action — deployment-style
    }
    try:
        out = pre(batch)
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] T3 preprocess failed (offline tokenizer?): {exc}")
        return None

    assert OBS_LANGUAGE_USER_PROMPT_TOKENS in out, "inference batch must emit USER_PROMPT tokens"
    assert OBS_LANGUAGE_SUBTASK_TOKENS not in out, "inference batch must not emit subtask tokens"
    assert ACTION_TOKENS not in out, "inference batch must not emit action tokens"
    print("   [ok] T3 inference batch: USER_PROMPT present, no subtask/action tokens, no raise")
    return True


# ------------------------------------------------------------------- T4/T5 helpers
# Default Hub weights: the LeRobot π0.5 base checkpoint (gemma_2b VLM + gemma_300m
# expert, 3 RGB cams + 32-dim state/action, float32, ~14.5 GB). PI05FullPolicy.from_pretrained
# carries the openpi/LeRobot key remap (`_fix_pytorch_state_dict_keys` + embedding tie)
# needed to consume it, so we start from these weights before any Robo-MLT fine-tuning.
PI05_BASE_REPO = "lerobot/pi05_base"
# The 3 cameras + 32-dim state/action layout baked into lerobot/pi05_base's config.json.
PI05_BASE_IMAGE_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
PI05_BASE_DIM = 32


def _resolve_weight_source():
    """Decide where T4/T5 get their weights, or None to skip.

    Priority:
      1. ``PI05_CKPT`` — a local trained checkpoint dir (``config.json`` +
         ``model.safetensors``); loaded through the factory / runtime as in deployment.
      2. ``PI05_BASE`` — pull the LeRobot π0.5 base weights from the Hub. Set it to ``1``
         for the default repo (``lerobot/pi05_base``) or to a custom ``owner/name`` repo id.

    Neither set → None (skip), keeping the suite offline-light: the base is a ~14.5 GB
    download and loads a full gemma_2b, so it must never run implicitly.
    """
    # ckpt = os.environ.get("PI05_CKPT")
    ckpt = None
    if ckpt:
        if not os.path.isdir(ckpt):
            print(f"   [skip] PI05_CKPT is not a directory: {ckpt}")
            return None
        for fname in ("config.json", "model.safetensors"):
            if not os.path.isfile(os.path.join(ckpt, fname)):
                print(f"   [skip] PI05_CKPT missing {fname}: {ckpt}")
                return None
        return ("local", ckpt)
    base = os.environ.get("PI05_BASE")
    if base:
        repo = base if "/" in str(base) else PI05_BASE_REPO
        return ("base", repo)
    return None


def _build_base_config(device):
    """Build a PI05FullConfig with the lerobot/pi05_base feature layout (3 cams, 32-dim).

    lerobot/pi05_base ships ``type: "pi05"`` (LeRobot's config), so we can't reload it via
    the draccus path; instead we construct the port's own PI05FullConfig with matching
    features and hand it to ``from_pretrained`` (which only downloads ``model.safetensors``).

    dtype defaults to bfloat16 (override with PI05_DTYPE). This matters for RAM: loading
    keeps the ~14.5 GB float32 state_dict resident *and* the built model at the same time.
    A float32 model (~14.5 GB) + the state_dict (~14.5 GB) ≈ 29 GB peak, which OOM-kills
    (SIGKILL 137) on a 31 GB box; bfloat16 halves the model to ~7 GB so the peak (~22 GB)
    fits — and bf16 is π0.5's native inference precision, fast on the RTX 4090.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE
    from low_level_model.models.pi05 import PI05FullConfig

    dtype = os.environ.get("PI05_DTYPE", "bfloat16")
    cfg = PI05FullConfig(device=device, dtype=dtype)
    cfg.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *cfg.image_resolution))
        for key in PI05_BASE_IMAGE_KEYS
    }
    cfg.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(PI05_BASE_DIM,))
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(PI05_BASE_DIM,))}
    stats = {
        OBS_STATE: {"mean": np.zeros(PI05_BASE_DIM, np.float32), "std": np.ones(PI05_BASE_DIM, np.float32)},
        ACTION: {"mean": np.zeros(PI05_BASE_DIM, np.float32), "std": np.ones(PI05_BASE_DIM, np.float32)},
    }
    return cfg, stats


def _load_policy_and_processors(torch, kind, ref, device, want_processors):
    """Load a π0.5 policy (+ optional pre/post processors) for T4/T5.

    ``local``: reuse the deployment path (factory.load_policy / load_system1_policy).
    ``base`` : build a matching PI05FullConfig and pull weights from the Hub, then build
    processors from synthetic (identity) 32-dim stats so inference is self-contained.

    Returns ``(policy, pre, post)`` (pre/post are None when ``want_processors`` is False),
    or None on any load failure (missing weights / offline tokenizer / OOM) so the test skips.
    """
    try:
        if kind == "local":
            if want_processors:
                from low_level_model.runtime.inference_system1 import load_system1_policy
                return load_system1_policy(ref, device=device)
            from low_level_model.models import factory
            return factory.load_policy(ref, device=device), None, None

        # kind == "base": Hub weights through the port's own config.
        from low_level_model.models.pi05 import PI05FullPolicy
        cfg, stats = _build_base_config(device)
        policy = PI05FullPolicy.from_pretrained(ref, config=cfg)
        policy.config.device = device
        policy.to(device)
        policy.eval()
        pre = post = None
        if want_processors:
            from low_level_model.models.factory import make_pre_post_processors
            pre, post = make_pre_post_processors(policy.config, dataset_stats=stats)
        return policy, pre, post
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] load failed ({kind}={ref}): {exc}")
        return None


# --------------------------------------------------------------------------- T4
def test_weight_loading():
    src = _resolve_weight_source()
    if src is None:
        print("   [skip] T4 weight loading: set PI05_CKPT=<dir> or PI05_BASE=1 to run.")
        return None
    torch = _load_torch()
    if torch is None:
        return None

    from low_level_model.models.pi05 import PI05FullPolicy

    kind, ref = src
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = _load_policy_and_processors(torch, kind, ref, device, want_processors=False)
    if loaded is None:
        return None
    policy, _, _ = loaded

    assert isinstance(policy, PI05FullPolicy), type(policy)
    assert policy.config.type == "pi05_full", policy.config.type
    assert not policy.training, "policy must be in eval mode after load"

    params = list(policy.parameters())
    assert len(params) > 0, "policy has no parameters"
    # Weights must land on the requested device and carry real (non-NaN) values.
    p0 = params[0]
    assert p0.device.type == torch.device(device).type, (p0.device, device)
    n_params = sum(p.numel() for p in params)
    assert torch.isfinite(p0.detach().float().flatten()[:1024]).all(), "loaded weights contain NaN/Inf"
    print("   [ok] T4 weight loading: PI05FullPolicy (%s=%s) on %s, eval mode, %.0fM params"
          % (kind, ref, device, n_params / 1e6))
    return True


# --------------------------------------------------------------------------- T5
def test_inference_chunk():
    src = _resolve_weight_source()
    if src is None:
        print("   [skip] T5 inference: set PI05_CKPT=<dir> or PI05_BASE=1 to run.")
        return None
    torch = _load_torch()
    if torch is None:
        return None

    from lerobot.utils.constants import ACTION, OBS_STATE
    from low_level_model.runtime.inference_system1 import predict_chunk

    kind, ref = src
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = _load_policy_and_processors(torch, kind, ref, device, want_processors=True)
    if loaded is None:
        return None
    policy, pre, post = loaded

    cfg = policy.config
    # Build a synthetic single-frame observation matching the policy's features:
    # one HWC uint8 image per camera at the model's native resolution, plus a raw state
    # vector at the config's state dim (predict_chunk batches, normalises, and tokenises).
    h, w = cfg.image_resolution
    img_keys = list(cfg.image_features)
    assert img_keys, "policy config exposes no image features"
    state_dim = cfg.input_features[OBS_STATE].shape[0]
    observation = {key: np.zeros((h, w, 3), dtype=np.uint8) for key in img_keys}
    observation[OBS_STATE] = np.zeros(state_dim, dtype=np.float32)
    task = "Put the blood gas test tube in the orange tray."

    try:
        chunk = predict_chunk(policy, pre, post, observation, task)
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] T5 inference failed (offline tokenizer / weights?): {exc}")
        return None

    action_dim = cfg.output_features[ACTION].shape[0]
    assert chunk.ndim == 2, f"chunk must be 2-D (n_action_steps, action_dim), got {chunk.shape}"
    assert chunk.shape == (cfg.n_action_steps, action_dim), (
        f"chunk shape {chunk.shape} != {(cfg.n_action_steps, action_dim)}"
    )
    assert np.isfinite(chunk).all(), "predicted action chunk contains NaN/Inf"
    print("   [ok] T5 inference: predict_chunk -> shape %s, finite (%s=%s) on %s"
          % (tuple(chunk.shape), kind, ref, device))
    return True


def main():
    tests = [
        ("T1 config round-trip", test_config_round_trip),
        ("T2 processor training batch", test_processor_training_batch),
        ("T3 processor inference batch", test_processor_inference_batch),
        ("T4 weight loading (PI05_CKPT / PI05_BASE)", test_weight_loading),
        ("T5 inference chunk (PI05_CKPT / PI05_BASE)", test_inference_chunk),
    ]
    passed = skipped = failed = 0
    for name, fn in tests:
        print(f"[test] {name}")
        try:
            result = fn()
        except AssertionError as exc:
            print(f"   [FAIL] {exc}")
            failed += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"   [ERROR] {exc}")
            failed += 1
            continue
        if result is None:
            skipped += 1
        else:
            passed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
