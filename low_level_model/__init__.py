"""System 1 (generative executor) of Robo-MLT, its baselines, and the deployment runtime.

System 1 couples a pretrained Qwen3-VL-2B backbone with a flow-matching action head
(``models/qwen3vl_vla``, paper Sec. III-C). The ACT, SmolVLA, pi0 and pi0.5 baselines live next to
it under ``models/`` and share the trainer (``training/train_system1.py``) and the runtime.

Public names are imported lazily, so lightweight modules such as ``robot.system2_controller`` can
be used without the LeRobot fork that the policies depend on.
"""

import importlib

_LAZY = {
    "get_policy_class": "low_level_model.models.factory",
    "make_policy": "low_level_model.models.factory",
    "load_policy": "low_level_model.models.factory",
    "System1AsyncStreamer": "low_level_model.runtime.inference_system1",
    "load_system1_policy": "low_level_model.runtime.inference_system1",
    "predict_chunk": "low_level_model.runtime.inference_system1",
    "resolve_policy_ref": "low_level_model.runtime.inference_system1",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'low_level_model' has no attribute {name!r}")
