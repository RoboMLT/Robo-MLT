# Third-Party Notices

Robo-MLT is licensed under Apache-2.0 (see `LICENSE`). Several modules adapt or port code from
other open-source projects; this file lists them and their upstream license terms as documented
at time of writing. Please verify against the current upstream license before further
redistribution.

## `low_level_model/models/pi0/`

A from-scratch reimplementation of the π0 (PI0) architecture — PaliGemma vision-language backbone
+ Gemma action expert + flow-matching action head — as described in Black et al., *"π0: A
Vision-Language-Action Flow Model for General Robot Control"* (Physical Intelligence). It plugs
into the [LeRobot](https://github.com/huggingface/lerobot) policy framework
(`PreTrainedConfig`/`PreTrainedPolicy`, optimizer/scheduler configs). LeRobot is Apache-2.0.

## `low_level_model/models/pi05/`

A PyTorch port that closely follows
[openpi](https://github.com/Physical-Intelligence/openpi)'s π0.5 reference implementation
(`PI0Pytorch`, its preprocessing, and its AdamW/cosine-decay schedule), adapted onto the LeRobot
policy framework with knowledge-insulation and subtask autoregressive decoding. openpi is
Apache-2.0.

## `low_level_model/models/qwen3vl_vla/`

A documented port of [starVLA](https://github.com/starVLA/starVLA)'s `Qwen_PI` framework
(`framework/QwenPI.py`, `modules/vlm/QWen3.py`,
`modules/action_model/LayerwiseFM_ActionHeader.py`, and its cross-attention flow-matching DiT
head) onto the LeRobot `PreTrainedPolicy` interface. **starVLA's repository does not assert a
license** (checked via the GitHub API license endpoint) — confirm terms directly with the starVLA
authors/repository before redistributing this module outside research use.

## `low_level_model/runtime/action_smoothing.py`

`TemporalChunkBlender.merge` and `interpolate_action` implement the χ₀ output-side temporal
chunk-smoothing scheme (Yu et al. 2026, Algorithm 1), with `StreamActionBuffer.integrate_new_chunk`
in [OpenDriveLab/kai0](https://github.com/OpenDriveLab/kai0) as the reference implementation.
kai0 is Apache-2.0.

## LeRobot fork (external, not vendored)

Real-robot deployment (`low_level_model/robot/`) depends on a fork of
[LeRobot](https://github.com/huggingface/lerobot) with added `piper`/`dual_piper` robot and
`PiperMotorsBus` support (see `README.md` → Installation). Upstream LeRobot is Apache-2.0; confirm
the fork you use preserves that license.
