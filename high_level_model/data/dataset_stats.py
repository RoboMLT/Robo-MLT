"""Report the dataset figures quoted in the paper (Sec. IV-A-3) from LeRobot v3 trees.

Reads parquet/JSON directly (pandas + pyarrow; no LeRobot install needed) and reports, per task:

    - nominal corpus: episodes, *distinct* episodes (md5 of the action array) and duplicates,
      complete workflows (episodes spanning >= 2 atomic skills; also how many cover every skill
      of the library) vs. single atomic-skill episodes, frames, fps, cameras, task text;
    - recovery corpus: episodes, how many carry recovery positives, the K_rec range, whether the
      positives form the final block of each episode, the intervention / failure split from
      ``meta/recovery_episodes.json``, and any overlap with the nominal corpus;
    - a byte-for-byte check of ``meta/subtasks.parquet`` against the skill library.

Usage::

    python -m high_level_model.data.dataset_stats \\
        --root <path/to>/RoboMLT-Lab/blood_gas_analysis \\
        --recovery-root <path/to>/RoboMLT-Lab/blood_gas_analysis_recovery \\
        --skill-library configs/skill_library/bloodgas.yaml --json stats_bga.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

__all__ = ["corpus_stats", "recovery_stats", "library_check", "to_markdown"]

_COLUMNS = ("episode_index", "action", "observation.state", "subtask_index", "back_event")


def _info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def _frames(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files under {root / 'data'}")
    names = set(pq.read_schema(files[0]).names)
    cols = [c for c in _COLUMNS if c in names]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def _scalar_column(df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    if name not in df.columns:
        return None
    return np.asarray([np.asarray(v).reshape(-1)[0] for v in df[name]], dtype=np.int64)


def _episodes(df: pd.DataFrame) -> List[dict]:
    """Per-episode length, action hash, subtask segments and recovery positives."""
    sub = _scalar_column(df, "subtask_index")
    back = _scalar_column(df, "back_event")
    out = []
    for ep, idx in df.groupby("episode_index").indices.items():
        idx = np.sort(idx)
        action = np.stack(df["action"].to_numpy()[idx]).astype(np.float32)
        rec = {"episode": int(ep), "length": int(len(idx)),
               "hash": hashlib.md5(action.tobytes()).hexdigest()}
        if "observation.state" in df.columns:
            state = np.stack(df["observation.state"].to_numpy()[idx]).astype(np.float32)
            rec["mean_abs_action_minus_state"] = float(np.abs(action - state).mean())
        if sub is not None:
            s = sub[idx]
            runs = [int(s[0])] + [int(v) for a, v in zip(s[:-1], s[1:], strict=True) if v != a]
            rec["segments"], rec["skills"] = runs, sorted(set(runs))
        if back is not None:
            b = back[idx] > 0
            n_pos = int(b.sum())
            rec["recovery_frames"] = n_pos
            rec["recovery_is_tail"] = bool(n_pos and b[-n_pos:].all())
        out.append(rec)
    return out


def _texts(root: Path, name: str, col: str) -> Dict[int, str]:
    path = root / "meta" / name
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if col in df.columns:            # LeRobot v3 layout: index = text, column = integer index
        return {int(i): str(t) for t, i in zip(df.index, df[col], strict=True)}
    return {int(i): str(t) for i, t in enumerate(df.index)}


def library_check(root: Path, library_path: Optional[str]) -> dict:
    """Compare meta/subtasks.parquet with the library's canonical instructions, index by index."""
    data_texts = _texts(root, "subtasks.parquet", "subtask_index")
    if not library_path:
        return {"subtasks": data_texts}
    lib = yaml.safe_load(Path(library_path).read_text(encoding="utf-8")) or {}
    lib_texts = {i: s["canonical_instruction"] for i, s in enumerate(lib.get("skills", []))}
    mismatches = {i: {"dataset": data_texts.get(i), "library": lib_texts.get(i)}
                  for i in sorted(set(data_texts) | set(lib_texts))
                  if data_texts.get(i) != lib_texts.get(i)}
    return {"num_skills_library": len(lib_texts), "num_skills_dataset": len(data_texts),
            "match": not mismatches, "mismatches": mismatches}


def corpus_stats(root: Path, num_skills: Optional[int] = None) -> dict:
    info = _info(root)
    eps = _episodes(_frames(root))
    seen, dup = set(), 0
    for e in eps:
        dup += e["hash"] in seen
        seen.add(e["hash"])
    with_seg = [e for e in eps if "segments" in e]
    workflow = [e for e in with_seg if len(e["segments"]) >= 2]
    stats = {
        "root": str(root),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "cameras": [k for k, v in info["features"].items() if v.get("dtype") in ("video", "image")],
        "tasks": sorted(set(_texts(root, "tasks.parquet", "task_index").values())),
        "episodes": len(eps),
        "distinct_episodes": len(seen),
        "duplicate_episodes": dup,
        "frames": int(sum(e["length"] for e in eps)),
        "complete_workflows": len(workflow),
        "atomic_skill_episodes": len(with_seg) - len(workflow),
        "episodes_with_recovery_positives": sum(1 for e in eps if e.get("recovery_frames")),
    }
    if num_skills:
        stats["workflows_covering_all_skills"] = sum(
            1 for e in workflow if len(e["skills"]) >= num_skills)
    gaps = [e["mean_abs_action_minus_state"] for e in eps if "mean_abs_action_minus_state" in e]
    if gaps:
        stats["mean_abs_action_minus_state"] = float(np.mean(gaps))
    stats["_hashes"] = seen
    return stats


def recovery_stats(root: Path, nominal_hashes: Optional[set] = None) -> dict:
    eps = _episodes(_frames(root))
    k = [e["recovery_frames"] for e in eps if e.get("recovery_frames")]
    tags_path = root / "meta" / "recovery_episodes.json"
    tags = json.loads(tags_path.read_text(encoding="utf-8")) if tags_path.exists() else {}
    kinds: Dict[str, int] = {}
    for tag in tags.values():
        kind = tag.get("type", "unknown") if isinstance(tag, dict) else str(tag)
        kinds[kind] = kinds.get(kind, 0) + 1
    hashes = {e["hash"] for e in eps}
    return {
        "root": str(root),
        "episodes": len(eps),
        "distinct_episodes": len(hashes),
        "episodes_with_positives": len(k),
        "episodes_without_positives": len(eps) - len(k),
        "k_rec_min": int(min(k)) if k else None,
        "k_rec_max": int(max(k)) if k else None,
        "k_rec_mean": float(np.mean(k)) if k else None,
        "positives_are_final_block": all(e.get("recovery_is_tail", False) for e in eps
                                         if e.get("recovery_frames")),
        "episode_types": kinds,
        "untagged_episodes": len(eps) - len(tags),
        "overlap_with_nominal": len(hashes & nominal_hashes) if nominal_hashes else None,
    }


def to_markdown(report: dict) -> str:
    n = report["nominal"]
    lines = [
        f"### {Path(n['root']).name}",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| robot_type / fps | {n['robot_type']} / {n['fps']} |",
        f"| cameras | {', '.join(c.split('.')[-1] for c in n['cameras'])} |",
        f"| episodes (distinct / duplicates) | {n['episodes']} ({n['distinct_episodes']} / "
        f"{n['duplicate_episodes']}) |",
        f"| complete workflows | {n['complete_workflows']}"
        + (f" ({n['workflows_covering_all_skills']} cover every skill)"
           if "workflows_covering_all_skills" in n else "") + " |",
        f"| atomic-skill episodes | {n['atomic_skill_episodes']} |",
        f"| frames | {n['frames']} |",
    ]
    r = report.get("recovery")
    if r:
        lines += [
            f"| recovery episodes (with positives) | {r['episodes']} ({r['episodes_with_positives']}) |",
            f"| recovery episode types | {r['episode_types'] or 'untagged'} |",
            f"| K_rec range | {r['k_rec_min']}–{r['k_rec_max']} (final block: "
            f"{r['positives_are_final_block']}) |",
            f"| recovery episodes also in the nominal corpus | {r['overlap_with_nominal']} |",
        ]
    lib = report.get("library", {})
    if "match" in lib:
        lines.append(f"| subtask text == skill library | {lib['match']} |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="Nominal corpus (LeRobot v3 root)")
    ap.add_argument("--recovery-root", type=Path, default=None, help="Recovery corpus root")
    ap.add_argument("--skill-library", default=None, help="configs/skill_library/<task>.yaml")
    ap.add_argument("--json", type=Path, default=None, help="Also write the report as JSON")
    args = ap.parse_args()

    library = library_check(args.root, args.skill_library)
    nominal = corpus_stats(args.root, library.get("num_skills_library"))
    hashes = nominal.pop("_hashes")
    report = {"nominal": nominal, "library": library}
    if args.recovery_root:
        report["recovery"] = recovery_stats(args.recovery_root, hashes)
    print(to_markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote %s", args.json)
    if nominal["duplicate_episodes"]:
        logger.warning("%d duplicated episodes in %s: quote the distinct count.",
                       nominal["duplicate_episodes"], args.root)
    if library.get("match") is False:
        logger.warning("Subtask text differs from the skill library: %s", library["mismatches"])


if __name__ == "__main__":
    main()
