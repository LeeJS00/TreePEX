"""pdk_paths.py — PDK-specific path + design registry shared by all TreePEX scripts.

Multi-PDK support: `intel22` (22nm, default) and `asap7` (7nm).

Path resolution priority:
  1. Environment variables (TREEPEX_<PDK>_V3_FEATURES, TREEPEX_<PDK>_V4_NEW_FEATS,
     TREEPEX_<PDK>_GOLDEN_DIR)
  2. Bundled defaults under the repo root (for the smoke-test deployment)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PDKConfig:
    name: str
    v3_features: str       # 41-D base feature CSV (per-net, with `split` column)
    v4_new_feats: str      # 26-D H3 aggressor feature CSV
    models_dir: Path       # 5-seed ensemble JSON weights
    golden_spef_dir: Path  # StarRC golden SPEF directory
    golden_pattern: str    # filename pattern, {design} or {base} placeholder
    test_designs: Tuple[str, ...]
    # Optional pretty name for SPEF header / leaderboard
    pretty: str = ""


def _env_or(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


PDK_REGISTRY: dict[str, PDKConfig] = {
    "intel22": PDKConfig(
        name="intel22",
        v3_features=_env_or(
            "TREEPEX_INTEL22_V3_FEATURES",
            "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
        v4_new_feats=_env_or(
            "TREEPEX_INTEL22_V4_NEW_FEATS",
            "/home/jslee/projects/PINNPEX/archive/pex_v4/results/new_features_with_ids.csv"),
        models_dir=ROOT / "models",
        golden_spef_dir=Path(_env_or(
            "TREEPEX_INTEL22_GOLDEN_DIR",
            str(ROOT / "data" / "golden_spef"))),
        golden_pattern="{design}_starrc.spef",
        test_designs=("intel22_tv80s_f3", "intel22_nova_f3"),
        pretty="Intel 22nm",
    ),
    "asap7": PDKConfig(
        name="asap7",
        v3_features=_env_or(
            "TREEPEX_ASAP7_V3_FEATURES",
            "/data/PINNPEX/data/processed_v3/asap7/features/all_designs.csv"),
        v4_new_feats=_env_or(
            "TREEPEX_ASAP7_V4_NEW_FEATS",
            str(ROOT / "inputs" / "asap7_new_features_with_ids.csv")),
        models_dir=ROOT / "models_asap7",
        golden_spef_dir=Path(_env_or(
            "TREEPEX_ASAP7_GOLDEN_DIR",
            str(ROOT / "data" / "golden_spef"))),
        # ASAP7 SPEF naming: asap7_<base>_fs_en_starrc.spef.typical where
        # design = asap7_<base>_x1 → strip _x1 suffix at lookup time
        golden_pattern="{base}_fs_en_starrc.spef.typical",
        test_designs=("asap7_tv80s_x1", "asap7_nova_x1"),
        pretty="ASAP7 7nm",
    ),
}


def get_pdk(name: str) -> PDKConfig:
    if name not in PDK_REGISTRY:
        raise ValueError(f"unknown pdk={name!r}; choices: {list(PDK_REGISTRY)}")
    return PDK_REGISTRY[name]


def golden_path(pdk: PDKConfig, design: str) -> Path:
    """Resolve golden SPEF path for one design within a PDK.

    intel22: <golden_dir>/<design>_starrc.spef[.gz]
    asap7:   <golden_dir>/<design-minus-_x1>_fs_en_starrc.spef.typical[.gz]

    Caller should handle .gz fallthrough (see 04_compare_golden.py).
    """
    if pdk.name == "asap7":
        base = design[:-3] if design.endswith("_x1") else design
        return pdk.golden_spef_dir / pdk.golden_pattern.format(base=base)
    return pdk.golden_spef_dir / pdk.golden_pattern.format(design=design)
