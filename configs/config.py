"""TreePEX path & runtime configuration (multi-PDK).

Supports `intel22` (22nm, default) and `asap7` (7nm). The default PDK is
intel22 for backward compatibility with `pex_cold.py`. The paper-benchmark
scripts (01_train_save_models, 02_inference, 03_write_spef, 04_compare_golden,
pex_tool) take a `--pdk` flag and resolve all paths through `pdk_paths.py`.

All paths resolve in this priority order:
  1. Environment variable (e.g. `TREEPEX_DEF_DIR`, `TREEPEX_TILE_CACHE_ROOT`)
  2. Bundled defaults under the repo root

This lets a deployment work out-of-the-box on bundled smoke-test data and
lets external users redirect paths via env vars.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of `configs/`
ROOT = Path(__file__).resolve().parent.parent

# ---- PDK (bundled, intel22 default for legacy pex_cold.py) ----
TECH_LEF_PATH = Path(os.environ.get(
    "TREEPEX_TECH_LEF",
    str(ROOT / "tool" / "pdk" / "22nm" / "tech_lef" / "p1222_js.lef"),
))
CELL_LEF_PATH = Path(os.environ.get(
    "TREEPEX_CELL_LEF",
    str(ROOT / "tool" / "pdk" / "22nm" / "cell_lef" / "b15_nn.lef"),
))
LAYERS_INFO_PATH = Path(os.environ.get(
    "TREEPEX_LAYERS_INFO",
    str(ROOT / "tool" / "pdk" / "22nm" / "layers" / "layers.info"),
))

# ---- Design data ----
# Bundled smoke-test: intel22_tv80s_f3 (3.9 MB DEF, 12 MB gz SPEF)
# +                   asap7_gcd_x1 (0.2 MB DEF, 0.5 MB gz SPEF)
DEF_DIR = Path(os.environ.get("TREEPEX_DEF_DIR", str(ROOT / "data" / "def")))
GOLDEN_SPEF_DIR = Path(os.environ.get(
    "TREEPEX_GOLDEN_DIR", str(ROOT / "data" / "golden_spef")))

# Tile cache (V4 H3 features). Not bundled — designed for site-local storage.
# For deployment without V4 H3, the cold-start pipeline still produces a
# 41-D-only prediction (less accurate). Override with TREEPEX_TILE_CACHE_ROOT.
TILE_CACHE_ROOT = Path(os.environ.get(
    "TREEPEX_TILE_CACHE_ROOT",
    "/data/PINNPEX/data/processed_v3/intel22",
))

# Known designs (DEF file paths). Used by `pex_cold.py` (intel22 only).
# ASAP7 designs are resolved via `scripts/pdk_paths.py::PDK_REGISTRY`.
DESIGNS = {
    "intel22_tv80s_f3": DEF_DIR / "intel22_tv80s_f3.def",
    "intel22_nova_f3":  DEF_DIR / "intel22_nova_f3.def",
}


def resolve_def(design: str) -> Path:
    """Resolve a design's DEF path; raises FileNotFoundError with a helpful message."""
    p = DESIGNS.get(design)
    if p is None:
        # ASAP7 path: <DEF_DIR>/<design>.def with same-name convention
        alt = DEF_DIR / f"{design}.def"
        if alt.exists():
            return alt
        raise SystemExit(
            f"Unknown design {design!r}. Add to configs.config.DESIGNS or "
            f"set DEF path via env (TREEPEX_DEF_DIR points to the dir holding "
            f"<design>.def).")
    if not p.exists():
        # Try a few fallback search paths.
        for alt in [
            ROOT / "data" / "def" / f"{design}.def",
            Path(os.environ.get("TREEPEX_DEF_DIR", "")) / f"{design}.def" if os.environ.get("TREEPEX_DEF_DIR") else None,
        ]:
            if alt is not None and alt.exists():
                return alt
        raise FileNotFoundError(
            f"DEF for {design} not found. Looked at {p}. "
            f"Set TREEPEX_DEF_DIR or symlink the file.")
    return p


def resolve_golden_spef(design: str) -> Path:
    """Resolve a design's golden SPEF path, transparently handling .gz.

    intel22 convention: `<design>_starrc.spef[.gz]`
    ASAP7 convention:   `<design-stem>_fs_en_starrc.spef.typical[.gz]`
                        where stem = design minus `_x1` suffix.
    """
    candidates = [
        GOLDEN_SPEF_DIR / f"{design}_starrc.spef",
        GOLDEN_SPEF_DIR / f"{design}_starrc.spef.gz",
    ]
    # ASAP7-style: strip _x1 suffix if present
    if design.endswith("_x1"):
        base = design[:-3]
        candidates += [
            GOLDEN_SPEF_DIR / f"{base}_fs_en_starrc.spef.typical",
            GOLDEN_SPEF_DIR / f"{base}_fs_en_starrc.spef.typical.gz",
            GOLDEN_SPEF_DIR / f"{base}_starrc.spef",
            GOLDEN_SPEF_DIR / f"{base}_starrc.spef.gz",
        ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Golden SPEF for {design} not found in {GOLDEN_SPEF_DIR}. "
        f"Tried: {[str(c.name) for c in candidates]}. "
        f"Set TREEPEX_GOLDEN_DIR or place the file there.")
