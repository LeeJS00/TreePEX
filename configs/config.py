"""TreePEX path & runtime configuration.

All paths resolve in this priority order:
  1. Environment variable (e.g. TREEPEX_DEF_DIR, TREEPEX_TILE_CACHE_ROOT)
  2. Bundled defaults under the repo root

This lets a deployment work out-of-the-box on bundled `intel22_tv80s_f3`
smoke-test data and lets external users redirect paths via env vars.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of `configs/`
ROOT = Path(__file__).resolve().parent.parent

# ---- PDK (bundled, ~6 MB total) ----
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
# Bundled smoke-test: intel22_tv80s_f3 only (3.9 MB DEF, 63 MB SPEF gzipped to 12 MB)
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

# Known designs (DEF file paths)
DESIGNS = {
    "intel22_tv80s_f3": DEF_DIR / "intel22_tv80s_f3.def",
    "intel22_nova_f3":  DEF_DIR / "intel22_nova_f3.def",
}


def resolve_def(design: str) -> Path:
    """Resolve a design's DEF path; raises FileNotFoundError with a helpful message."""
    p = DESIGNS.get(design)
    if p is None:
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
    """Resolve a design's golden SPEF path, transparently handling .gz."""
    for cand in [
        GOLDEN_SPEF_DIR / f"{design}_starrc.spef",
        GOLDEN_SPEF_DIR / f"{design}_starrc.spef.gz",
    ]:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Golden SPEF for {design} not found in {GOLDEN_SPEF_DIR}. "
        f"Set TREEPEX_GOLDEN_DIR or place {design}_starrc.spef(.gz) there.")
