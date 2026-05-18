#!/usr/bin/env python3
"""TreePEX refinement-sprint ablation runner.

Reads cached cold-start features parquet (pex_cold writes them on every cold
run), runs per-seed prediction + optional L5 calibration + optional L11
specialist switch, writes per-net (net_name, gold_*, pred_gnd, pred_cpl)
CSV per (design, ablation, seed). Avoids re-running V4 H3 (1583 s on nova).

Ablations:
  baseline    canonical L11 stack: L5 ON + L11 specialist ON + XGB fanout proxy
  A1_no_L5    L5 calibration OFF
  A2_ridge    fanout proxy XGB -> Ridge baseline (simpler proxy)
  A4_no_L11   L11 specialist OFF (canonical preds only, L5 still ON)
  A6_no_L5_L11  joint: L5 OFF + L11 OFF
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
SCRIPTS_DIR = THIS.parent
TREEPEX_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# pex_cold resolves PDK from sys.argv at import time. Inject --pdk=asap7 if
# missing so MODELS_DIR / GOLDEN_SPEF_DIR / cached features all point to ASAP7.
_PDK_ARG = "asap7"
for _i, _a in enumerate(sys.argv):
    if _a == "--pdk" and _i + 1 < len(sys.argv):
        _PDK_ARG = sys.argv[_i + 1]; break
    if _a.startswith("--pdk="):
        _PDK_ARG = _a.split("=", 1)[1]; break
else:
    sys.argv.insert(1, f"--pdk={_PDK_ARG}")

import pex_cold as PC  # noqa: E402

MODELS_DIR = PC.MODELS_DIR
FEATURE_COLS_67 = PC.FEATURE_COLS_67
SEEDS = list(PC.SEEDS)
COLD_REPORT_DIR = PC.COLD_REPORT_DIR
ABLATION_DIR = TREEPEX_ROOT / "outputs" / "ablation"


def load_per_seed_models(weights_dir: Path = None):
    import xgboost as xgb
    d = weights_dir or MODELS_DIR
    out = {}
    for s in SEEDS:
        mg = xgb.XGBRegressor(); mg.load_model(str(d / f"tweedie_gnd_seed{s}.json"))
        mc = xgb.XGBRegressor(); mc.load_model(str(d / f"tweedie_cpl_seed{s}.json"))
        out[s] = (mg, mc)
    return out


def load_specialist_models(weights_dir: Path = None):
    import xgboost as xgb
    d = weights_dir or MODELS_DIR
    out = {}
    for s in SEEDS:
        mg = xgb.XGBRegressor(); mg.load_model(str(d / f"tweedie_specialist_gnd_seed{s}.json"))
        mc = xgb.XGBRegressor(); mc.load_model(str(d / f"tweedie_specialist_cpl_seed{s}.json"))
        out[s] = (mg, mc)
    return out


def apply_ridge_fanout(df: pd.DataFrame) -> np.ndarray:
    """Force fanout proxy = Ridge baseline (for A2_ridge)."""
    rp = json.loads((MODELS_DIR / "fanout_proxy_ridge.json").read_text())
    coef = np.asarray(rp["coef"], dtype=np.float64)
    intercept = float(rp["intercept"])
    feats = rp["feats"]
    X = df[feats].fillna(0.0).values
    X_log = np.log1p(X)
    pred_log = X_log @ coef + intercept
    return np.maximum(np.expm1(pred_log), 1.0)


def apply_l5(pred_g, pred_c, feat_df, calib):
    """L5 3-stage isotonic (extracted from pex_cold.run_design 1947-1995)."""
    def _classify(name):
        n = str(name).upper()
        if n.startswith("CTS_") or "_CTS_" in n: return "cts"
        if n.startswith("FE_DBT") or n.startswith("FE_OFC") or n.startswith("FE_OFN"): return "cts_buf"
        if n.startswith("FE_RN_") or n.startswith("FE_PSB"): return "fe_buf"
        if "_REG_" in n or "_REG[" in n: return "reg"
        if "[" in n and "]" in n: return "bus"
        if n.startswith("N_"): return "auto"
        return "other"

    def _isoband(pred, x, band):
        if band is None or len(band.get("x", [])) < 2:
            return pred
        return pred * np.interp(x, np.asarray(band["x"]), np.asarray(band["ratio"]))

    cat = feat_df["net_name"].apply(_classify)
    cat_dict = calib.get("step_a_per_category", {})
    rg = cat.map(lambda c: cat_dict.get(c, {}).get("ratio_g", 1.0)).values
    rc = cat.map(lambda c: cat_dict.get(c, {}).get("ratio_c", 1.0)).values
    pred_g = pred_g * rg
    pred_c = pred_c * rc

    fanout_log = np.log1p(feat_df["fanout"].values)
    pred_g = _isoband(pred_g, fanout_log, calib.get("step_b_per_fanout_log", {}).get("gnd"))
    pred_c = _isoband(pred_c, fanout_log, calib.get("step_b_per_fanout_log", {}).get("cpl"))

    pred_t = pred_g + pred_c
    pred_t_log = np.log1p(pred_t)
    pred_t_new = _isoband(pred_t, pred_t_log, calib.get("step_c_per_total_log"))
    scale = pred_t_new / np.maximum(pred_t, 1e-6)
    pred_g = pred_g * scale
    pred_c = pred_c * scale
    return pred_g, pred_c


def apply_l11(pred_g, pred_c, feat_df, X, spec_models, spec_meta):
    """L11 specialist switch (extracted from pex_cold.run_design 1997-2030)."""
    sw_feat = spec_meta["switch_feature"]
    sw_thr = float(spec_meta["switch_threshold"])
    # Per-config: use the seed-matched specialist for THIS seed's canonical preds.
    # Caller passes a single (mg, mc) tuple via spec_models for this seed.
    mg, mc = spec_models
    spec_pred_g = np.maximum(mg.predict(X), 0.0)
    spec_pred_c = np.maximum(mc.predict(X), 0.0)
    switch_mask = feat_df[sw_feat].values > sw_thr
    pred_g = np.where(switch_mask, spec_pred_g, pred_g)
    pred_c = np.where(switch_mask, spec_pred_c, pred_c)
    return pred_g, pred_c, int(switch_mask.sum())


ABLATIONS = {
    "baseline":     dict(l5=True,  l11=True,  proxy="xgb"),
    "A1_no_L5":     dict(l5=False, l11=True,  proxy="xgb"),
    "A2_ridge":     dict(l5=True,  l11=True,  proxy="ridge"),
    "A4_no_L11":    dict(l5=True,  l11=False, proxy="xgb"),
    "A6_no_L5_L11": dict(l5=False, l11=False, proxy="xgb"),
}


def run_one(design: str, ablation: str, seed: int, feat_df: pd.DataFrame,
            base_models: dict, spec_models, calib: dict, spec_meta,
            out_dir: Path):
    cfg = ABLATIONS[ablation]
    df = feat_df.copy()

    # Step 1: fanout proxy variant
    if cfg["proxy"] == "ridge":
        df["fanout"] = apply_ridge_fanout(df)
    # "xgb" branch: cached parquet already has XGB-proxy fanout, leave it.

    X = df[FEATURE_COLS_67].astype(np.float32).values

    # Step 2: per-seed base prediction (single seed, no mean)
    mg, mc = base_models[seed]
    pred_g = np.maximum(mg.predict(X), 0.0)
    pred_c = np.maximum(mc.predict(X), 0.0)

    # Step 3: L5 calibration (optional)
    if cfg["l5"]:
        pred_g, pred_c = apply_l5(pred_g, pred_c, df, calib)

    # Step 4: L11 specialist (optional, and only if PDK provides specialist weights)
    n_routed = 0
    if cfg["l11"] and spec_models is not None and spec_meta is not None:
        pred_g, pred_c, n_routed = apply_l11(pred_g, pred_c, df, X, spec_models[seed], spec_meta)

    out = pd.DataFrame({
        "net_name": df["net_name"].values,
        "pred_gnd": pred_g,
        "pred_cpl": pred_c,
        "pred_total": pred_g + pred_c,
        "fanout_used": df["fanout"].values,
        "wire_length_um": df["total_wire_length_um"].values,
    })
    out_path = out_dir / f"{design}_{ablation}_seed{seed}.csv"
    out.to_csv(out_path, index=False)
    return out_path, n_routed


def main():
    global FEATURE_COLS_67
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", choices=["intel22", "asap7"], default="asap7",
                    help="(consumed at import time; this just makes argparse happy)")
    ap.add_argument("--designs", nargs="+", required=True,
                    help="e.g. asap7_tv80s_x1 asap7_nova_x1")
    ap.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default outputs/ablation/refine_<UTC>)")
    ap.add_argument("--canonical_dir", type=str, default=None,
                    help="dir holding tweedie_{gnd,cpl}_seed*.json (default = PDK models_dir)")
    ap.add_argument("--specialist_dir", type=str, default=None,
                    help="dir holding tweedie_specialist_*.json + specialist_meta.json")
    ap.add_argument("--calib_path", type=str, default=None,
                    help="override calibration.json path; pass /dev/null or non-existent to disable L5")
    args = ap.parse_args()

    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ABLATION_DIR / f"refine_{ts}"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> ablation out dir: {out_dir}", flush=True)

    # Load shared artifacts once
    canon_dir = Path(args.canonical_dir) if args.canonical_dir else MODELS_DIR
    spec_dir = Path(args.specialist_dir) if args.specialist_dir else MODELS_DIR
    print(f">>> canonical weights: {canon_dir}", flush=True)
    print(f">>> specialist weights: {spec_dir}", flush=True)

    # Schema-aware: prefer canonical_dir/FEATURE_ORDER.txt over default 67-D.
    # B3 V3-only retrain writes a 41-D FEATURE_ORDER.txt to its own dir.
    feat_order_file = canon_dir / "FEATURE_ORDER.txt"
    if feat_order_file.exists():
        feat_cols = feat_order_file.read_text().strip().split("\n")
    else:
        feat_cols = list(FEATURE_COLS_67)
    print(f">>> feature schema: {len(feat_cols)}-D ({feat_order_file})", flush=True)
    # Patch module-level constant so run_one's `X = df[FEATURE_COLS_67]` uses
    # the active schema. This is a destructive monkey-patch — safe because
    # ablation_runner.main is a single-run entry point. `global` declared at
    # top of main() so this rebind is legal.
    FEATURE_COLS_67 = feat_cols

    base_models = load_per_seed_models(canon_dir)

    spec_meta_path = spec_dir / "specialist_meta.json"
    if spec_meta_path.exists():
        spec_models = load_specialist_models(spec_dir)
        spec_meta = json.loads(spec_meta_path.read_text())
        print(f">>> specialist loaded ({spec_meta['config']['depth']} depth)", flush=True)
    else:
        spec_models, spec_meta = None, None
        print(">>> specialist MISSING → L11 effectively OFF (e.g., intel22)", flush=True)

    calib_path = Path(args.calib_path) if args.calib_path else (MODELS_DIR / "calibration.json")
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        print(f">>> calibration loaded: {calib_path}", flush=True)
    else:
        # L5 dropped (post-2026-05-18): every (cfg["l5"]=True) path is a no-op.
        calib = {"step_a_per_category": {}, "step_b_per_fanout_log": {}, "step_c_per_total_log": None}
        print(f">>> calibration MISSING → L5 effectively OFF", flush=True)

    manifest_rows = []
    for design in args.designs:
        feat_path = COLD_REPORT_DIR / f"{design}_cold_features.parquet"
        if not feat_path.exists():
            print(f"!! missing cached features: {feat_path}", flush=True)
            continue
        feat_df = pd.read_parquet(feat_path)
        feat_df = feat_df.dropna(subset=FEATURE_COLS_67).reset_index(drop=True)
        print(f">>> {design}: {len(feat_df):,} nets cached", flush=True)

        for ablation in args.ablations:
            t0 = time.time()
            for seed in args.seeds:
                p, n_rt = run_one(design, ablation, seed, feat_df,
                                  base_models, spec_models, calib, spec_meta, out_dir)
                manifest_rows.append({
                    "design": design, "ablation": ablation, "seed": seed,
                    "n_nets": len(feat_df), "n_routed_l11": n_rt,
                    "pred_csv": str(p.relative_to(out_dir))})
            print(f"  {ablation}: {len(args.seeds)}-seed in {time.time()-t0:.1f}s", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (out_dir / "ablations_used.json").write_text(json.dumps({
        "designs": args.designs, "ablations": args.ablations, "seeds": args.seeds,
        "ABLATIONS_def": ABLATIONS,
    }, indent=2))
    print(f">>> done: {len(manifest)} runs -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
