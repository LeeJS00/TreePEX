"""fit_calibration.py — fit PDK-specific post-hoc calibration on valid split.

Pipeline (per PDK):
  1. Load valid split features + targets
  2. Override `fanout` with proxy output (cold-mode simulation)
  3. Run 5-seed ensemble → cold-style predictions (pred_gnd, pred_cpl, pred_total)
  4. Build per-net category labels from net name pattern
  5. Fit calibration components in order:
       (a) per-net-category multiplicative correction (gnd, cpl independently)
       (b) per-fanout-band isotonic regression (gnd, cpl independently)
       (c) per-cap-magnitude isotonic regression (total)
  6. Save calibration JSON under MODELS_DIR/calibration.json

Cold inference (pex_cold.py) loads this JSON and applies the same 3 steps after
base 5-seed prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pdk_paths import get_pdk  # noqa: E402

EPS = 1e-3
SEEDS = [42, 0, 1, 2, 3]


def classify_net(name: str) -> str:
    n = str(name).upper()
    if n.startswith("CTS_") or "_CTS_" in n:
        return "cts"
    if n.startswith("FE_DBT") or n.startswith("FE_OFC") or n.startswith("FE_OFN"):
        return "cts_buf"
    if n.startswith("FE_RN_") or n.startswith("FE_PSB"):
        return "fe_buf"
    if "_REG_" in n or "_REG[" in n.replace(" ", ""):
        return "reg"
    if "[" in n and "]" in n:
        return "bus"
    if n.startswith("N_"):
        return "auto"
    return "other"


def mape_med(p, g):
    p = np.asarray(p); g = np.asarray(g)
    return float(np.median(np.abs(p - g) / np.maximum(np.abs(g), EPS) * 100))


def _fit_isotonic_band(x, y, n_bins=20):
    """Piecewise-linear y(x) regression via quantile binning + median."""
    if len(x) < n_bins * 3:
        return None
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.quantile(xs, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    bin_x = []; bin_ratio = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (xs >= lo) & (xs <= hi if i == len(edges) - 2 else xs < hi)
        if mask.sum() < 5:
            continue
        x_med = float(np.median(xs[mask]))
        y_med = float(np.median(ys[mask]))
        bin_x.append(x_med); bin_ratio.append(y_med)
    return {"x": bin_x, "ratio": bin_ratio}


def _apply_isotonic_band(pred, x_arr, band):
    """Apply piecewise-linear ratio correction."""
    if band is None or len(band["x"]) < 2:
        return pred
    x_bin = np.asarray(band["x"]); r_bin = np.asarray(band["ratio"])
    ratio = np.interp(x_arr, x_bin, r_bin)
    return pred * ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", default="asap7", choices=["intel22", "asap7"])
    args = ap.parse_args()
    pdk = get_pdk(args.pdk)
    MODELS = pdk.models_dir
    print(f">>> fit_calibration pdk={pdk.name}", flush=True)

    # 1) Load features + targets
    print(">>> loading features ...", flush=True)
    base = pd.read_csv(pdk.v3_features)
    new = pd.read_csv(pdk.v4_new_feats)
    df = base.merge(new, on=["design_name", "net_name"], how="left")
    # Feature order from saved model
    feat_order = (MODELS / "FEATURE_ORDER.txt").read_text().strip().split("\n")
    df = df.dropna(subset=feat_order).reset_index(drop=True)
    print(f"   joined: {len(df):,}  feats={len(feat_order)}", flush=True)

    # 2) Override fanout with proxy (cold-mode)
    print(">>> applying proxy fanout (cold-mode override) ...", flush=True)
    import xgboost as xgb
    meta = json.loads((MODELS / "fanout_proxy_meta.json").read_text())
    proxy = xgb.XGBRegressor()
    proxy.load_model(str(MODELS / meta["model_file"]))
    Xp = df[meta["feats"]].fillna(0.0).values.astype(np.float32)
    df["fanout"] = np.maximum(proxy.predict(Xp), 1.0)

    # 3) Run 5-seed ensemble inference
    print(">>> 5-seed ensemble inference ...", flush=True)
    X = df[feat_order].astype(np.float32).values
    g_models = []; c_models = []
    for s in SEEDS:
        mg = xgb.XGBRegressor(); mg.load_model(str(MODELS / f"tweedie_gnd_seed{s}.json"))
        g_models.append(mg)
        mc = xgb.XGBRegressor(); mc.load_model(str(MODELS / f"tweedie_cpl_seed{s}.json"))
        c_models.append(mc)
    pred_g = np.mean([m.predict(X).clip(0) for m in g_models], axis=0)
    pred_c = np.mean([m.predict(X).clip(0) for m in c_models], axis=0)
    df["pred_gnd_cold"] = pred_g
    df["pred_cpl_cold"] = pred_c
    df["pred_total_cold"] = pred_g + pred_c

    # 4) Filter to valid split + add category
    valid = df[df["split"] == "valid"].reset_index(drop=True)
    print(f">>> valid split: {len(valid):,} nets", flush=True)
    valid["category"] = valid["net_name"].apply(classify_net)

    # ---- Baseline (no calibration) MAPE on valid ----
    base_mape_g = mape_med(valid["pred_gnd_cold"], valid["c_gnd_fF"])
    base_mape_c = mape_med(valid["pred_cpl_cold"], valid["c_cpl_total_fF"])
    base_mape_t = mape_med(valid["pred_total_cold"], valid["c_gnd_fF"] + valid["c_cpl_total_fF"])
    print(f"  baseline (no calib): MAPE_med gnd={base_mape_g:.2f}% cpl={base_mape_c:.2f}% tot={base_mape_t:.2f}%",
          flush=True)

    # ---- Step (a) per-category multiplicative correction ----
    print(">>> fitting per-category multipliers ...", flush=True)
    cat_factors = {}
    for cat, sub in valid.groupby("category", observed=True):
        if len(sub) < 30:
            continue
        # Optimal multiplier minimizing sum |alpha*pred - gold|
        # → use median ratio (gold/pred)
        ratio_g = float(np.median(sub["c_gnd_fF"] / np.maximum(sub["pred_gnd_cold"], EPS)))
        ratio_c = float(np.median(sub["c_cpl_total_fF"] / np.maximum(sub["pred_cpl_cold"], EPS)))
        # Clamp to [0.5, 2.0] to prevent runaway
        ratio_g = float(np.clip(ratio_g, 0.5, 2.0))
        ratio_c = float(np.clip(ratio_c, 0.5, 2.0))
        cat_factors[cat] = {"n": int(len(sub)), "ratio_g": ratio_g, "ratio_c": ratio_c}
        print(f"   {cat:10s}  n={len(sub):>6,}  ratio_g={ratio_g:.3f}  ratio_c={ratio_c:.3f}",
              flush=True)

    # Apply step (a)
    def apply_cat(row, ch):
        cat = classify_net(row["net_name"])
        f = cat_factors.get(cat, {}).get(f"ratio_{ch}", 1.0)
        return row[f"pred_{ch}_cold"] * f
    cat_arr = valid["category"].map(lambda c: cat_factors.get(c, {}))
    valid["pred_gnd_a"] = valid["pred_gnd_cold"] * cat_arr.map(
        lambda d: d.get("ratio_g", 1.0)).values
    valid["pred_cpl_a"] = valid["pred_cpl_cold"] * cat_arr.map(
        lambda d: d.get("ratio_c", 1.0)).values
    a_mape_g = mape_med(valid["pred_gnd_a"], valid["c_gnd_fF"])
    a_mape_c = mape_med(valid["pred_cpl_a"], valid["c_cpl_total_fF"])
    a_mape_t = mape_med(valid["pred_gnd_a"] + valid["pred_cpl_a"],
                        valid["c_gnd_fF"] + valid["c_cpl_total_fF"])
    print(f"  after step (a) cat: gnd={a_mape_g:.2f}% cpl={a_mape_c:.2f}% tot={a_mape_t:.2f}%",
          flush=True)

    # ---- Step (b) per-fanout-band isotonic on residual ratio ----
    print(">>> fitting per-fanout-band isotonic ...", flush=True)
    valid["fanout_log"] = np.log1p(valid["fanout"])
    ratio_g = (valid["c_gnd_fF"] / np.maximum(valid["pred_gnd_a"], EPS)).clip(0.3, 3.0)
    ratio_c = (valid["c_cpl_total_fF"] / np.maximum(valid["pred_cpl_a"], EPS)).clip(0.3, 3.0)
    fanout_band_g = _fit_isotonic_band(valid["fanout_log"].values, ratio_g.values, n_bins=12)
    fanout_band_c = _fit_isotonic_band(valid["fanout_log"].values, ratio_c.values, n_bins=12)
    if fanout_band_g:
        print(f"   gnd ratio @ fanout: x={[round(x,2) for x in fanout_band_g['x'][::2]]} "
              f"ratio={[round(r,3) for r in fanout_band_g['ratio'][::2]]}", flush=True)
    if fanout_band_c:
        print(f"   cpl ratio @ fanout: x={[round(x,2) for x in fanout_band_c['x'][::2]]} "
              f"ratio={[round(r,3) for r in fanout_band_c['ratio'][::2]]}", flush=True)
    valid["pred_gnd_b"] = _apply_isotonic_band(
        valid["pred_gnd_a"].values, valid["fanout_log"].values, fanout_band_g)
    valid["pred_cpl_b"] = _apply_isotonic_band(
        valid["pred_cpl_a"].values, valid["fanout_log"].values, fanout_band_c)
    b_mape_g = mape_med(valid["pred_gnd_b"], valid["c_gnd_fF"])
    b_mape_c = mape_med(valid["pred_cpl_b"], valid["c_cpl_total_fF"])
    b_mape_t = mape_med(valid["pred_gnd_b"] + valid["pred_cpl_b"],
                        valid["c_gnd_fF"] + valid["c_cpl_total_fF"])
    print(f"  after step (b) fanout: gnd={b_mape_g:.2f}% cpl={b_mape_c:.2f}% tot={b_mape_t:.2f}%",
          flush=True)

    # ---- Step (c) per-total-cap isotonic on total ----
    print(">>> fitting per-cap-magnitude isotonic on total ...", flush=True)
    valid["pred_total_b"] = valid["pred_gnd_b"] + valid["pred_cpl_b"]
    valid["pred_total_log"] = np.log1p(valid["pred_total_b"])
    gold_total = valid["c_gnd_fF"] + valid["c_cpl_total_fF"]
    ratio_t = (gold_total / np.maximum(valid["pred_total_b"], EPS)).clip(0.3, 3.0)
    total_band = _fit_isotonic_band(valid["pred_total_log"].values, ratio_t.values, n_bins=12)
    if total_band:
        print(f"   total ratio @ log1p(pred): x={[round(x,2) for x in total_band['x'][::2]]} "
              f"ratio={[round(r,3) for r in total_band['ratio'][::2]]}", flush=True)
    pred_total_c = _apply_isotonic_band(
        valid["pred_total_b"].values, valid["pred_total_log"].values, total_band)
    c_mape_t = mape_med(pred_total_c, gold_total)
    print(f"  after step (c) total: tot={c_mape_t:.2f}%", flush=True)

    # ---- Save calibration JSON ----
    out = {
        "pdk": pdk.name,
        "step_a_per_category": cat_factors,
        "step_b_per_fanout_log": {
            "gnd": fanout_band_g,
            "cpl": fanout_band_c,
        },
        "step_c_per_total_log": total_band,
        "baseline_valid_mape": {
            "gnd": round(base_mape_g, 3), "cpl": round(base_mape_c, 3),
            "tot": round(base_mape_t, 3),
        },
        "after_a_valid_mape": {
            "gnd": round(a_mape_g, 3), "cpl": round(a_mape_c, 3),
            "tot": round(a_mape_t, 3),
        },
        "after_b_valid_mape": {
            "gnd": round(b_mape_g, 3), "cpl": round(b_mape_c, 3),
            "tot": round(b_mape_t, 3),
        },
        "after_c_valid_mape_tot": round(c_mape_t, 3),
    }
    out_path = MODELS / "calibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n>>> wrote {out_path}  (gnd {base_mape_g:.2f}→{b_mape_g:.2f}, "
          f"cpl {base_mape_c:.2f}→{b_mape_c:.2f}, "
          f"tot {base_mape_t:.2f}→{c_mape_t:.2f})", flush=True)


if __name__ == "__main__":
    main()
