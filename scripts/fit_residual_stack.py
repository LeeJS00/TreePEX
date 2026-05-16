"""fit_residual_stack.py — L8 stacked residual XGBoost.

Train a shallow XGBoost to predict (gold - base_pred) given features + base_pred
on the valid split (out-of-sample for the base 5-seed ensemble).

Applied at cold inference: final = base_pred + residual_pred. Captures
systematic biases the base model missed (e.g. high-fanout over-prediction,
mid-fanout under-prediction observed in per-net error analysis).
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


def mape_med(p, g):
    return float(np.median(np.abs(p - g) / np.maximum(np.abs(g), EPS) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", default="asap7", choices=["intel22", "asap7"])
    args = ap.parse_args()
    pdk = get_pdk(args.pdk)
    MODELS = pdk.models_dir

    # 1) Features + targets
    print(">>> loading features ...", flush=True)
    base = pd.read_csv(pdk.v3_features)
    new = pd.read_csv(pdk.v4_new_feats)
    df = base.merge(new, on=["design_name", "net_name"], how="left")
    feat_order = (MODELS / "FEATURE_ORDER.txt").read_text().strip().split("\n")
    df = df.dropna(subset=feat_order).reset_index(drop=True)

    # 2) Apply proxy fanout (cold-mode)
    import xgboost as xgb
    meta = json.loads((MODELS / "fanout_proxy_meta.json").read_text())
    proxy = xgb.XGBRegressor()
    proxy.load_model(str(MODELS / meta["model_file"]))
    Xp = df[meta["feats"]].fillna(0.0).values.astype(np.float32)
    df["fanout"] = np.maximum(proxy.predict(Xp), 1.0)

    # 3) Base 5-seed ensemble predictions
    print(">>> base ensemble predict ...", flush=True)
    X = df[feat_order].astype(np.float32).values
    g_models = []; c_models = []
    for s in SEEDS:
        mg = xgb.XGBRegressor(); mg.load_model(str(MODELS / f"tweedie_gnd_seed{s}.json"))
        g_models.append(mg)
        mc = xgb.XGBRegressor(); mc.load_model(str(MODELS / f"tweedie_cpl_seed{s}.json"))
        c_models.append(mc)
    pred_g = np.mean([m.predict(X).clip(0) for m in g_models], axis=0)
    pred_c = np.mean([m.predict(X).clip(0) for m in c_models], axis=0)
    df["pred_gnd_base"] = pred_g
    df["pred_cpl_base"] = pred_c

    # 4) Filter to valid only (out-of-sample for base)
    valid = df[df["split"] == "valid"].reset_index(drop=True)
    print(f">>> valid: {len(valid):,} nets", flush=True)

    # Baseline cold-style MAPE (no residual stack yet)
    base_g_mape = mape_med(valid["pred_gnd_base"], valid["c_gnd_fF"])
    base_c_mape = mape_med(valid["pred_cpl_base"], valid["c_cpl_total_fF"])
    base_t_mape = mape_med(valid["pred_gnd_base"] + valid["pred_cpl_base"],
                            valid["c_gnd_fF"] + valid["c_cpl_total_fF"])
    print(f"  baseline (no stack): gnd={base_g_mape:.2f}%  cpl={base_c_mape:.2f}%  tot={base_t_mape:.2f}%",
          flush=True)

    # 5) Fit residual models (gnd + cpl independently)
    print(">>> fitting residual stack ...", flush=True)
    residual_feats = feat_order + ["pred_gnd_base", "pred_cpl_base"]
    X_v = valid[residual_feats].astype(np.float32).values
    r_g = (valid["c_gnd_fF"] - valid["pred_gnd_base"]).values  # residual to learn
    r_c = (valid["c_cpl_total_fF"] - valid["pred_cpl_base"]).values

    # K-fold within valid for OOF if we wanted, but valid is small enough
    # to use 80/20 holdout — fit on 80%, validate stack on 20%
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(valid))
    n_fit = int(0.8 * len(valid))
    fit_idx = perm[:n_fit]; val_idx = perm[n_fit:]

    rm_g = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=16, tree_method="hist",
        early_stopping_rounds=30)
    rm_g.fit(X_v[fit_idx], r_g[fit_idx],
             eval_set=[(X_v[val_idx], r_g[val_idx])], verbose=False)
    print(f"   gnd residual model: best_iter={rm_g.best_iteration}", flush=True)
    rm_c = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=16, tree_method="hist",
        early_stopping_rounds=30)
    rm_c.fit(X_v[fit_idx], r_c[fit_idx],
             eval_set=[(X_v[val_idx], r_c[val_idx])], verbose=False)
    print(f"   cpl residual model: best_iter={rm_c.best_iteration}", flush=True)

    # 6) Eval on val_idx (held-out 20% of valid)
    pred_resid_g = rm_g.predict(X_v[val_idx])
    pred_resid_c = rm_c.predict(X_v[val_idx])
    corrected_g = valid.iloc[val_idx]["pred_gnd_base"].values + pred_resid_g
    corrected_c = valid.iloc[val_idx]["pred_cpl_base"].values + pred_resid_c
    corrected_t = corrected_g + corrected_c
    eval_g_mape = mape_med(corrected_g, valid.iloc[val_idx]["c_gnd_fF"].values)
    eval_c_mape = mape_med(corrected_c, valid.iloc[val_idx]["c_cpl_total_fF"].values)
    eval_t_mape = mape_med(corrected_t,
                            (valid.iloc[val_idx]["c_gnd_fF"] +
                             valid.iloc[val_idx]["c_cpl_total_fF"]).values)
    # baseline on same subset
    sub = valid.iloc[val_idx]
    base_g = mape_med(sub["pred_gnd_base"], sub["c_gnd_fF"])
    base_c = mape_med(sub["pred_cpl_base"], sub["c_cpl_total_fF"])
    base_t = mape_med(sub["pred_gnd_base"] + sub["pred_cpl_base"],
                      sub["c_gnd_fF"] + sub["c_cpl_total_fF"])
    print(f"\n  held-out 20% baseline:  gnd={base_g:.2f}% cpl={base_c:.2f}% tot={base_t:.2f}%",
          flush=True)
    print(f"  held-out 20% +residual: gnd={eval_g_mape:.2f}% cpl={eval_c_mape:.2f}% tot={eval_t_mape:.2f}%",
          flush=True)

    # 7) Save residual models
    rm_g.save_model(str(MODELS / "residual_gnd.json"))
    rm_c.save_model(str(MODELS / "residual_cpl.json"))
    meta_out = {
        "feats": residual_feats,
        "baseline_held_out": {"gnd": round(base_g, 3), "cpl": round(base_c, 3), "tot": round(base_t, 3)},
        "stacked_held_out": {"gnd": round(eval_g_mape, 3), "cpl": round(eval_c_mape, 3), "tot": round(eval_t_mape, 3)},
        "delta_tot_pp": round(eval_t_mape - base_t, 3),
    }
    (MODELS / "residual_stack_meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"\n>>> wrote {MODELS}/residual_{{gnd,cpl}}.json + meta", flush=True)
    print(f"   tot Δ on held-out: {eval_t_mape - base_t:+.3f}pp", flush=True)


if __name__ == "__main__":
    main()
