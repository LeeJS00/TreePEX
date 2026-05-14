"""00_fit_fanout_proxy.py — fit XGB Tweedie + Ridge fanout proxies.

Both proxies live next to the main models so cold-start inference
(pex_cold.py) can fall back to a feature-derived estimate when the
SPEF-derived `fanout` column is unavailable.

Usage:
  python 00_fit_fanout_proxy.py --pdk intel22
  python 00_fit_fanout_proxy.py --pdk asap7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pdk_paths import get_pdk  # noqa: E402

FEATS = ["n_aggressor_nets", "n_cuboids", "n_edges_lt_1um", "n_edges_1_to_3um",
         "broadside_overlap_total_um2", "lateral_overlap_total_um2",
         "total_metal_area_um2", "bbox_xy_um2"]


def mape_med(p, g):
    p = np.maximum(p, 1.0)
    return float(np.median(np.abs(p - g) / np.maximum(np.abs(g), 1.0) * 100))


def mape_mean(p, g):
    p = np.maximum(p, 1.0)
    return float(np.mean(np.abs(p - g) / np.maximum(np.abs(g), 1.0) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", default="intel22", choices=["intel22", "asap7"])
    args = ap.parse_args()
    pdk = get_pdk(args.pdk)
    MODELS = pdk.models_dir
    MODELS.mkdir(parents=True, exist_ok=True)

    w = pd.read_csv(pdk.v3_features)
    train = w[w["split"] == "train"].reset_index(drop=True)
    valid = w[w["split"] == "valid"].reset_index(drop=True)
    test = w[w["split"] == "test"].reset_index(drop=True)
    print(f">>> pdk={pdk.name}  train={len(train):,}  valid={len(valid):,}  test={len(test):,}")

    X = train[FEATS].fillna(0.0).values
    y = train["fanout"].fillna(1.0).values.astype(float)

    # ---- XGB Tweedie (primary) ----
    import xgboost as xgb
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    t0 = time.time()
    m_xgb = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.5,
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=16, tree_method="hist")
    m_xgb.fit(X, y)
    print(f"XGB-Tweedie fit: {time.time()-t0:.1f}s")
    for split_name, sub in [("valid", valid), ("test", test)]:
        Xs = sub[FEATS].fillna(0.0).values
        p = np.maximum(m_xgb.predict(Xs), 1.0)
        print(f"  XGB-Tweedie  split={split_name:<5s}  MAPE_med={mape_med(p, sub.fanout.values):.1f}% "
              f"MAPE_mean={mape_mean(p, sub.fanout.values):.1f}%  "
              f"pred_mean={p.mean():.1f}  true_mean={sub.fanout.mean():.1f}", flush=True)

    out_xgb = MODELS / "fanout_proxy_xgb_tweedie.json"
    m_xgb.save_model(str(out_xgb))
    print(f"\nwrote {out_xgb}")

    # ---- Ridge (fallback) ----
    from sklearn.linear_model import Ridge
    Xt_log = np.log1p(X)
    yt_log = np.log1p(y)
    m_rg = Ridge(alpha=1.0, random_state=42)
    m_rg.fit(Xt_log, yt_log)
    print(f"Ridge fit done (log-log)")
    for split_name, sub in [("valid", valid), ("test", test)]:
        Xs = np.log1p(sub[FEATS].fillna(0.0).values)
        p = np.maximum(np.expm1(m_rg.predict(Xs)), 1.0)
        print(f"  Ridge        split={split_name:<5s}  MAPE_med={mape_med(p, sub.fanout.values):.1f}% "
              f"MAPE_mean={mape_mean(p, sub.fanout.values):.1f}%  "
              f"pred_mean={p.mean():.1f}  true_mean={sub.fanout.mean():.1f}", flush=True)
    ridge_payload = {
        "kind": "ridge_loglog",
        "feats": FEATS,
        "coef": m_rg.coef_.tolist(),
        "intercept": float(m_rg.intercept_),
    }
    out_rg = MODELS / "fanout_proxy_ridge.json"
    out_rg.write_text(json.dumps(ridge_payload, indent=2))
    print(f"wrote {out_rg}")

    meta = {
        "feats": FEATS,
        "kind": "xgb_tweedie",
        "model_file": "fanout_proxy_xgb_tweedie.json",
    }
    out_meta = MODELS / "fanout_proxy_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_meta}")


if __name__ == "__main__":
    main()
