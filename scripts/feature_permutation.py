#!/usr/bin/env python3
"""F2 — Permutation feature importance on cached cold features.

For each (PDK, design): baseline MAPE = 5-seed ensemble inference on
cached cold features. For each of 67-D features: shuffle that column,
re-predict, compute Δ MAPE. Δ MAPE > 0 = harmful drop (essential); Δ ≈ 0 =
redundant feature; Δ < 0 = noise.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
SCRIPTS = THIS.parent
sys.path.insert(0, str(SCRIPTS))

_PDK_ARG = "asap7"
for _i, _a in enumerate(sys.argv):
    if _a == "--pdk" and _i+1 < len(sys.argv): _PDK_ARG = sys.argv[_i+1]; break
    if _a.startswith("--pdk="): _PDK_ARG = _a.split("=",1)[1]; break
else:
    sys.argv.insert(1, f"--pdk={_PDK_ARG}")

import pex_cold as PC  # noqa

SEEDS = list(PC.SEEDS)
FEATURE_COLS_67 = list(PC.FEATURE_COLS_67)


def load_models(mdir: Path):
    import xgboost as xgb
    out = {}
    for s in SEEDS:
        mg = xgb.XGBRegressor(); mg.load_model(str(mdir/f"tweedie_gnd_seed{s}.json"))
        mc = xgb.XGBRegressor(); mc.load_model(str(mdir/f"tweedie_cpl_seed{s}.json"))
        out[s] = (mg, mc)
    return out


def load_gold(design):
    candidates = [
        Path(f"/home2/hyshin/ICCAD2026/results/spef/asap7_starrc_fs/asap7_{design.replace('asap7_','').replace('_x1','')}_fs_en_starrc.spef.typical"),
        Path(f"/home2/hyshin/ICCAD2026/results/spef/intel22_starrc_fs/intel22_{design.replace('intel22_','').replace('_f3','')}_f3_starrc.spef"),
        Path(f"/home/jslee/projects/TreePEX/data/golden_spef/{design}_starrc.spef.gz"),
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(f"no golden for {design}: {candidates}")
    data = PC.parse_spef_full(p)
    rows = [{"net_name": n, "gold_total": float(info.get("gnd",0)) + float(info.get("cpl",0))}
            for n, info in data.items()]
    return pd.DataFrame(rows)


def mape_med(p, g, eps=1e-3):
    return float(np.median(np.abs(p-g) / np.maximum(np.abs(g), eps) * 100))


def predict_ensemble_gc(models_dict, X):
    pg = np.mean([models_dict[s][0].predict(X).clip(0) for s in SEEDS], axis=0)
    pc = np.mean([models_dict[s][1].predict(X).clip(0) for s in SEEDS], axis=0)
    return pg + pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", choices=["intel22","asap7"], default="asap7")
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--canonical_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_shuffle", type=int, default=3,
                    help="number of random shuffles per feature (averaged)")
    args = ap.parse_args()

    canon_dir = Path(args.canonical_dir)
    feat_order_file = canon_dir / "FEATURE_ORDER.txt"
    feats = feat_order_file.read_text().strip().split("\n") if feat_order_file.exists() else FEATURE_COLS_67
    print(f">>> feature schema: {len(feats)}-D ({canon_dir})", flush=True)

    base_models = load_models(canon_dir)

    rng = np.random.RandomState(42)
    out_rows = []

    for design in args.designs:
        feat_path = PC.COLD_REPORT_DIR / f"{design}_cold_features.parquet"
        feat_df = pd.read_parquet(feat_path).dropna(subset=feats).reset_index(drop=True)
        gold = load_gold(design)
        merged = feat_df.merge(gold, on="net_name", how="inner").reset_index(drop=True)
        print(f"  {design}: {len(merged):,} nets", flush=True)

        X = merged[feats].astype(np.float32).values
        gold_total = merged["gold_total"].values
        base_pred = predict_ensemble_gc(base_models, X)
        base_mape = mape_med(base_pred, gold_total)
        out_rows.append({"design": design, "feature": "_BASELINE_",
                         "delta_mape_med": 0.0, "abs_delta": 0.0, "base_mape_med": base_mape})
        print(f"    baseline MAPE_med = {base_mape:.3f}%", flush=True)

        for fi, fname in enumerate(feats):
            deltas = []
            for sh in range(args.n_shuffle):
                X_perm = X.copy()
                idx = rng.permutation(len(X_perm))
                X_perm[:, fi] = X_perm[idx, fi]
                pred = predict_ensemble_gc(base_models, X_perm)
                deltas.append(mape_med(pred, gold_total) - base_mape)
            d_mean = float(np.mean(deltas))
            d_std = float(np.std(deltas))
            out_rows.append({"design": design, "feature": fname,
                             "delta_mape_med": d_mean, "abs_delta": abs(d_mean),
                             "delta_std": d_std, "base_mape_med": base_mape})
            if (fi+1) % 10 == 0:
                print(f"    {fi+1}/{len(feats)} done", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f">>> wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
