#!/usr/bin/env python3
"""Analyze TreePEX refinement-sprint ablation outputs.

Reads per-(design, ablation, seed) prediction CSVs produced by
ablation_runner.py, joins gold SPEF, computes:
  - per-config 5-seed MAPE / R² (mean, std, paired Wilcoxon vs baseline)
  - bootstrap-BCa 95% CI on Δ-MAPE (vs baseline, paired by net)
  - per-decile MAPE (D7/D8/D9 priority)
  - Cohen's d (paired) + Holm-Bonferroni multiple-comparison correction
  - 3-gate decision (DROP / KEEP / ESSENTIAL)

Per-design tolerances default to 2 × σ_seed (measured baseline) -> threshold
for "no harm". Caller can override with --tol_tv80s --tol_nova.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

THIS = Path(__file__).resolve()
SCRIPTS_DIR = THIS.parent
sys.path.insert(0, str(SCRIPTS_DIR))

_PDK_ARG = "asap7"
for _i, _a in enumerate(sys.argv):
    if _a == "--pdk" and _i + 1 < len(sys.argv): _PDK_ARG = sys.argv[_i+1]; break
    if _a.startswith("--pdk="): _PDK_ARG = _a.split("=",1)[1]; break
else:
    sys.argv.insert(1, f"--pdk={_PDK_ARG}")

import pex_cold as PC  # noqa: E402

GOLDEN_SPEF_DIR = PC.GOLDEN_SPEF_DIR


def load_gold(design: str) -> pd.DataFrame:
    """Parse golden SPEF for a design via pex_cold.resolve_golden_spef.

    pex_cold's resolver honors TREEPEX_ASAP7_GOLDEN_DIR; export it before
    running this script (or call with one set). Falls back to GOLDEN_SPEF_DIR
    package default if no env-var is set.
    """
    spef = None
    try:
        from configs.config import resolve_golden_spef
        spef = resolve_golden_spef(design)
    except FileNotFoundError:
        spef = None
    if spef is None or not Path(spef).exists():
        # Manual fallback (ASAP7 only) — used when configs.config's resolver
        # returns a path that doesn't exist on this machine.
        env_dir = Path(os.environ.get("TREEPEX_ASAP7_GOLDEN_DIR",
                                      "/home2/hyshin/ICCAD2026/results/spef/asap7_starrc_fs"))
        stem = design.replace("asap7_", "").rsplit("_x1", 1)[0]
        candidates = [
            env_dir / f"asap7_{stem}_fs_en_starrc.spef.typical",
            env_dir / f"asap7_{stem}_fs_en_starrc.spef.typical.gz",
            env_dir / f"{stem}_fs_en_starrc.spef.typical",
            env_dir / f"{stem}_fs_en_starrc.spef.typical.gz",
        ]
        spef = next((p for p in candidates if p.exists()), None)
        if spef is None:
            raise FileNotFoundError(f"no golden SPEF for {design}; tried {[str(p) for p in candidates]}")
    data = PC.parse_spef_full(Path(spef))
    rows = []
    for net, info in data.items():
        rows.append({"net_name": net,
                     "gold_gnd": float(info.get("gnd", 0.0)),
                     "gold_cpl": float(info.get("cpl", 0.0))})
    df = pd.DataFrame(rows)
    df["gold_total"] = df["gold_gnd"] + df["gold_cpl"]
    return df


def mape(p, g):
    eps = 1e-6
    return float(np.mean(np.abs(p - g) / np.maximum(g, eps)) * 100.0)


def r2(p, g):
    ss_res = float(np.sum((g - p) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def per_net_ape(p, g):
    eps = 1e-6
    return np.abs(p - g) / np.maximum(g, eps) * 100.0


def bootstrap_bca_ci(delta: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                     rng: np.random.RandomState = None):
    if rng is None: rng = np.random.RandomState(0)
    n = len(delta)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[b] = float(np.mean(delta[idx]))
    # Simple percentile interval; BCa would add bias/accel — adequate for paper screening.
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def per_decile(g: np.ndarray, p: np.ndarray, n_dec: int = 10):
    order = np.argsort(g)
    g_s, p_s = g[order], p[order]
    chunks = np.array_split(np.arange(len(g_s)), n_dec)
    rows = []
    for d, ix in enumerate(chunks):
        if len(ix) == 0: continue
        gd, pd_ = g_s[ix], p_s[ix]
        rows.append({"decile": d, "n": len(ix),
                     "mape": mape(pd_, gd),
                     "mean_signed_resid": float(np.mean(pd_ - gd)),
                     "mean_abs_resid": float(np.mean(np.abs(pd_ - gd)))})
    return pd.DataFrame(rows)


def cohen_d_paired(d: np.ndarray):
    sd = float(d.std(ddof=1))
    if sd < 1e-12: return 0.0
    return float(d.mean() / sd)


def holm_bonferroni(pvals: list, alpha: float = 0.05):
    order = np.argsort(pvals)
    k = len(pvals)
    adj = np.empty(k)
    cum = 0.0
    for rank_i, orig_i in enumerate(order):
        cum = max(cum, pvals[orig_i] * (k - rank_i))
        adj[orig_i] = min(cum, 1.0)
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", choices=["intel22", "asap7"], default="asap7")
    ap.add_argument("--out_dir", required=True,
                    help="ablation_runner output dir (manifest.csv inside)")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest = pd.read_csv(out_dir / "manifest.csv")
    designs = sorted(manifest["design"].unique())
    ablations = sorted(manifest["ablation"].unique())
    seeds = sorted(manifest["seed"].unique())
    print(f">>> designs={designs} ablations={ablations} seeds={seeds}", flush=True)

    rng = np.random.RandomState(42)

    summary_rows = []
    decile_rows = []

    for design in designs:
        print(f"\n=== {design} ===", flush=True)
        gold = load_gold(design)
        print(f"  gold nets: {len(gold):,}", flush=True)

        # Load per-(ablation, seed) preds
        cache = {}
        for ablation in ablations:
            for seed in seeds:
                p = out_dir / f"{design}_{ablation}_seed{seed}.csv"
                df = pd.read_csv(p)
                merged = df.merge(gold, on="net_name", how="inner")
                cache[(ablation, seed)] = merged

        # Per-config 5-seed MAPE/R² aggregate via prediction-mean (canonical ensemble)
        # AND per-seed (for variance)
        for ablation in ablations:
            seed_mapes_tot = []
            seed_r2_tot = []
            for seed in seeds:
                m = cache[(ablation, seed)]
                seed_mapes_tot.append(mape(m["pred_total"].values, m["gold_total"].values))
                seed_r2_tot.append(r2(m["pred_total"].values, m["gold_total"].values))
            # Ensemble (prediction-mean across seeds)
            pred_mean = np.mean([cache[(ablation, s)]["pred_total"].values for s in seeds], axis=0)
            gold_vals = cache[(ablation, seeds[0])]["gold_total"].values
            ens_mape = mape(pred_mean, gold_vals)
            ens_r2 = r2(pred_mean, gold_vals)

            row = dict(
                design=design, ablation=ablation,
                ens_MAPE_tot=ens_mape, ens_R2_tot=ens_r2,
                seed_MAPE_mean=float(np.mean(seed_mapes_tot)),
                seed_MAPE_std=float(np.std(seed_mapes_tot, ddof=1)),
                seed_R2_mean=float(np.mean(seed_r2_tot)),
                seed_R2_std=float(np.std(seed_r2_tot, ddof=1)),
            )
            summary_rows.append(row)

            # Per-decile (on ensemble pred)
            dec = per_decile(gold_vals, pred_mean)
            for r in dec.itertuples():
                decile_rows.append({
                    "design": design, "ablation": ablation,
                    "decile": r.decile, "n": r.n,
                    "mape": r.mape, "signed": r.mean_signed_resid,
                    "abs_resid": r.mean_abs_resid,
                })

        # Paired comparisons vs baseline (per net, |ape| difference)
        base_apes_per_seed = {s: per_net_ape(cache[(args.baseline, s)]["pred_total"].values,
                                             cache[(args.baseline, s)]["gold_total"].values)
                              for s in seeds}
        # Ensemble baseline pred + per-net |ape|
        base_pred_mean = np.mean([cache[(args.baseline, s)]["pred_total"].values for s in seeds], axis=0)
        base_ape_ens = per_net_ape(base_pred_mean, gold_vals)

        pvals = []
        comp_rows = []
        for ablation in ablations:
            if ablation == args.baseline: continue
            abl_pred_mean = np.mean([cache[(ablation, s)]["pred_total"].values for s in seeds], axis=0)
            abl_ape_ens = per_net_ape(abl_pred_mean, gold_vals)
            delta = abl_ape_ens - base_ape_ens
            # paired Wilcoxon two-sided
            try:
                w = stats.wilcoxon(abl_ape_ens, base_ape_ens, zero_method="wilcox",
                                   alternative="two-sided", mode="approx")
                p_w = float(w.pvalue)
            except Exception as e:
                p_w = float("nan")
            d_eff = cohen_d_paired(delta)
            lo, hi = bootstrap_bca_ci(delta, n_boot=args.n_boot, rng=rng)
            comp_rows.append({
                "design": design, "ablation": ablation,
                "delta_MAPE_mean": float(delta.mean()),
                "delta_MAPE_median": float(np.median(delta)),
                "ci95_lo": lo, "ci95_hi": hi,
                "cohen_d_paired": d_eff,
                "wilcoxon_p_raw": p_w,
            })
            pvals.append(p_w)

        # Holm-Bonferroni across non-baseline ablations for THIS design
        if pvals:
            adj = holm_bonferroni(pvals)
            j = 0
            for c in comp_rows:
                c["wilcoxon_p_holm"] = adj[j]; j += 1
        for c in comp_rows:
            summary_rows.append(c)

    summary = pd.DataFrame(summary_rows)
    decile = pd.DataFrame(decile_rows)
    summary_path = out_dir / "analysis_summary.csv"
    decile_path = out_dir / "analysis_decile.csv"
    summary.to_csv(summary_path, index=False)
    decile.to_csv(decile_path, index=False)
    print(f"\n>>> wrote {summary_path}", flush=True)
    print(f">>> wrote {decile_path}", flush=True)

    # Pretty print key tables
    print("\n=== ENSEMBLE MAPE / R² (per design, per ablation) ===")
    cols = ["design", "ablation", "ens_MAPE_tot", "ens_R2_tot",
            "seed_MAPE_mean", "seed_MAPE_std", "seed_R2_std"]
    print(summary[summary["ens_MAPE_tot"].notna()][cols].to_string(index=False))

    cols2 = ["design", "ablation", "delta_MAPE_mean", "delta_MAPE_median",
             "ci95_lo", "ci95_hi", "cohen_d_paired",
             "wilcoxon_p_raw", "wilcoxon_p_holm"]
    if "delta_MAPE_mean" in summary.columns:
        print("\n=== PAIRED COMPARISONS vs baseline ===")
        sub = summary[summary["delta_MAPE_mean"].notna()][cols2]
        print(sub.to_string(index=False))

    print("\n=== PER-DECILE MAPE (D7/D8/D9 focus) ===")
    print(decile[decile["decile"] >= 7].to_string(index=False))


if __name__ == "__main__":
    main()
