# Cross-PDK support (intel22 + ASAP7 7nm)

TreePEX ships with two pre-trained 5-seed Tweedie XGBoost ensembles:

| PDK | Model dir | TEST mean MAPE (tot) | TRAIN designs |
|---|---|---:|---|
| Intel 22nm (canonical) | `models/` | **5.13 %** (tv80s 4.98 / nova 5.28) | 9 |
| ASAP7 7nm | `models_asap7/` | **6.86 %** (tv80s 6.68 / nova 7.03) | 9 |

Methodology is **bit-identical** between PDKs (same architecture, 67-D feature
schema, 5 seeds, `depth=8, n_est=500, vp=1.5`). The only changes are:
- model weights (per-PDK retrain on per-PDK TRAIN_9 SPEFs)
- PDK files (`tool/pdk/{22nm, 7nm}/`)
- design list (intel22_* vs asap7_*_x1)

## Running the paper benchmark

The four-stage scripts (`01_train_save_models`, `02_inference`, `03_write_spef`,
`04_compare_golden`, `pex_tool`) all take a `--pdk {intel22, asap7}` flag.

### Intel 22nm (default)

```bash
# Bundled smoke (uses pre-trained models + bundled tv80s DEF + gz SPEF):
python3 scripts/pex_tool.py --design intel22_tv80s_f3

# OR explicit:
python3 scripts/pex_tool.py --pdk intel22 --design intel22_tv80s_f3
python3 scripts/pex_tool.py --pdk intel22 --all   # both intel22 test designs
```

### ASAP7 7nm

```bash
# Bundled smoke (uses pre-trained ASAP7 models + bundled gcd_x1 DEF + gz SPEF):
python3 scripts/pex_tool.py --pdk asap7 --design asap7_gcd_x1

# Test designs (require external feature CSV + golden SPEFs):
python3 scripts/pex_tool.py --pdk asap7 --all   # tv80s_x1 + nova_x1
```

## Feature CSV requirements (paper-benchmark scripts)

The `02_inference.py` script reads pre-computed feature CSVs:
- **V3 base** (41-D per-net feature CSV): geometry + aggressor stats + analytic estimates
- **V4 H3** (26-D per-net top-K aggressor CSV): coupling-relevant local geometry

For deployment without these CSVs, generate them via:

```bash
# 1. Build cuboid tile cache (offline, one-time per design)
python3 ../PINNPEX/scripts/build_dataset_multi.py --config config_asap7

# 2. Extract V3 base features (parallel, ~30-60 min per PDK)
python3 ../PINNPEX/pex_v6/scripts/00_build_asap7_features.py

# 3. Extract V4 H3 features (32-worker, ~90 min)
python3 ../PINNPEX/archive/pex_v4/scripts/29_extract_new_features.py \
  --manifest-csv /data/PINNPEX/data/processed_v3/asap7/dataset_manifest.csv \
  --data-root   /data/PINNPEX/data/processed_v3/asap7 \
  --out-csv     inputs/asap7_new_features_with_ids.csv \
  --n-workers   32
```

Or override paths via environment variables:

```bash
export TREEPEX_ASAP7_V3_FEATURES=/path/to/asap7/all_designs.csv
export TREEPEX_ASAP7_V4_NEW_FEATS=/path/to/asap7_new_features_with_ids.csv
export TREEPEX_ASAP7_GOLDEN_DIR=/path/to/asap7_starrc_spefs
```

## ASAP7 cross-PDK paper results

See `paper_benchmark/CROSS_PDK_TABLE.md` for full per-design + per-channel
breakdown including per-cap-decile metrics, R² values, and the methodology
lockdown statement.

### Headline

| PDK | tv80s tot MAPE | nova tot MAPE | tv80s e2e | nova e2e |
|---|---:|---:|---:|---:|
| **intel22** (canonical) | **4.98 %** | **5.28 %** | 10.2 s | 70.6 s |
| **ASAP7** (this work) | **6.68 %** | **7.03 %** | 10.4 s | 34.0 s |

Both PDKs clear the **Phase F minimum criterion** (≤ 7.0 % total MAPE).
Per-channel breakdown shows ASAP7 has *better* coupling MAPE (9.10/9.35 %) but
*worse* ground MAPE (20.17/21.22 %) compared to intel22 (gnd 18.02/17.40,
cpl 13.27/14.96).

## ASAP7 PDK files bundled

| File | Source | Size |
|---|---|---:|
| `tool/pdk/7nm/lef/asap7_tech_1x_201209_JS.lef` | ASAP7 academic PDK (Synopsys/ARM) | 20 KB |
| `tool/pdk/7nm/lef/asap7sc7p5t_28_R_1x_220121a.lef` | OpenROAD ASAP7 platform v28 R-only | 391 KB |
| `tool/pdk/7nm/layers/layers.info` | ASAP7 BEOL layer stack (M1–M9 + Pad) | 4 KB |
| `data/def/asap7_gcd_x1.def` | ASAP7-routed gcd (smoke design) | 224 KB |
| `data/golden_spef/asap7_gcd_fs_en_starrc.spef.typical.gz` | StarRC golden (typical corner) | 530 KB |

## Known limitations

1. **`pex_cold.py` is intel22-only**. The cold-start single-script pipeline
   (DEF → tiling → features → SPEF) does not yet support ASAP7 PDK paths. Use
   the paper-benchmark `pex_tool.py` flow with pre-computed feature CSVs.

2. **V4 H3 feature CSV (134 MB)** is not bundled. Generate locally per
   instructions above or symlink an existing copy into `inputs/`.

3. **Layer-feature mapping caveat**: The base feature extractor maps several
   layer-keyed features (`layer_hist_M*`, `vss_shield_*`, `density_M*`) into
   the M1 bucket on ASAP7 due to a segment-name→lvl_idx resolution issue.
   The current 6.86 % mean MAPE is the floor with that caveat; fixing the
   mapping is expected to push ASAP7 closer to intel22's 5 % range. The trees
   compensate via `vss_n_cuboids`, `total_wire_length_um`, `compact_*_estimate`
   features and the 26 H3 features which carry most of the signal.
