# TreePEX — VLSI 기생 capacitance 예측기

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Input**: routed DEF + tech LEF + cell LEF + Liberty + layer.info  
**Output**: per-net ground + coupling capacitance SPEF (IEEE 1481-1999 호환, StarRC schema)  
**Model**: 5-seed Tweedie XGBoost prediction-mean ensemble on 67-D hand features  
**Hardware**: CPU only (16-core fork-Pool 권장; GPU 불필요)

---

## 1. 한 줄 소개

License-free PEX 도구. 사용자가 **DEF + LEF + layer.info만** 주면, 사전학습 모델로 SPEF를 cold from-scratch (no pre-computed features, no golden) 생성한다. **사용 시나리오: GitHub clone → 본인의 DEF → `pex_cold.py` 한 줄 → SPEF 파일**. 이 워크플로의 wall-clock + MAPE 가 paper-honest claim.

| PDK | Design | Cold wall (DEF→SPEF) | MAPE_tot | 비고 |
|---|---|---:|---:|---|
| **Intel 22nm** | intel22_tv80s_f3 | **50 s** | **5.13 %** | 번들 smoke (작동 검증) |
| **ASAP7 7nm** | asap7_gcd_x1 | **6.3 s** | 13.19 % | 번들 smoke (in-distribution) |
| **ASAP7 7nm** | asap7_tv80s_x1 | **62 s** | **11.23 %** | OOD test, cold-honest |

Cold-from-scratch 만 사용자 경험이고 paper claim. Pre-computed feature CSV 가 있을 때의 warm-eval 수치는 internal development benchmark (label-leaking — `fanout` 이 SPEF에서 추출됨) — `docs/CROSS_PDK.md` 참조.

---

## 2. 빠른 시작 (Smoke test)

### 2.1 환경 준비

```bash
git clone https://github.com/LeeJS00/TreePEX.git
cd TreePEX
pip install -r requirements.txt
```

`requirements.txt`는 numpy, pandas, xgboost, numba (선택). Python 3.11+.

### 2.2 번들 데이터로 즉시 실행

**Intel 22nm** (intel22_tv80s_f3):
```bash
python3 scripts/pex_cold.py --design intel22_tv80s_f3 --v3-algo njit
# → 50 s, MAPE tot=5.13% gnd=17.91% cpl=13.82% R²=0.9919
```
- 입력: `data/def/intel22_tv80s_f3.def` + `tool/pdk/22nm/` (모두 번들)
- 출력: `outputs/spef/intel22_tv80s_f3_cold_pred.spef`

**ASAP7 7nm** (asap7_gcd_x1, 번들 smoke):
```bash
python3 scripts/pex_cold.py --pdk asap7 --design asap7_gcd_x1 --v3-algo njit
# → 6.3 s, MAPE tot=13.2% (gcd is TRAIN-set, in-distribution sanity check)
```
- 입력: `data/def/asap7_gcd_x1.def` + `tool/pdk/7nm/` (모두 번들)
- 모델: `models_asap7/` (5-seed Tweedie XGBoost ensemble)

ASAP7 의 진짜 OOD test 인 `tv80s_x1`, `nova_x1` 은 DEF + golden SPEF 가 사이트에 있어야 하고 (용량 때문에 미번들), `--design asap7_tv80s_x1` + `TREEPEX_DEF_DIR`/`TREEPEX_GOLDEN_DIR` env override 로 실행.

### 2.3 새로운 design 처리 (GitHub clone 시 사용자 워크플로)

```bash
# (1) DEF 파일 경로 설정 (필수 — 본인 design 위치)
export TREEPEX_DEF_DIR=/path/to/your/def
export TREEPEX_GOLDEN_DIR=/path/to/your/golden_spef  # 선택, MAPE 측정용 (없으면 SPEF만 생성)

# (2) configs/config.py의 DESIGNS dict 또는 scripts/pex_cold.py
#     PDK_DESIGNS 등록 — 이름은 자유롭게 (e.g. "my_chip")

# (3) 실행 (PDK 선택)
python3 scripts/pex_cold.py --pdk intel22 --design my_chip --v3-algo njit
# 또는 ASAP7:
python3 scripts/pex_cold.py --pdk asap7 --design my_asap7_chip --v3-algo njit
```

**Paper-grade workflow** (사용자가 실제로 경험하는 것):
1. `git clone https://github.com/LeeJS00/TreePEX.git`
2. `cd TreePEX && pip install -r requirements.txt`
3. 본인 DEF/LEF 준비
4. `python3 scripts/pex_cold.py --pdk <intel22|asap7> --design <name>`
5. `outputs/spef/<name>_cold_pred.spef` 생성

이게 paper에서 claim 하는 **전체 wall-clock + MAPE** 의 기준.

### 2.4 V3 backend 선택

```bash
--v3-algo legacy      # numpy broadcast (가장 안정, 가장 느림)
--v3-algo auto        # threshold-gated (Round 3 기본; long-tail만 per-target)
--v3-algo per_target  # numpy per-target-cuboid 항상 사용
--v3-algo njit        # Numba JIT kernel (Round 4 추천, ~2× V3 빠름)
```

---

## 3. 입력 / 출력 명세

### 3.1 입력

| 항목 | 형식 | 위치 | 비고 |
|---|---|---|---|
| `<design>.def` | LEF/DEF v5.8 | `data/def/` 또는 `$TREEPEX_DEF_DIR` | net 라우팅 geometry |
| tech LEF (p1222_js.lef) | LEF | `tool/pdk/22nm/tech_lef/` | 번들 |
| cell LEF (b15_nn.lef) | LEF | `tool/pdk/22nm/cell_lef/` | 번들 |
| layer.info | text | `tool/pdk/22nm/layers/` | 번들 |
| golden SPEF (선택) | IEEE 1481-1999 + .gz | `data/golden_spef/` 또는 `$TREEPEX_GOLDEN_DIR` | 비교용 |

Env vars로 경로 override:
- `TREEPEX_TECH_LEF`, `TREEPEX_CELL_LEF`, `TREEPEX_LAYERS_INFO`
- `TREEPEX_DEF_DIR`, `TREEPEX_GOLDEN_DIR`
- `TREEPEX_TILE_CACHE_ROOT` (V4 H3 feature; site-specific)

### 3.2 출력

```
outputs/spef/<design>_pred.spef                    # IEEE 1481-1999 SPEF
outputs/predictions/<design>_pred.csv              # per-net 예측값
outputs/cold_reports/cold_summary.json             # timing + MAPE
outputs/cold_reports/<design>_treepex_per_net.csv  # 비교 (golden 존재 시)
outputs/cold_reports/<design>_treepex_summary.json
```

SPEF schema (per net):
```
*D_NET <net_name> <total_cap_fF>
  *CONN
    *I <pin1> ...
  *CAP
    1 <net_name>:0 <ground_cap_fF>
    2 <net_name>:1 <agg_net_1>:0 <coupling_cap_fF>
    ...
  *END
```

---

## 4. 67-D feature pack

자세한 설명: [`docs/FEATURE_SPEC.md`](docs/FEATURE_SPEC.md).

| 그룹 | 차원 | 설명 |
|---|---:|---|
| V3 wire geometry | 3 | n_cuboids, total_wire_length_um, total_metal_area_um2 |
| V3 net bbox | 3 | bbox_xy_um2, bbox_z_um, aspect_ratio |
| V3 layer histogram | 10 | layer_hist_M1..M8, layer_hist_M9_plus, n_layers_present |
| V3 aggressor pair | 12 | n_aggressor_nets, broadside/lateral overlap × {total, p95}, spacing × {min,p25,p50,p95}, n_edges × {lt1um,1to3um,3to4um} |
| V3 dielectric | 3 | eps_min, eps_max, eps_mean |
| V3 metal density | 3 | density_M1_M3, M4_M5, M6_plus |
| V3 VSS shielding | 5 | vss_n_cuboids, vss_total_area, vss_shield × {M1_M3, M4_M5, M6_plus} |
| V3 analytic priors | 2 | compact_gnd_estimate_fF (Sakurai-Tamaru), compact_cpl_estimate_total_fF |
| **V3 subtotal** | **41** | |
| V4 self check | 1 | target_n_cuboids_check |
| V4 aggressor counts | 5 | agg_n_distinct, agg_count × {above_z, below_z, within_{1,3,5}um} |
| V4 top-3 pair geometry | 18 | top{1,2,3} × {score, overlap_um2, min_xy_dist_um, mean_dz_um, agg_size_um2, layer_diff_flag} |
| V4 concentration | 1 | topk_score_concentration |
| **V4 subtotal** | **26** | |
| **TOTAL** | **67** | |

---

## 5. 성능 (golden = StarRC)

### 5.1 Cold from-scratch — **paper-grade (GitHub clone 시나리오)**

DEF → V3 + V4 H3 feature 추출 → 5-seed XGBoost inference → SPEF write. 사용자가 본인 머신에서 `pex_cold.py` 한 줄로 측정한 wall-clock + MAPE.

| PDK | Design | nets | Cold wall | MAPE_tot | MAPE_gnd | MAPE_cpl | R²_tot |
|---|---|---:|---:|---:|---:|---:|---:|
| **Intel 22nm** | intel22_tv80s_f3 | 3,280 | **49.7 s** | **5.13 %** | 17.91 % | 13.82 % | 0.9919 |
| **Intel 22nm** | intel22_nova_f3 | 113,812 | **4906 s** (~82 min) | **5.12 %** | 17.91 % | 13.81 % | 0.9920 |
| **ASAP7 7nm** | asap7_tv80s_x1 | 3,328 | **62.1 s** | **11.23 %** | 25.18 % | 13.80 % | 0.9655 |
| **ASAP7 7nm** | asap7_nova_x1 | 125,499 | _measuring_ | _measuring_ | — | — | — |

**ASAP7 cold MAPE 가 warm-eval (6.68%) 보다 4.55pp 높은 이유**: `fanout` 칼럼은 학습 데이터에서 SPEF의 coupled_caps 개수에서 추출됨 → warm-eval inference 는 golden SPEF 의 ground-truth fanout 을 보고 예측 (label-leaking). pex_cold 는 SPEF 가 없으므로 8-feature XGB-Tweedie proxy 로 대체 — ASAP7 proxy OOS MAPE 20.7%. cpl XGBoost 의 `fanout` feature_importance = 0.81 이라 proxy 정확도가 cpl→total 을 직격. Intel22 proxy 는 12% OOS 라 cold/warm gap 0.15pp 에 그침. (→ Future work: deterministic netlist-derived fanout)

자세한 cross-PDK + cold breakdown: [paper_benchmark/CROSS_PDK_TABLE.md](paper_benchmark/CROSS_PDK_TABLE.md).

### 5.2 Warm-eval — **internal development benchmark only**

사이트에 pre-computed V3 + V4 H3 feature CSV + golden SPEF 가 있을 때 `02_inference.py` 만 실행. **사용자가 보는 cold 워크플로가 아니라** 개발자가 모델만 격리해서 튜닝할 때 쓰는 internal tool. `fanout` 이 SPEF에서 직접 가져와 ASAP7 number 가 cold 보다 좋게 나옴 — paper claim 으로 사용 금지.

| PDK | Design | warm MAPE | 비고 |
|---|---|---:|---|
| intel22 | tv80s_f3 | 4.98 % | golden-fanout, dev-only |
| intel22 | nova_f3 | 5.28 % | golden-fanout, dev-only |
| ASAP7 | tv80s_x1 | 6.68 % | golden-fanout, dev-only |
| ASAP7 | nova_x1 | 7.03 % | golden-fanout, dev-only |

### 5.3 Wall-clock breakdown (cold, 16-worker, Round 4 njit)

| Design | Pipeline | DEF parse | V3 features | V4 features | Inference | SPEF write |
|---|---:|---:|---:|---:|---:|---:|
| intel22_tv80s_f3 | **49.7 s** | 2.1 s | **4.8 s** | 38.0 s | 4.5 s | 0.12 s |
| intel22_nova_f3 | **4906 s** | 94 s | **2352 s** | 2384 s | 6.4 s | 3.8 s |
| asap7_tv80s_x1 | **62.1 s** | 1.6 s | **8.0 s** | **48.4 s** (78%) | 3.9 s | 0.11 s |

Round 0 (pre-patch) 대비 누적 V3 가속: tv80s **3.52×**, nova **2.38×**. ASAP7 의 V4 H3 stage 가 cold 의 78% — 향후 `<design>_v4_pernet.cubs.npy` indexed cache 빌드 (intel22 fastpath mirror) 로 10-30× V4 speedup 가능. 세부: [`COLD_START_SPEEDUP_REPORT.md`](COLD_START_SPEEDUP_REPORT.md).

### 5.4 경쟁 도구 비교 (intel22 cold)

| Tool | License | tot MAPE | tv80s wall |
|---|---|---:|---:|
| **TreePEX cold** | **free (MIT)** | **5.13 %** | **50 s** |
| v12 PINN | free | 8.23 % | 10 s |
| Cadence Innovus | commercial | 6.96 % | ~120 s |
| OpenRCX | free | 8.83 % | ~60 s |
| StarRC (oracle) | commercial | reference | minutes |

---

## 6. 디렉 구조

```
TreePEX/
├── README.md                       # (this file)
├── REPORT.md                       # paper-style 종합 보고서
├── COLD_START_SPEEDUP_REPORT.md    # Round 1-4 엔지니어링 deep-dive
├── LICENSE                         # MIT
├── requirements.txt
├── run.sh                          # bash wrapper for new designs
├── configs/
│   └── config.py                   # 경로 + 환경변수 중앙화
├── docs/
│   ├── FEATURE_SPEC.md             # 67-D feature 정의 + 출처
│   ├── FEATURE_SPEEDUP_PLAN.md     # Round 1-4 working plan (history)
│   ├── COLD_START_REPORT.md        # 초기 cold-start report (legacy)
│   └── PROGRESS_REPORT.md          # 초기 progress (legacy)
├── models/                         # 13 production weight files + cards
│   ├── MODEL_CARD.md
│   ├── FEATURE_ORDER.txt
│   ├── tweedie_{gnd,cpl}_seed{42,0,1,2,3}.json    # 10 main predictors
│   └── fanout_proxy_{meta,xgb_tweedie,ridge}.json # 3 essential aux
├── scripts/
│   ├── pex_cold.py                 # ★ DEF→SPEF end-to-end (canonical entry)
│   ├── pex_tool.py                 # split-stage runner (cached features)
│   ├── 01_train_save_models.py     # offline 5-seed Tweedie training
│   ├── 02_inference.py             # split stage 1
│   ├── 03_write_spef.py            # split stage 2
│   ├── 04_compare_golden.py        # split stage 3
│   ├── dump_features.py            # diagnostic feature dump
│   ├── compare_features.py         # diagnostic drift report
│   └── summarize_cold_results.py
├── src/
│   ├── preprocessing/              # DEF/LEF/layer.info parsers
│   └── physics/                    # BEOL material stack
├── tool/pdk/22nm/                  # PDK assets (bundled, ~6 MB)
│   ├── tech_lef/p1222_js.lef
│   ├── cell_lef/b15_nn.lef
│   └── layers/layers.info
├── data/                           # smoke-test data (bundled)
│   ├── def/intel22_tv80s_f3.def    # 3.9 MB DEF
│   └── golden_spef/intel22_tv80s_f3_starrc.spef.gz  # 12 MB compressed
├── paper_benchmark/                # PAPER_TABLE.md + bench scripts
├── presentation/                   # figures + PPT
└── outputs/                        # runtime artifacts (gitignored)
    ├── spef/
    ├── predictions/
    ├── reports/
    └── cold_reports/
```

---

## 7. 재학습 (rare; 새 PDK 또는 새 design 추가 시)

기본 학습 파이프라인은 외부 cached feature CSV에 의존:

```bash
# 외부 feature CSV 경로 지정
export TREEPEX_V3_FEATURES=/path/to/all_designs_features.csv
export TREEPEX_V4_NEW_FEATS=/path/to/new_features_with_ids.csv
python3 scripts/01_train_save_models.py
```

학습 hyperparameters (`01_train_save_models.py`):
- objective: `reg:tweedie`, variance_power=1.5
- depth=8, n_est=500, lr=0.05, subsample=0.8, colsample_bytree=0.8
- early_stopping=100 rounds on validation MAPE
- seeds: 42, 0, 1, 2, 3

> 동일 PDK (intel22) 사용 시 retrain 불필요 — `models/`에 동봉된 10개 weight 그대로 사용.

---

## 8. 알려진 한계 (Known limits)

- **MAPE 천장 ~4.66%**: hand-feature 4-way oracle bound. 그 이하로 가려면 새 input modality (voxel CNN over rasterized routing) 또는 다른 paradigm 필요.
- **DEF/LEF/Liberty 입력의 정보 천장**: gnd MAPE ~17-18%는 representation-bound, NOT input-bound. StarRC도 동일 입력을 받지만 NXTGRD pattern-lookup + 3D field solver로 정확도를 얻음.
- **per-pin cap 분포 없음**: lumped per-net `*CAP`만 emit.
- **R (parasitic resistance) 미포함**: TreePEX는 C only.
- **22nm intel22 PDK 전용**: 7nm ASAP7 cross-PDK 전이는 미검증.
- **번들 데이터는 tv80s 1 design뿐**: nova (164 MB DEF + 2.9 GB golden SPEF)는 너무 크므로 외부에서 확보 필요.

---

## 9. 라이선스 + 인용

- TreePEX source + trained models: MIT License (`LICENSE`)
- 번들 PDK (`tool/pdk/22nm/`): 원본 PDK 배포 조건을 따름 (research use only)
- 의존성: numpy, pandas, xgboost (Apache 2.0), numba (BSD-2)

문의: [github.com/LeeJS00/TreePEX](https://github.com/LeeJS00/TreePEX) issue tracker.
