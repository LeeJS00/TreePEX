# TreePEX — VLSI 기생 capacitance 예측기

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Input**: routed DEF + tech LEF + cell LEF + Liberty + layer.info  
**Output**: per-net ground + coupling capacitance SPEF (IEEE 1481-1999 호환, StarRC schema)  
**Model**: 5-seed Tweedie XGBoost prediction-mean ensemble on 67-D hand features  
**Hardware**: CPU only (16-core fork-Pool 권장; GPU 불필요)

---

## 1. 한 줄 소개

License-free PEX 도구로, StarRC와 **동일한 입력**(DEF / LEF / Liberty / layer stack)을 받아
**4.98% MAPE / 7 s (tv80s)** 또는 **5.28% MAPE / 70 s (nova)** 정확도/속도로 SPEF를 생성한다.
Cadence Innovus (6.96% / 분 단위)와 OpenRCX (8.83% / 분 단위) 대비 정확도 + 속도 모두 우위.

**Cross-PDK** (2026-05-14 추가): Intel 22nm + **ASAP7 7nm** 두 PDK 모두 지원.
ASAP7 5-seed Tweedie XGBoost ensemble — tv80s 6.68% / nova 7.03% MAPE.
**Methodology bit-identical** (same hyperparams, 67-D feature schema, 5 seeds).
Per-PDK 사용법: `python3 scripts/pex_tool.py --pdk asap7 --design asap7_gcd_x1` ([docs/CROSS_PDK.md](docs/CROSS_PDK.md)).

---

## 2. 빠른 시작 (Smoke test)

### 2.1 환경 준비

```bash
git clone https://github.com/LeeJS00/TreePEX.git
cd TreePEX
pip install -r requirements.txt
```

`requirements.txt`는 numpy, pandas, xgboost, numba (선택). Python 3.11+.

### 2.2 번들 데이터로 즉시 실행 (intel22_tv80s_f3)

```bash
python3 scripts/pex_cold.py --design intel22_tv80s_f3 --v3-algo njit
```

- 입력: `data/def/intel22_tv80s_f3.def` (3.9 MB, 동봉)
- PDK: `tool/pdk/22nm/` (~6 MB, 동봉)
- Golden SPEF: `data/golden_spef/intel22_tv80s_f3_starrc.spef.gz` (12 MB compressed, 자동 읽기)
- 출력: `outputs/spef/intel22_tv80s_f3_pred.spef`, `outputs/cold_reports/cold_summary.json`

End-to-end 약 50 s (16-worker CPU). 기대 결과:
```
intel22_tv80s_f3 | n=3,280 | MAPE tot=5.12% gnd=17.91% cpl=13.81% | R²_tot=0.9920
```

### 2.3 새로운 design 처리

```bash
# (1) DEF 파일 경로 설정 (선택 — 기본 data/def/ 사용)
export TREEPEX_DEF_DIR=/path/to/your/def
export TREEPEX_GOLDEN_DIR=/path/to/your/golden_spef  # 선택, golden 비교용

# (2) configs/config.py의 DESIGNS dict에 design 추가
#     "my_design": DEF_DIR / "my_design.def"

# (3) 실행
python3 scripts/pex_cold.py --design my_design --v3-algo njit
```

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

### 5.1 정확도

| PDK | Design | tot MAPE | gnd MAPE | cpl MAPE | R²_tot |
|---|---|---:|---:|---:|---:|
| **Intel 22nm** | **intel22_tv80s_f3** (3,280 nets) | **4.98 %** | 18.02 % | 13.27 % | 0.9940 |
| **Intel 22nm** | **intel22_nova_f3** (113,812 nets) | **5.28 %** | 17.40 % | 14.96 % | 0.9911 |
| **ASAP7 7nm** | **asap7_tv80s_x1** (3,328 nets) | **6.68 %** | 20.17 % | 9.10 % | 0.9801 |
| **ASAP7 7nm** | **asap7_nova_x1** (125,499 nets) | **7.03 %** | 21.22 % | 9.35 % | 0.9816 |

ASAP7 모델은 `models_asap7/` 에 동봉. 자세한 cross-PDK 분석은 [paper_benchmark/CROSS_PDK_TABLE.md](paper_benchmark/CROSS_PDK_TABLE.md) 참조.

### 5.2 Wall-clock (DEF → SPEF, 16-worker, Round 4 njit)

| Design | Pipeline | DEF parse | V3 features | V4 features | Inference | SPEF write |
|---|---:|---:|---:|---:|---:|---:|
| tv80s | **48.2 s** | 2.1 s | **3.5 s** | 38.2 s | 3.9 s | 0.1 s |
| nova  | **4,906 s** | 94 s | **2,352 s** | 2,384 s | 6.4 s | 3.8 s |

Round 0 (pre-patch) 대비 누적 가속: tv80s **3.52×**, nova **1.64×**. V3 단독 nova **2.38×**. 세부: [`COLD_START_SPEEDUP_REPORT.md`](COLD_START_SPEEDUP_REPORT.md).

### 5.3 경쟁 도구 비교

| Tool | License | tot MAPE | tv80s wall |
|---|---|---:|---:|
| **TreePEX** | **free (MIT)** | **4.98 %** | **7 s** |
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
