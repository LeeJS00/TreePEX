"""pex_tool.py — TreePEX orchestrator: features → SPEF → golden compare.

Run end-to-end pipeline for one design (intel22 or ASAP7 PDK).

  python pex_tool.py --design intel22_tv80s_f3
  python pex_tool.py --design intel22_nova_f3
  python pex_tool.py --all                              # both intel22 test designs
  python pex_tool.py --pdk asap7 --design asap7_tv80s_x1
  python pex_tool.py --pdk asap7 --all                  # both ASAP7 test designs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
REPORTS = ROOT / "outputs" / "reports"
PY = "/tool/etc/python/install/3.11.9/bin/python3"

sys.path.insert(0, str(SCRIPTS))
from pdk_paths import get_pdk  # noqa: E402


def run_one(design: str, pdk: str) -> dict:
    print(f"\n========== TreePEX ▶ pdk={pdk} {design} ==========\n", flush=True)
    timings = {"design": design, "pdk": pdk}
    pdk_args = ["--pdk", pdk]

    # Stage 1: inference
    t0 = time.time()
    rc = subprocess.run([PY, str(SCRIPTS / "02_inference.py"), "--design", design] + pdk_args).returncode
    timings["stage1_inference_s"] = round(time.time() - t0, 3)
    if rc != 0:
        timings["error"] = f"stage1 rc={rc}"; return timings

    # Stage 2: SPEF write
    t0 = time.time()
    rc = subprocess.run([PY, str(SCRIPTS / "03_write_spef.py"), "--design", design] + pdk_args).returncode
    timings["stage2_spef_write_s"] = round(time.time() - t0, 3)
    if rc != 0:
        timings["error"] = f"stage2 rc={rc}"; return timings

    # Stage 3: compare to golden (uses configs.config.resolve_golden_spef which
    # auto-handles both intel22 and ASAP7 naming conventions)
    t0 = time.time()
    rc = subprocess.run([PY, str(SCRIPTS / "04_compare_golden.py"), "--design", design]).returncode
    timings["stage3_compare_s"] = round(time.time() - t0, 3)
    if rc != 0:
        timings["error"] = f"stage3 rc={rc}"; return timings

    # Read report
    rep_json = REPORTS / f"{design}_report.json"
    if rep_json.exists():
        rep = json.loads(rep_json.read_text())
        timings.update({k: rep[k] for k in ("MAPE_tot_med", "MAPE_gnd_med",
                                              "MAPE_cpl_med", "R2_tot", "R2_gnd",
                                              "R2_cpl", "n_nets_compared",
                                              "spef_roundtrip_max_abs_err_fF")})
    timings["status"] = "OK"
    return timings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--pdk", default="intel22", choices=["intel22", "asap7"])
    args = p.parse_args()
    pdk = get_pdk(args.pdk)
    if args.all:
        designs = list(pdk.test_designs)
    elif args.design:
        designs = [args.design]
    else:
        print("Provide --design <name> or --all"); return 1

    summaries = []
    for d in designs:
        s = run_one(d, args.pdk)
        summaries.append(s)

    print("\n========== SUMMARY ==========")
    for s in summaries:
        if "error" in s:
            print(f"  {s['design']}: ERROR — {s['error']}")
            continue
        wall = s["stage1_inference_s"] + s["stage2_spef_write_s"] + s["stage3_compare_s"]
        print(f"  {s['design']}: tot_med={s['MAPE_tot_med']:.3f}%  gnd={s['MAPE_gnd_med']:.2f}%  "
              f"cpl={s['MAPE_cpl_med']:.2f}%  R²={s['R2_tot']:.4f}  "
              f"n={s['n_nets_compared']:,}  total_wall={wall:.2f}s "
              f"(infer={s['stage1_inference_s']:.2f}s + spef={s['stage2_spef_write_s']:.2f}s + cmp={s['stage3_compare_s']:.2f}s)")
    suffix = f"_{args.pdk}" if args.pdk != "intel22" else ""
    out_json = REPORTS / f"tool_summary{suffix}.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"\n>>> wrote {out_json}")


if __name__ == "__main__":
    raise SystemExit(main())
