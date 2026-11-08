import csv
from collections import defaultdict
from pathlib import Path

p = Path(__file__).with_name("pd_fa_results_latest.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8")))
print(f"n_rows={len(rows)}")

exp = {"NUAA-SIRST": 86, "NUDT-SIRST": 664, "IRSTD-1k": 201, "SIRST-V2": 256}
bad = []
for r in rows:
    n = int(float(r["n_images"]))
    if n != exp[r["dataset"]]:
        bad.append((r["backbone"], r["dataset"], r["role"], n))
print("img_count_mismatch", bad)

paper_iou = {
    ("dna", "teacher"): {"NUAA-SIRST": 0.7171, "NUDT-SIRST": 0.8488, "IRSTD-1k": 0.6230, "SIRST-V2": 0.7112},
    ("dna", "student"): {"NUAA-SIRST": 0.7387, "NUDT-SIRST": 0.9226, "IRSTD-1k": 0.6706, "SIRST-V2": 0.7464},
    ("uiu", "teacher"): {"NUAA-SIRST": 0.7686, "NUDT-SIRST": 0.8596, "IRSTD-1k": 0.6376, "SIRST-V2": 0.8558},
    ("uiu", "student"): {"NUAA-SIRST": 0.7741, "NUDT-SIRST": 0.8471, "IRSTD-1k": 0.6439, "SIRST-V2": 0.8625},
    ("msh", "teacher"): {"NUAA-SIRST": 0.6909, "NUDT-SIRST": 0.7558, "IRSTD-1k": 0.6779, "SIRST-V2": 0.7320},
    ("msh", "student"): {"NUAA-SIRST": 0.6959, "NUDT-SIRST": 0.7729, "IRSTD-1k": 0.6653, "SIRST-V2": 0.7220},
}

print("\nIoU vs paper (pp):")
for r in rows:
    key = (r["backbone"], r["role"])
    if key not in paper_iou:
        continue
    ds = r["dataset"]
    got = float(r["IoU"])
    ref = paper_iou[key][ds]
    d = (got - ref) * 100
    flag = " !" if abs(d) > 1.0 else ""
    print(f"  {r['backbone']:7s} {r['role']:7s} {ds:12s} script={got:.4f} paper={ref:.4f} d={d:+.2f}pp{flag}")

sct = defaultdict(list)
for r in rows:
    if r["backbone"] == "sctrans":
        sct[r["role"]].append((r["dataset"], float(r["IoU"]), float(r["Pd"]), float(r["Fa"])))

print("\nSCTrans detail:")
for role in ("teacher", "student"):
    vals = sct[role]
    mean_iou = sum(v[1] for v in vals) / len(vals)
    mean_pd = sum(v[2] for v in vals) / len(vals)
    mean_fa = sum(v[3] for v in vals) / len(vals)
    print(f"  {role}: mean IoU={mean_iou:.4f} Pd={mean_pd:.4f} Fa={mean_fa:.2e}")
    for ds, iou, pd, fa in vals:
        print(f"    {ds:12s} IoU={iou:.4f} Pd={pd:.4f} Fa={fa:.2e}")

print("\nPd/Fa backbone means:")
for bb in ["dna", "uiu", "msh", "sctrans"]:
    t = [float(r["Pd"]) for r in rows if r["backbone"] == bb and r["role"] == "teacher"]
    s = [float(r["Pd"]) for r in rows if r["backbone"] == bb and r["role"] == "student"]
    ft = [float(r["Fa"]) for r in rows if r["backbone"] == bb and r["role"] == "teacher"]
    fs = [float(r["Fa"]) for r in rows if r["backbone"] == bb and r["role"] == "student"]
    print(f"  {bb}: Pd T/S={sum(t)/4:.4f}/{sum(s)/4:.4f}  Fa T/S={sum(ft)/4:.2e}/{sum(fs)/4:.2e}")

# cases where student Pd drops >2pp or Fa rises >2x
print("\nNotable Pd/Fa deltas (student - teacher):")
by = {(r["backbone"], r["dataset"], r["role"]): r for r in rows}
for bb in ["dna", "uiu", "msh", "sctrans"]:
    for ds in ["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1k", "SIRST-V2"]:
        t = by[(bb, ds, "teacher")]
        s = by[(bb, ds, "student")]
        dpd = (float(s["Pd"]) - float(t["Pd"])) * 100
        diou = (float(s["IoU"]) - float(t["IoU"])) * 100
        fa_t, fa_s = float(t["Fa"]), float(s["Fa"])
        ratio = fa_s / fa_t if fa_t > 0 else float("inf")
        mark = []
        if dpd < -1.5:
            mark.append(f"Pd{dpd:+.1f}pp")
        if diou < -2.0:
            mark.append(f"IoU{diou:+.1f}pp")
        if ratio > 2.0:
            mark.append(f"Fa×{ratio:.1f}")
        if mark:
            print(f"  {bb}/{ds}: {', '.join(mark)}  (Pd {t['Pd']}->{s['Pd']}, Fa {fa_t:.2e}->{fa_s:.2e})")
