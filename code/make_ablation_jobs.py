import csv

from pathlib import Path

SEEDS = [42, 69, 123, 456, 789, 1024, 2048, 3333, 5555, 7777,

         9999, 100, 200, 300, 400, 500, 1234, 5678, 8888, 31337]

ARCHS = [("old", 32, 2, 0.3), ("new", 64, 2, 0.2)]

LR = 0.001

WD = 0.001

LOOKBACK_DEFAULT = 4

DATA_ROOT = "/home/chris/Documents/mlife/data/cleaned"

CANON_DS = f"{DATA_ROOT}/weight_windows_future_targets_all_weeks.csv"

def ds_for(window, stride):

    if window == 6 and stride == 1:

        return CANON_DS

    return f"{DATA_ROOT}/sweeps/sweep_window{window}_stride{stride}.csv"

rows = []

for w in [2, 3, 5, 8, 12]:

    for variant in ("supportwc", "wconly"):

        for arch, hd, nl, dp in ARCHS:

            for s in SEEDS:

                jid = f"swcA_w{w}_s1_h20_arch{arch}_{variant}_seed{s}"

                rows.append([jid, ds_for(w, 1), "next_20wk_weight_change",

                             variant, s, LOOKBACK_DEFAULT, hd, nl, dp, LR, WD])

for h in [4, 8, 12, 26]:

    for variant in ("supportwc", "wconly"):

        for arch, hd, nl, dp in ARCHS:

            for s in SEEDS:

                jid = f"swcB_w6_s1_h{h}_arch{arch}_{variant}_seed{s}"

                rows.append([jid, CANON_DS, f"next_{h}wk_weight_change",

                             variant, s, LOOKBACK_DEFAULT, hd, nl, dp, LR, WD])

for st in [2, 3, 5]:

    for variant in ("supportwc", "wconly"):

        for arch, hd, nl, dp in ARCHS:

            for s in SEEDS:

                jid = f"swcC_w6_s{st}_h20_arch{arch}_{variant}_seed{s}"

                rows.append([jid, ds_for(6, st), "next_20wk_weight_change",

                             variant, s, LOOKBACK_DEFAULT, hd, nl, dp, LR, WD])

TYPES = ["weight", "diet", "activity", "tip"]

for t in TYPES:

    for variant in (f"swc_no_{t}", f"swc_only_{t}"):

        for arch, hd, nl, dp in ARCHS:

            for s in SEEDS:

                jid = f"stype_w6_s1_h20_arch{arch}_{variant}_seed{s}"

                rows.append([jid, CANON_DS, "next_20wk_weight_change",

                             variant, s, LOOKBACK_DEFAULT, hd, nl, dp, LR, WD])

for lkb in [2, 6, 8, 12]:

    for variant in ("supportwc", "wconly"):

        for arch, hd, nl, dp in ARCHS:

            for s in SEEDS:

                jid = f"lkb_w6_s1_h20_arch{arch}_{variant}_lkb{lkb}_seed{s}"

                rows.append([jid, CANON_DS, "next_20wk_weight_change",

                             variant, s, lkb, hd, nl, dp, LR, WD])

out_path = Path("/Users/clee/Documents/Lab/mlife/code/overnight_jobs.csv")

with open(out_path, "w", newline="") as f:

    w = csv.writer(f)

    w.writerow(["job_id","dataset_path","target_col","variant","seed","lookback",

                "hidden_dim","num_layers","dropout","lr","weight_decay"])

    w.writerows(rows)

print(f"Wrote {len(rows)} jobs → {out_path}")

from collections import Counter

print("\nBy sweep prefix:")

for prefix, cnt in Counter(r[0].split("_")[0] for r in rows).items():

    print(f"  {prefix}: {cnt}")

print("\nBy variant:")

for v, cnt in Counter(r[3] for r in rows).items():

    print(f"  {v}: {cnt}")
