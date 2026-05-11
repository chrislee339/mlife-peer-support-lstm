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

VARIANT = "swc_only_activity"  

rows = []

for w in [2, 3, 5, 8, 12]:

    for arch, hd, nl, dp in ARCHS:

        for s in SEEDS:

            jid = f"actA_w{w}_s1_h20_arch{arch}_{VARIANT}_seed{s}"

            rows.append([jid, ds_for(w, 1), "next_20wk_weight_change",

                         VARIANT, s, LOOKBACK_DEFAULT, hd, nl, dp, LR, WD])

for h in [4, 8, 12, 26]:

    for arch, hd, nl, dp in ARCHS:

        for s in SEEDS:

            jid = f"actB_w6_s1_h{h}_arch{arch}_{VARIANT}_seed{s}"

            rows.append([jid, CANON_DS, f"next_{h}wk_weight_change",

                         VARIANT, s, LOOKBACK_DEFAULT, hd, nl, dp, LR, WD])

for st in [2, 3, 5]:

    for arch, hd, nl, dp in ARCHS:

        for s in SEEDS:

            jid = f"actC_w6_s{st}_h20_arch{arch}_{VARIANT}_seed{s}"

            rows.append([jid, ds_for(6, st), "next_20wk_weight_change",

                         VARIANT, s, LOOKBACK_DEFAULT, hd, nl, dp, LR, WD])

for lkb in [2, 6, 8, 12]:

    for arch, hd, nl, dp in ARCHS:

        for s in SEEDS:

            jid = f"actL_w6_s1_h20_arch{arch}_{VARIANT}_lkb{lkb}_seed{s}"

            rows.append([jid, CANON_DS, "next_20wk_weight_change",

                         VARIANT, s, lkb, hd, nl, dp, LR, WD])

out_path = Path("/Users/clee/Documents/Lab/mlife/code/activity_jobs.csv")

with open(out_path, "w", newline="") as f:

    w = csv.writer(f)

    w.writerow(["job_id","dataset_path","target_col","variant","seed","lookback",

                "hidden_dim","num_layers","dropout","lr","weight_decay"])

    w.writerows(rows)

print(f"Wrote {len(rows)} jobs → {out_path}")

import random

random.seed(0)

random.shuffle(rows)

half = len(rows) // 2

for gpu, sub in enumerate([rows[:half], rows[half:]]):

    p = Path(f"/Users/clee/Documents/Lab/mlife/code/activity_g{gpu}.csv")

    with open(p, "w", newline="") as f:

        w = csv.writer(f)

        w.writerow(["job_id","dataset_path","target_col","variant","seed","lookback",

                    "hidden_dim","num_layers","dropout","lr","weight_decay"])

        w.writerows(sub)

    print(f"  GPU{gpu}: {len(sub)} jobs → {p}")
