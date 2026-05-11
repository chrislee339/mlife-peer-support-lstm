import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GroupKFold

from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parent.parent

DATA = REPO / "data" / "cleaned" / "weight_windows_future_targets_all_weeks.csv"

OUT = REPO / "results" / "baseline_ml_results.csv"

OUT.parent.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 69, 123, 456, 789, 1024, 2048, 3333, 5555, 7777,

         9999, 100, 200, 300, 400, 500, 1234, 5678, 8888, 31337]

LOOKBACK = 4

TARGET = "next_20wk_weight_change"

SUPPORT_COLS = ["support_weight_received", "support_diet_received",

                "support_activity_received", "support_tip_received"]

DEMO_COLS = ["bmi_0", "adults"]

VARIANT_TEMPORAL = {

    "supportwc": SUPPORT_COLS + ["weight_change"],

    "wconly": ["weight_change"],

    "swc_only_activity": ["support_activity_received", "weight_change"],

}

def build_arrays_flat(df, temporal_cols, lookback):

    required = ["participant_id", "window_start"] + temporal_cols + DEMO_COLS + [TARGET]

    df_clean = df[required].dropna().sort_values(["participant_id", "window_start"])

    Xs, ys, pids = [], [], []

    for pid, grp in df_clean.groupby("participant_id"):

        grp = grp.sort_values("window_start").reset_index(drop=True)

        if len(grp) < lookback:

            continue

        sv = grp[DEMO_COLS].values[0]

        for i in range(lookback, len(grp) + 1):

            seq = grp[temporal_cols].iloc[i - lookback:i].values.flatten()

            x = np.concatenate([seq, sv])

            Xs.append(x)

            ys.append(grp[TARGET].iloc[i - 1])

            pids.append(pid)

    return np.array(Xs), np.array(ys), np.array(pids)

df = pd.read_csv(DATA)

results = []

for variant_name, temporal_cols in VARIANT_TEMPORAL.items():

    X, y, pids = build_arrays_flat(df, temporal_cols, LOOKBACK)

    print(f"\n=== {variant_name}: X={X.shape}, y={y.shape}, pids={len(set(pids))} ===")

    for seed in SEEDS:

        gkf = GroupKFold(n_splits=5)

        for model_name, build in [

            ("ridge", lambda s: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=s))])),

            ("rf",    lambda s: RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=s)),

            ("mlp",   lambda s: Pipeline([("sc", StandardScaler()),

                                          ("m", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=200,

                                                             random_state=s, early_stopping=True))])),

        ]:

            fold_r2 = []

            preds = np.zeros_like(y, dtype=float)

            for tr, te in gkf.split(X, y, groups=pids):

                m = build(seed)

                m.fit(X[tr], y[tr])

                preds[te] = m.predict(X[te])

                fold_r2.append(r2_score(y[te], preds[te]))

            overall = r2_score(y, preds)

            results.append({

                "model": model_name, "variant": variant_name, "seed": seed,

                "n_samples": len(y), "n_pids": len(set(pids)),

                "fold_r2_mean": np.mean(fold_r2), "fold_r2_std": np.std(fold_r2),

                "overall_r2": overall,

            })

        if seed == SEEDS[0]:

            print(f"  seed={seed}: ridge={results[-3]['overall_r2']:.4f} "

                  f"rf={results[-2]['overall_r2']:.4f} mlp={results[-1]['overall_r2']:.4f}")

pd.DataFrame(results).to_csv(OUT, index=False)

print(f"\nWrote {len(results)} rows → {OUT}")

df_r = pd.DataFrame(results)

print("\n=== Summary by (model, variant) ===")

print(df_r.groupby(["model", "variant"])["overall_r2"].agg(["mean", "std", "count"]).round(4))
