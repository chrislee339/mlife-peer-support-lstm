from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

REF_W, REF_S, REF_H, REF_LKB = 6, 1, 20, 4
SUPPORT_TYPES = ("activity", "diet", "tip", "weight")


def annotate(df):
    df = df.copy()
    df["arch"] = df["job_id"].str.extract(r"arch(\w+?)_")
    df["w"] = df["job_id"].str.extract(r"_w(\d+)_").astype(int)
    df["s"] = df["job_id"].str.extract(r"_s(\d+)_").astype(int)
    df["h"] = df["job_id"].str.extract(r"_h(\d+)_").astype(int)
    return df


def paired_delta(a_series, b_series):
    common = a_series.index.intersection(b_series.index)
    a = a_series.loc[common]
    b = b_series.loc[common]
    delta = a - b
    t, p = stats.ttest_rel(a, b) if len(common) >= 2 else (np.nan, np.nan)
    try:
        _, pw = stats.wilcoxon(a, b, alternative="two-sided") if len(common) >= 2 else (np.nan, np.nan)
    except ValueError:
        pw = np.nan
    return {
        "n": len(common),
        "mean_a": a.mean() if len(a) else np.nan,
        "mean_b": b.mean() if len(b) else np.nan,
        "mean_delta": delta.mean() if len(delta) else np.nan,
        "sd_delta": delta.std() if len(delta) else np.nan,
        "n_pos": int((delta > 0).sum()) if len(delta) else 0,
        "t": t,
        "p_ttest": p,
        "p_wilcoxon": pw,
    }


def at_ref(df):
    return df[(df["w"] == REF_W) & (df["s"] == REF_S)
              & (df["h"] == REF_H) & (df["lookback"] == REF_LKB)]


def table_1a(ablation, reference_baselines):
    wc = at_ref(reference_baselines)
    wc = wc[wc["variant"] == "wconly"].set_index(["seed", "arch"])["overall_r2"]
    rows = []
    for t in SUPPORT_TYPES:
        a = at_ref(ablation)
        a = a[a["variant"] == f"swc_only_{t}"].set_index(["seed", "arch"])["overall_r2"]
        rows.append({"support_type": t, **paired_delta(a, wc)})
    return pd.DataFrame(rows)


def table_1b(ablation, reference_baselines):
    sup = at_ref(reference_baselines)
    sup = sup[sup["variant"] == "supportwc"].set_index(["seed", "arch"])["overall_r2"]
    rows = []
    for t in SUPPORT_TYPES:
        a = at_ref(ablation)
        a = a[a["variant"] == f"swc_no_{t}"].set_index(["seed", "arch"])["overall_r2"]
        rows.append({"removed_type": t, **paired_delta(a, sup)})
    return pd.DataFrame(rows)


def table_2(ablation, sensitivity, reference_baselines):
    activity_alone = pd.concat([
        sensitivity[sensitivity["variant"] == "swc_only_activity"],
        ablation[ablation["variant"] == "swc_only_activity"],
    ], ignore_index=True)
    wconly = pd.concat([
        ablation[ablation["variant"] == "wconly"],
        reference_baselines[reference_baselines["variant"] == "wconly"],
    ], ignore_index=True)
    perturbations = [
        ("window", [2, 3, 5, 6, 8, 12], lambda v: (v, 1, 20, 4)),
        ("stride", [2, 3, 5], lambda v: (6, v, 20, 4)),
        ("lookback", [2, 6, 8, 12], lambda v: (6, 1, 20, v)),
        ("horizon", [4, 8, 12, 26], lambda v: (6, 1, v, 4)),
    ]
    rows = []
    for pname, values, key in perturbations:
        for v in values:
            w, s, h, lkb = key(v)
            a = activity_alone[
                (activity_alone["w"] == w) & (activity_alone["s"] == s)
                & (activity_alone["h"] == h) & (activity_alone["lookback"] == lkb)
            ].set_index(["seed", "arch"])["overall_r2"]
            b = wconly[
                (wconly["w"] == w) & (wconly["s"] == s)
                & (wconly["h"] == h) & (wconly["lookback"] == lkb)
            ].set_index(["seed", "arch"])["overall_r2"]
            if len(a) == 0 or len(b) == 0:
                continue
            d = paired_delta(a, b)
            rows.append({"perturbation": pname, "value": v,
                         "r2_activity": d["mean_a"], "r2_wconly": d["mean_b"],
                         "delta": d["mean_delta"], "sd": d["sd_delta"],
                         "n_pos": d["n_pos"], "t": d["t"], "p": d["p_ttest"]})
    return pd.DataFrame(rows)


def table_3(baseline, ablation, reference_baselines):
    rows = []
    for model in sorted(baseline["model"].unique()):
        sub = baseline[baseline["model"] == model]
        row = {"model": model}
        for variant in ["wconly", "supportwc", "swc_only_activity"]:
            row[variant] = sub[sub["variant"] == variant]["overall_r2"].mean()
        rows.append(row)
    lstm_ref = pd.concat([at_ref(ablation), at_ref(reference_baselines)], ignore_index=True)
    row = {"model": "lstm"}
    for variant in ["wconly", "supportwc", "swc_only_activity"]:
        row[variant] = lstm_ref[lstm_ref["variant"] == variant]["overall_r2"].mean()
    rows.append(row)
    return pd.DataFrame(rows)


def main():
    ablation = annotate(pd.read_csv(RESULTS / "ablation_results.csv"))
    sensitivity = annotate(pd.read_csv(RESULTS / "sensitivity_results.csv"))
    reference_baselines = annotate(pd.read_csv(RESULTS / "reference_baselines.csv"))
    baseline = pd.read_csv(RESULTS / "baseline_ml_results.csv")

    pd.set_option("display.float_format", "{:+.4f}".format)

    print("\n=== Table 1a: per-type alone vs wconly (reference cell) ===")
    print(table_1a(ablation, reference_baselines).to_string(index=False))

    print("\n=== Table 1b: drop-one vs supportwc (reference cell) ===")
    print(table_1b(ablation, reference_baselines).to_string(index=False))

    print("\n=== Table 2: activity sensitivity grid ===")
    print(table_2(ablation, sensitivity, reference_baselines).to_string(index=False))

    print("\n=== Table 3: model family comparison (mean R² at reference cell) ===")
    print(table_3(baseline, ablation, reference_baselines).to_string(index=False))


if __name__ == "__main__":
    main()
