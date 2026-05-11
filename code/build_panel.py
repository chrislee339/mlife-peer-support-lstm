import argparse

import sys

from pathlib import Path

import numpy as np

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--window-weeks", type=int, required=True, help="Number of weekly observations per window")

parser.add_argument("--stride-weeks", type=int, default=1)

parser.add_argument("--output", type=str, required=True)

parser.add_argument("--max-consec-miss-to-interp", type=int, default=3)

parser.add_argument("--drop-gap-weeks", type=int, default=8)

args = parser.parse_args()

WINDOW_WEEKS = args.window_weeks

STRIDE_WEEKS = args.stride_weeks

MAX_CONSEC_MISS_TO_INTERP = args.max_consec_miss_to_interp

DROP_GAP_WEEKS = args.drop_gap_weeks

DATA_CLEANED_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"

df_weights = pd.read_csv(DATA_CLEANED_DIR / "weights_cleaned_3.csv")

df_support = pd.read_csv(DATA_CLEANED_DIR / "social_support_log_with_affected_user.csv")

df_demo = pd.read_csv(DATA_CLEANED_DIR / "demographics.csv")

def normalize_id(series):

    s = series.copy()

    try:

        s_num = pd.to_numeric(s, errors="coerce")

        if s_num.notna().any():

            s = s_num.astype("Int64").astype("string")

        else:

            s = s.astype("string")

    except Exception:

        s = s.astype("string")

    return s.str.strip()

def pick_col(df, candidates, required=True, label="column"):

    for c in candidates:

        if c in df.columns:

            return c

    if required:

        raise ValueError(f"Could not find {label}. Tried: {candidates}")

    return None

def parse_datetime(df, date_candidates, time_candidates=None):

    dt_col = pick_col(df, date_candidates, required=False, label="date")

    if dt_col is None:

        return None, df.copy()

    dfx = df.copy()

    t_col = pick_col(df, time_candidates or [], required=False, label="time")

    if t_col:

        dfx["_dt"] = pd.to_datetime(dfx[dt_col].astype(str) + " " + dfx[t_col].astype(str), errors="coerce")

    else:

        dfx["_dt"] = pd.to_datetime(dfx[dt_col], errors="coerce")

    return "_dt", dfx

w_pid_col = pick_col(df_weights, ["Participant ID", "participant_id", "pid", "mlife_id"])

w_wt_col  = pick_col(df_weights, ["Weight", "weight", "weight_kg", "Weight (kg)", "Weight_lbs_num"])

w_dt_col, dfx = parse_datetime(df_weights, date_candidates=["_parsed_dt", "Date", "date"], time_candidates=["Time", "time"])

if w_dt_col is None:

    raise ValueError("No datetime in weights")

base = dfx[[w_pid_col, w_wt_col, w_dt_col]].copy()

base[w_pid_col] = normalize_id(base[w_pid_col])

base[w_wt_col] = pd.to_numeric(base[w_wt_col], errors="coerce")

base = base.dropna(subset=[w_pid_col, w_dt_col]).sort_values([w_pid_col, w_dt_col])

weekly_records = []

window_records = []

for pid, g in base.groupby(w_pid_col):

    if g.empty:

        continue

    s = g.set_index(w_dt_col)[w_wt_col].sort_index()

    s_week = s.resample("7D").mean()

    isna = s_week.isna()

    grp = (isna != isna.shift()).cumsum()

    run_len = isna.groupby(grp).transform("size")

    if (isna & (run_len >= DROP_GAP_WEEKS)).any():

        continue

    s_all_interp = s_week.interpolate(method="time", limit_direction="both")

    fill_mask = isna & (run_len <= MAX_CONSEC_MISS_TO_INTERP)

    s_filled = s_week.copy()

    s_filled[fill_mask] = s_all_interp[fill_mask]

    for dt, val in s_filled.items():

        if not np.isnan(val):

            weekly_records.append({"participant_id": pid, "week_dt": dt, "weekly_weight": float(val)})

    vals = s_filled.values

    idx = s_filled.index

    n = len(s_filled)

    win = WINDOW_WEEKS

    stride = STRIDE_WEEKS

    i = 0

    while i + win <= n:

        window_vals = vals[i:i+win]

        if not np.isnan(window_vals).any():

            start_wt = float(window_vals[0])

            end_wt = float(window_vals[-1])

            delta = end_wt - start_wt

            pct = (delta / start_wt) if start_wt != 0 else np.nan

            window_records.append({

                "participant_id": pid,

                "window_start": idx[i],

                "window_end": idx[i+win-1],

                "start_weight": start_wt,

                "end_weight": end_wt,

                "weight_change": delta,

                "pct_weight_change": pct,

            })

        i += stride

df_weight_windows = pd.DataFrame.from_records(window_records)

df_weekly = pd.DataFrame.from_records(weekly_records)

df_weekly["participant_id"] = df_weekly["participant_id"].astype(str)

print(f"weight windows: {df_weight_windows.shape}, weekly series: {df_weekly.shape}", file=sys.stderr)

s_pid_col = pick_col(df_support, ["AffectedUser", "Participant ID", "participant_id", "mlife_id"])

s_action_col = pick_col(df_support, ["Action", "action", "action_id"])

s_group_col = "Group" if "Group" in df_support.columns else None

s_dt_col, dfs = parse_datetime(df_support, date_candidates=["Date", "date"], time_candidates=None)

if s_dt_col is None:

    raise ValueError("No datetime in support")

dfs = dfs[[s_pid_col, s_action_col, s_dt_col] + ([s_group_col] if s_group_col else [])].dropna(subset=[s_pid_col, s_action_col, s_dt_col]).copy()

dfs[s_pid_col] = normalize_id(dfs[s_pid_col])

dfs[s_action_col] = pd.to_numeric(dfs[s_action_col], errors="coerce")

dfs = dfs.dropna(subset=[s_action_col])

actions_of_interest = [7, 8, 9, 10]

dfs = dfs[dfs[s_action_col].isin(actions_of_interest)].copy()

dfs = dfs.sort_values([s_pid_col, s_dt_col])

grp_modal_overall = dfs.groupby(s_pid_col)[s_group_col].agg(

    lambda x: x.dropna().mode().iloc[0] if s_group_col and not x.dropna().empty else pd.NA

) if s_group_col else None

support_records = []

for pid, g in dfs.groupby(s_pid_col):

    win = df_weight_windows[df_weight_windows["participant_id"] == pid]

    if win.empty:

        continue

    for _, wrow in win.iterrows():

        ws = wrow["window_start"]

        we = wrow["window_end"]

        sub = g[(g[s_dt_col] >= ws) & (g[s_dt_col] <= we)]

        if sub.empty:

            counts = {7:0, 8:0, 9:0, 10:0}

            modal_group = grp_modal_overall.get(pid) if s_group_col else pd.NA

        else:

            ct = sub[s_action_col].value_counts().to_dict()

            counts = {a: int(ct.get(a, 0)) for a in actions_of_interest}

            if s_group_col:

                ss = sub[s_group_col].dropna()

                modal_group = ss.mode().iloc[0] if len(ss.mode()) > 0 else grp_modal_overall.get(pid)

            else:

                modal_group = pd.NA

        support_records.append({

            "participant_id": pid,

            "window_start": ws,

            "window_end": we,

            "support_weight_received": counts[7],

            "support_diet_received": counts[8],

            "support_activity_received": counts[9],

            "support_tip_received": counts[10],

            "total_support_received": sum(counts.values()),

            "Group": modal_group,

        })

df_support_windows = pd.DataFrame.from_records(support_records)

print(f"support windows: {df_support_windows.shape}", file=sys.stderr)

demo_id_col = pick_col(df_demo, ["mlife_id", "participant_id", "Participant ID"])

df_demo2 = df_demo.rename(columns={demo_id_col: "participant_id"}).copy()

df_demo2["participant_id"] = normalize_id(df_demo2["participant_id"])

df_w = df_weight_windows.copy()

df_s = df_support_windows.copy()

df_w["participant_id"] = normalize_id(df_w["participant_id"])

df_s["participant_id"] = normalize_id(df_s["participant_id"])

m = df_w.merge(df_s, on=["participant_id", "window_start", "window_end"], how="left")

df_windows = m.merge(df_demo2, on="participant_id", how="left")

df_windows["participant_id"] = df_windows["participant_id"].astype(str)

max_weeks_per_pid = df_weekly.groupby("participant_id").size()

max_horizon = int(max_weeks_per_pid.quantile(0.5))

print(f"max horizon: {max_horizon}", file=sys.stderr)

wks = df_weekly.sort_values(["participant_id", "week_dt"]).copy()

for h in range(1, max_horizon + 1):

    wks[f"weight_plus_{h}"] = wks.groupby("participant_id")["weekly_weight"].shift(-h)

merge_base = df_windows[["participant_id", "window_end", "end_weight"]].copy()

merge_base["window_end"] = pd.to_datetime(merge_base["window_end"])

wks["week_dt"] = pd.to_datetime(wks["week_dt"])

extended = merge_base.merge(

    wks[["participant_id", "week_dt"] + [f"weight_plus_{h}" for h in range(1, max_horizon + 1)]],

    left_on=["participant_id", "window_end"],

    right_on=["participant_id", "week_dt"],

    how="left",

).drop(columns=["week_dt"])

for h in range(1, max_horizon + 1):

    extended[f"next_{h}wk_weight"] = extended[f"weight_plus_{h}"]

    extended[f"next_{h}wk_weight_change"] = extended[f"next_{h}wk_weight"] - extended["end_weight"]

    extended[f"next_{h}wk_pct_change"] = extended[f"next_{h}wk_weight_change"] / extended["end_weight"]

    extended = extended.drop(columns=[f"weight_plus_{h}"])

df_windows["window_end"] = pd.to_datetime(df_windows["window_end"])

final = df_windows.merge(

    extended.drop(columns=["end_weight"]),

    on=["participant_id", "window_end"],

    how="left",

)

out_path = Path(args.output)

out_path.parent.mkdir(parents=True, exist_ok=True)

final.to_csv(out_path, index=False)

n_pids = final["participant_id"].nunique(); print(f"saved -> {out_path}  shape={final.shape}  pids={n_pids}")
