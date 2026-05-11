
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse

import time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from torch.utils.data import DataLoader, TensorDataset

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True, warn_only=True)

parser = argparse.ArgumentParser()

parser.add_argument("--jobs", required=True)

parser.add_argument("--output", required=True)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[device] {device}", flush=True)

EPOCHS, PATIENCE, BATCH_SIZE = 200, 30, 32

N_FOLDS = 5

SUPPORT_COLS = ["support_weight_received", "support_diet_received",

                "support_activity_received", "support_tip_received"]

WEIGHT_COLS = ["weight_change", "end_weight"]

DEMO_COLS = ["bmi_0", "adults"]

class LSTMNet(nn.Module):

    def __init__(self, temporal_dim, static_dim, hidden_dim, num_layers, dropout):

        super().__init__()

        self.lstm = nn.LSTM(input_size=temporal_dim, hidden_size=hidden_dim,

                            num_layers=num_layers, batch_first=True,

                            dropout=dropout if num_layers > 1 else 0)

        self.fc1 = nn.Linear(hidden_dim + static_dim, 32)

        self.fc2 = nn.Linear(32, 16)

        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, x_static):

        out, _ = self.lstm(x_seq)

        out = out[:, -1, :]

        out = torch.cat([out, x_static], dim=1)

        out = self.relu(self.fc1(out))

        out = self.dropout(out)

        out = self.relu(self.fc2(out))

        out = self.dropout(out)

        return self.fc3(out)

def build_arrays(df, target_col, lookback, variant):

    SUPPORT_BY_TYPE = {

        "weight":   "support_weight_received",

        "diet":     "support_diet_received",

        "activity": "support_activity_received",

        "tip":      "support_tip_received",

    }

    if variant in ("full", "fullhighreg"):

        temporal_cols = SUPPORT_COLS + WEIGHT_COLS

    elif variant in ("no_support", "nosupporthighreg"):

        temporal_cols = WEIGHT_COLS

    elif variant == "supportonly":

        temporal_cols = SUPPORT_COLS

    elif variant == "supportwc":

        temporal_cols = SUPPORT_COLS + ["weight_change"]

    elif variant == "wconly":

        temporal_cols = ["weight_change"]

    elif variant.startswith("swc_no_"):

        t = variant[len("swc_no_"):]

        if t not in SUPPORT_BY_TYPE:

            raise ValueError(f"unknown support type {t}")

        keep = [c for k, c in SUPPORT_BY_TYPE.items() if k != t]

        temporal_cols = keep + ["weight_change"]

    elif variant.startswith("swc_only_"):

        t = variant[len("swc_only_"):]

        if t not in SUPPORT_BY_TYPE:

            raise ValueError(f"unknown support type {t}")

        temporal_cols = [SUPPORT_BY_TYPE[t]] + ["weight_change"]

    elif variant.startswith("swc_combo_"):

        types = variant[len("swc_combo_"):].split("_")

        for t in types:

            if t not in SUPPORT_BY_TYPE:

                raise ValueError(f"unknown support type {t}")

        temporal_cols = [SUPPORT_BY_TYPE[t] for t in types] + ["weight_change"]

    else:

        raise ValueError(variant)

    required = ["participant_id", "window_start"] + temporal_cols + DEMO_COLS + [target_col]

    df_clean = df[required].dropna().sort_values(["participant_id", "window_start"])

    seqs, statics, targets, pids = [], [], [], []

    for pid, grp in df_clean.groupby("participant_id"):

        grp = grp.sort_values("window_start")

        if len(grp) < lookback:

            continue

        tv = grp[temporal_cols].values

        sv = grp[DEMO_COLS].values[0]

        yv = grp[target_col].values

        for i in range(lookback - 1, len(grp)):

            seqs.append(tv[i - lookback + 1 : i + 1])

            statics.append(sv)

            targets.append(yv[i])

            pids.append(pid)

    return (np.array(seqs, dtype=np.float32),

            np.array(statics, dtype=np.float32),

            np.array(targets, dtype=np.float32),

            np.array(pids))

def run_cv(X_seq, X_static, y, pids, seed, hidden_dim, num_layers, dropout, lr, weight_decay):

    np.random.seed(seed); torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    gkf = GroupKFold(n_splits=N_FOLDS)

    all_preds = np.full(len(y), np.nan)

    fold_r2s = []

    for fold, (tr, te) in enumerate(gkf.split(X_seq, y, groups=pids)):

        X_seq_tr, X_seq_te = X_seq[tr], X_seq[te]

        X_st_tr,  X_st_te  = X_static[tr], X_static[te]

        y_tr,     y_te     = y[tr], y[te]

        ssc = StandardScaler()

        n_tr, T, F = X_seq_tr.shape

        X_seq_tr_s = ssc.fit_transform(X_seq_tr.reshape(-1, F)).reshape(n_tr, T, F)

        n_te = X_seq_te.shape[0]

        X_seq_te_s = ssc.transform(X_seq_te.reshape(-1, F)).reshape(n_te, T, F)

        stsc = StandardScaler()

        X_st_tr_s = stsc.fit_transform(X_st_tr)

        X_st_te_s = stsc.transform(X_st_te)

        gen = torch.Generator(); gen.manual_seed(seed + fold)

        train_ds = TensorDataset(torch.FloatTensor(X_seq_tr_s), torch.FloatTensor(X_st_tr_s), torch.FloatTensor(y_tr))

        test_ds  = TensorDataset(torch.FloatTensor(X_seq_te_s), torch.FloatTensor(X_st_te_s), torch.FloatTensor(y_te))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        model = LSTMNet(F, X_static.shape[1], hidden_dim, num_layers, dropout).to(device)

        crit = nn.MSELoss()

        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

        best_loss, pcount, best_state = float("inf"), 0, None

        for ep in range(EPOCHS):

            model.train()

            for xb, xs, yb in train_loader:

                xb, xs, yb = xb.to(device), xs.to(device), yb.to(device)

                opt.zero_grad()

                loss = crit(model(xb, xs).squeeze(), yb)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                opt.step()

            model.eval()

            with torch.no_grad():

                tloss = sum(crit(model(xb.to(device), xs.to(device)).squeeze(), yb.to(device)).item() * len(yb)

                            for xb, xs, yb in test_loader) / len(test_ds)

            sch.step(tloss)

            if tloss < best_loss:

                best_loss, pcount = tloss, 0

                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            else:

                pcount += 1

            if pcount >= PATIENCE:

                break

        model.load_state_dict(best_state); model.eval()

        with torch.no_grad():

            preds = np.concatenate([model(xb.to(device), xs.to(device)).squeeze().cpu().numpy()

                                    for xb, xs, _ in test_loader])

        all_preds[te] = preds

        fold_r2s.append(r2_score(y_te, preds))

    valid = ~np.isnan(all_preds)

    return r2_score(y[valid], all_preds[valid]), fold_r2s

jobs = pd.read_csv(args.jobs)

print(f"[jobs] loaded {len(jobs)} from {args.jobs}", flush=True)

done_ids = set()

if Path(args.output).exists():

    done_ids = set(pd.read_csv(args.output)["job_id"].tolist())

    print(f"[resume] {len(done_ids)} done", flush=True)

pending = jobs[~jobs["job_id"].isin(done_ids)]

print(f"[run] {len(pending)} pending", flush=True)

if not Path(args.output).exists():

    pd.DataFrame(columns=["job_id", "dataset_path", "target_col", "variant", "seed", "lookback",

                          "hidden_dim", "num_layers", "dropout", "lr", "weight_decay",

                          "n_samples", "n_pids", "overall_r2", "fold_r2_mean", "fold_r2_std", "elapsed_sec"]).to_csv(args.output, index=False)

ds_cache = {}

def load(p):

    if p not in ds_cache:

        ds_cache[p] = pd.read_csv(p)

    return ds_cache[p]

t_total = time.time()

for _, row in pending.iterrows():

    t0 = time.time()

    df = load(row["dataset_path"])

    X_seq, X_st, y, pids = build_arrays(df, row["target_col"], int(row["lookback"]), row["variant"])

    if len(y) == 0:

        jid = row["job_id"]; print(f"[skip] {jid} no samples", flush=True); continue

    r2, fr2s = run_cv(X_seq, X_st, y, pids, int(row["seed"]),

                      int(row["hidden_dim"]), int(row["num_layers"]), float(row["dropout"]),

                      float(row["lr"]), float(row["weight_decay"]))

    elapsed = time.time() - t0

    out = {**row.to_dict(),

           "n_samples": len(y), "n_pids": len(np.unique(pids)),

           "overall_r2": r2, "fold_r2_mean": float(np.mean(fr2s)), "fold_r2_std": float(np.std(fr2s)),

           "elapsed_sec": elapsed}

    pd.DataFrame([out]).to_csv(args.output, mode="a", header=False, index=False)

    jid = row["job_id"]

    print(f"[done] {jid:60s} R²={r2:+.4f}  N={len(y)}  pids={len(np.unique(pids))}  t={elapsed:.1f}s  total={time.time()-t_total:.0f}s", flush=True)

print("[finished]", flush=True)
