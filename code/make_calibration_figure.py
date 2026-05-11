import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from torch.utils.data import DataLoader, TensorDataset

from statsmodels.nonparametric.smoothers_lowess import lowess

import statsmodels.api as sm

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

try:

    torch.use_deterministic_algorithms(True, warn_only=True)

except Exception:

    pass

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")

print(f"[device] {device}", flush=True)

REPO = Path(__file__).resolve().parent.parent

DATA = REPO / "data" / "cleaned" / "weight_windows_future_targets_all_weeks.csv"

OUT_DIR = REPO / "results"

OUT_DIR.mkdir(exist_ok=True)

PRED_CSV = OUT_DIR / "figure_s2_calibration_predictions.csv"

FIG_PNG = OUT_DIR / "figure_s2_calibration.png"

EPOCHS, PATIENCE, BATCH_SIZE = 200, 30, 32

N_FOLDS = 5

LOOKBACK = 4

TARGET = "next_20wk_weight_change"

SEED = 42

HIDDEN_DIM, NUM_LAYERS, DROPOUT = 32, 2, 0.3   

LR, WEIGHT_DECAY = 1e-3, 1e-3

DEMO_COLS = ["bmi_0", "adults"]

SUPPORT_BY_TYPE = {

    "weight":   "support_weight_received",

    "diet":     "support_diet_received",

    "activity": "support_activity_received",

    "tip":      "support_tip_received",

}

VARIANTS = {

    "wconly": ["weight_change"],

    "swc_only_activity": [SUPPORT_BY_TYPE["activity"], "weight_change"],

}

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

def build_arrays(df, temporal_cols, lookback):

    required = ["participant_id", "window_start"] + temporal_cols + DEMO_COLS + [TARGET]

    df_clean = df[required].dropna().sort_values(["participant_id", "window_start"])

    seqs, statics, targets, pids = [], [], [], []

    for pid, grp in df_clean.groupby("participant_id"):

        grp = grp.sort_values("window_start")

        if len(grp) < lookback:

            continue

        tv = grp[temporal_cols].values

        sv = grp[DEMO_COLS].values[0]

        yv = grp[TARGET].values

        for i in range(lookback - 1, len(grp)):

            seqs.append(tv[i - lookback + 1: i + 1])

            statics.append(sv)

            targets.append(yv[i])

            pids.append(pid)

    return (np.array(seqs, dtype=np.float32),

            np.array(statics, dtype=np.float32),

            np.array(targets, dtype=np.float32),

            np.array(pids))

def run_cv_predictions(X_seq, X_static, y, pids, seed):

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

        model = LSTMNet(F, X_static.shape[1], HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)

        crit = nn.MSELoss()

        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

        print(f"  fold {fold + 1}/{N_FOLDS}: R² = {fold_r2s[-1]:+.4f}", flush=True)

    valid = ~np.isnan(all_preds)

    overall = r2_score(y[valid], all_preds[valid])

    return all_preds, overall, fold_r2s

def calibration_stats(y_true, y_pred):

    X = sm.add_constant(y_pred)

    res = sm.OLS(y_true, X).fit()

    alpha, beta = res.params[0], res.params[1]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return dict(alpha=alpha, beta=beta, rmse=rmse,

                ci_alpha=res.conf_int()[0], ci_beta=res.conf_int()[1])

df = pd.read_csv(DATA)

print(f"[data] {DATA.name}: {df.shape}", flush=True)

results = {}

all_pred_rows = []

for vname, temporal_cols in VARIANTS.items():

    print(f"\n=== {vname} ({temporal_cols}) ===", flush=True)

    X_seq, X_st, y, pids = build_arrays(df, temporal_cols, LOOKBACK)

    print(f"  X_seq={X_seq.shape}, y={y.shape}, n_pids={len(np.unique(pids))}", flush=True)

    t0 = time.time()

    preds, overall_r2, fold_r2s = run_cv_predictions(X_seq, X_st, y, pids, SEED)

    elapsed = time.time() - t0

    print(f"  overall R² = {overall_r2:+.4f} (folds: {[f'{f:+.3f}' for f in fold_r2s]})  t={elapsed:.0f}s", flush=True)

    cal = calibration_stats(y, preds)

    print(f"  calibration: alpha={cal['alpha']:+.3f}  beta={cal['beta']:+.3f}  RMSE={cal['rmse']:.3f}", flush=True)

    results[vname] = dict(y=y, preds=preds, r2=overall_r2, fold_r2s=fold_r2s, cal=cal)

    for yt, yp in zip(y, preds):

        all_pred_rows.append(dict(variant=vname, y_true=yt, y_pred=yp))

pd.DataFrame(all_pred_rows).to_csv(PRED_CSV, index=False)

print(f"\n[saved] predictions → {PRED_CSV}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

labels = {"wconly": "Weight trajectory only (baseline)",

          "swc_only_activity": "Activity support + weight (headline)"}

for ax, (vname, r) in zip(axes, results.items()):

    y_true, y_pred = r["y"], r["preds"]

    cal = r["cal"]

    ax.scatter(y_pred, y_true, s=8, alpha=0.25, color="#3F76C1", edgecolor="none")

    lo = min(y_true.min(), y_pred.min())

    hi = max(y_true.max(), y_pred.max())

    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="y = x (perfect)")

    sm_xy = lowess(y_true, y_pred, frac=0.4, return_sorted=True)

    ax.plot(sm_xy[:, 0], sm_xy[:, 1], color="#D14545", linewidth=2.5, label="LOWESS smoother")

    ax.set_xlabel("Predicted percent weight change at 20 weeks")

    ax.set_ylabel("Observed percent weight change at 20 weeks")

    ax.set_title(f"{labels[vname]}\nR² = {r['r2']:.3f}, calibration α = {cal['alpha']:+.3f}, β = {cal['beta']:+.3f}")

    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)

    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)

    ax.legend(loc="upper left", fontsize=9)

    ax.grid(alpha=0.2)

plt.suptitle("Supplementary Figure S2. Calibration of canonical LSTM (single seed = 42, 5 fold GroupKFold).",

             y=1.02, fontsize=11)

plt.tight_layout()

plt.savefig(FIG_PNG, dpi=200, bbox_inches="tight")

print(f"[saved] figure → {FIG_PNG}", flush=True)

print("\n=== Summary ===")

for vname, r in results.items():

    print(f"  {vname:25s}: R²={r['r2']:+.4f}  α={r['cal']['alpha']:+.3f}  β={r['cal']['beta']:+.3f}  RMSE={r['cal']['rmse']:.3f}")
