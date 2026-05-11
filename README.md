# mLIFE Peer Support LSTM

Code and analysis artifacts for the manuscript *Does Peer Support Drive Weight Loss? A Deep Learning Investigation of the mLIFE Trial* (Lee, Delgado-Diaz, Turner-McGrievy, Valafar).

The paper applies a recurrent neural network (LSTM) to a weekly panel from the mLIFE randomized behavioral weight-loss trial and uses seed-paired feature ablation to identify which of four peer-support modalities (activity, diet, weight, read-tip thumbs ups) predicts weight change at 20 weeks ahead.

## Repository layout

```
.
├── code/
│   ├── build_panel.py                  rebuild the rolling-window panel from cleaned mLIFE data
│   ├── train_lstm.py                   LSTM training driver (deterministic CUDA)
│   ├── train_baselines.py              Ridge / Random Forest / MLP baselines (Table 3)
│   ├── compute_paired_deltas.py        paired-Δ R² statistics reproducing Tables 1a, 1b, 2, 3
│   ├── make_calibration_figure.py      Supplementary Figure S2 generator
│   ├── make_ablation_jobs.py           job CSV generator for the per-type ablation sweep
│   └── make_sensitivity_jobs.py        job CSV generator for the activity sensitivity grid
├── results/
│   ├── ablation_results.csv            per-type ablation sweep results (LSTM, all variants at reference cell + sensitivity cells)
│   ├── sensitivity_results.csv         activity sensitivity grid results (LSTM, swc_only_activity across off-reference cells)
│   ├── reference_baselines.csv         reference-cell baseline runs (LSTM wconly, supportwc, supportonly)
│   ├── baseline_ml_results.csv         Ridge / RF / MLP results (Table 3)
│   ├── figure_s2_calibration.png       Supplementary Figure S2
│   └── supplementary_table_s1.csv      Supplementary Table S1 (included vs excluded participant characteristics)
└── data/                               data not included (see data/README.md)
```

## Reproducing the paper

The pipeline assumes the cleaned mLIFE panel data is available locally under `data/cleaned/`. Because the mLIFE participant data is protected health information, the cleaned and raw data are not included in this repository. See `data/README.md`.

Given access to the cleaned data:

1. **Rebuild the rolling-window panel** for each (window, stride) combination used in the paper:
   `python code/build_panel.py --window-weeks 6 --stride-weeks 1 --output data/cleaned/sweeps/sweep_window6_stride1.csv`

2. **Generate job CSVs** for the deterministic sweeps:
   `python code/make_ablation_jobs.py`
   `python code/make_sensitivity_jobs.py`

3. **Run the LSTM sweeps** under deterministic CUDA execution:
   `python code/train_lstm.py --jobs code/overnight_g0.csv --output results/ablation_results_gpu0.csv`
   Splitting jobs across two GPUs is supported via `CUDA_VISIBLE_DEVICES`.

4. **Run the baseline models** for Table 3:
   `python code/train_baselines.py`

5. **Render the calibration figure** for Supplementary Figure S2:
   `python code/make_calibration_figure.py`

6. **Verify the published statistics** from the included sweep result CSVs:
   `python code/compute_paired_deltas.py`

`compute_paired_deltas.py` reproduces every value in Tables 1a, 1b, 2, and 3 of the paper directly from the included CSVs and requires no access to the underlying mLIFE data.

## Computing environment

LSTM training uses deterministic CUDA execution (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`) to ensure bit-exact reproducibility across paired runs. The sweeps in the paper were run on a workstation with two NVIDIA GeForce RTX 3090 Ti GPUs running Ubuntu 22.04 with PyTorch 2.1.0 / CUDA 12.1 / cuDNN 8.9.

## License

MIT. See `LICENSE`.
