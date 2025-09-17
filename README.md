This repository provides a local-executable Python script for sensor-space SPN analyses, cluster-based permutation tests (MNE), mixed-effects models (statsmodels), and a binomial GLM linking SPN (in noise) to speech-in-noise accuracy.

## Repository layout

```
spn-eeg-analysis/
├── analyze_spn.py          # Main analysis script (CLI)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT
├── .gitignore
├── data/                   # Put your CSVs here (not tracked by git)
└── outputs/                # Results are written here
```

## Data inputs

- `spn_electrode_long.csv` (required)

  Required columns:
  - `Participant` (string/ID)
  - `Sex` (F/M)
  - `Condition` (labels will be normalized to `in noise` / `in silence`)
  - `Electrode` (channel name, e.g., Fz, P7, T7)
  - `SPN_uV` (float)

- `electrode_metadata.csv` (optional)

  Columns (at minimum):
  - `Electrode`
  - `ROI` (Frontal/Central/Parietal/Occipital/Temporal/Other)
  - `Hemisphere` (Left/Right/Midline)

  If omitted, the script creates minimal metadata (ROI/Hemisphere) from electrode names.

- `speech_accuracy.csv` (optional; required only for the GLM)

  Required columns:
  - `Participant`
  - `SNR_dB` (integer; e.g., -15, -12, ...)
  - `n_correct`
  - `n_trials`

## Installation

```bash
# (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Python 3.10 is recommended.

## Usage

Place your CSV files under `./data/` or pass absolute file paths.

### Minimal (LMM + cluster test)

```bash
python analyze_spn.py --data_dir ./data --spn spn_electrode_long.csv --out_dir ./outputs
```

### With electrode metadata

```bash
python analyze_spn.py --data_dir ./data --spn spn_electrode_long.csv --meta electrode_metadata.csv --out_dir ./outputs
```

### With behavioral GLM

```bash
python analyze_spn.py \
  --data_dir ./data \
  --spn spn_electrode_long.csv \
  --meta electrode_metadata.csv \
  --beh speech_accuracy.csv \
  --out_dir ./outputs \
  --n_perm 5000 --seed 42
```

- If `--meta` is not given (or not found), minimal metadata will be inferred.
- If `--beh` is not given, the GLM section is skipped.
- Figures are saved to `./outputs/figtables` as both `.tif` and `.png`. Tables (CSVs) are saved under `./outputs/`.

## Analysis overview

1. **Preprocessing**
   - Normalize `Condition` to `in noise`/`in silence`.
   - Effect coding: `in silence` = +0.5, `in noise` = −0.5.
   - Keep participants observed in both conditions.
   - Merge ROI/Hemisphere metadata (infer if absent).

2. **LMM: ROI × Hemisphere (main analysis)**
   - Model: `SPN_uV ~ Condition_c * Hem_c + C(ROI) + Sex_c`, with subject-level random effects (random slope for Condition when possible; fallback to random intercept).
   - Saves fixed-effect table: `Table_LMM_ROIxHem_fixed_effects.csv`.

3. **Per-electrode LMM (FDR for Condition effect)**
   - Saves `Table_perElectrode_LMM.csv` (includes uncorrected p and FDR-adjusted p).

4. **Topomap and cluster-based permutation test (MNE)**
   - Compute Noise−Silence differences per subject/channel.
   - Impute missing cells by channel mean (fallback to global mean).
   - Two-tailed 1-sample cluster test with `n_permutations` and `seed`.
   - Save cluster membership tables and a masked topomap figure.

5. **LMM on representative cluster mean (Condition × Sex)**
   - Saves fixed-effect table and violin plot.

6. **Sensitivity analyses**
   - LOEO (drop-one-electrode) and participant bootstrap (B=500).
   - Save CSVs and simple figures.

7. **GLM (optional)**
   - Binomial logit with cluster-robust SE by participant:
     ```
     prop ~ C(SNR_dB) + SPN_c + C(SNR_dB):SPN_c + Sex01
     ```
   - Save model summary, odds ratios, and SNR-wise simple effects (with FDR).

## Expected outputs

- `FillSummary_perElectrode.csv`
- `Table_LMM_ROIxHem_fixed_effects.csv`
- `Table_perElectrode_LMM.csv`
- `Supp_Table_cluster_electrodes_*.csv` (per cluster) and `_ALL.csv`
- Figures under `outputs/figtables/`:
  - `Fig1_topomap_mask.*`
  - `Fig2_cluster_violin_cond.*`
  - `Fig3a_*`, `Fig3b_*`, `Fig3c_*`, `Fig3d_*` (if sensitivity analyses run)
- GLM (if run):
  - `GLM_logit_summary.txt`
  - `GLM_odds_ratios.csv`
  - `Table_simple_OR_by_SNR.csv`

## Notes and troubleshooting

- **Headless environments**: The script forces the `Agg` backend for matplotlib, so plots save without a GUI.
- **MNE montage**: Uses the standard `10-20` montage (`standard_1020`), with a visualization-only mapping `{T3→T7, T4→T8, T5→P7, T6→P8}`.
- **Condition labels**: If you use `noise`/`silence`, they are normalized to `in noise`/`in silence`.
- **Errors**:
  - *Missing columns*: The script validates required columns and stops with a clear message.
  - *Multiple filename matches*: Provide the exact filename or a full path to disambiguate.
  - *No significant cluster*: Cluster-mean analyses and GLM are skipped.

## Reproducibility

Set `--seed` to control random components (cluster test permutations and bootstrap sampling). Permutation tests will still have Monte Carlo variability.

## License

MIT (see `LICENSE`).
