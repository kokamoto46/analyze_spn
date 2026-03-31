# SPN/CNV EEG preprocessing and analysis pipeline
This repository contains the analysis code only.  
The dataset is available on Zenodo: [https://doi.org/10.5281/zenodo.19344047](https://doi.org/10.5281/zenodo.19344047)

This repository contains a MATLAB/EEGLAB preprocessing pipeline and a Python-based analysis workflow for SPN and CNV EEG data.

The repository is organized around two parts:

- `matlab/`: preprocessing, epoch cleaning, quality control, and long-format export
- `python/`: statistical analysis, descriptive summaries, and figure generation

## Overview

The preprocessing pipeline is divided into three stages.

1. **Stage 1** prepares datasets for manual ICA review.
   - Import raw EDF files
   - Assign channel locations
   - Import event CSV files
   - Apply a 30 Hz low-pass filter
   - Detect bad channels with PREP
   - Run ICA on good channels
   - Save datasets for manual IC rejection

2. **Stage 2A** creates marked epoched datasets after manual ICA cleaning.
   - Load manually ICA-cleaned datasets
   - Epoch SPN and CNV events
   - Apply automatic threshold-based marking
   - Save epoched datasets for manual epoch review

3. **Stage 2B** produces the final cleaned datasets.
   - Load reviewed epoched datasets
   - Reject marked epochs
   - Apply baseline correction
   - Save non-interpolated primary-analysis datasets
   - Optionally save interpolated supplementary datasets
   - Export run-level trial logs

Additional MATLAB scripts generate QC tables and export long-format CSV files. The Python script performs mixed-effects modeling, electrode-wise supplementary analyses, and figure generation.

## Repository structure

```text
spn-cnv-pipeline/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ matlab/
│  ├─ stage1_prepare_ica.m
│  ├─ stage2a_epoch_and_mark.m
│  ├─ stage2b_finalize_epochs.m
│  ├─ make_run_qc_tables.m
│  ├─ export_anticipatory_long_noninterp.m
│  └─ export_anticipatory_long_interp.m
├─ python/
│  └─ run_spn_cnv_analysis.py
├─ docs/
│  ├─ pipeline_overview.md
│  ├─ input_file_specifications.md
│  └─ output_file_specifications.md
├─ examples/
│  ├─ example_commands.md
│  └─ example_directory_layout.md
```

## Dependencies

### MATLAB

The MATLAB workflow requires:

- MATLAB
- [EEGLAB](https://sccn.ucsd.edu/eeglab/)
- PREP pipeline
- BIOSIG support through EEGLAB

### Python

The Python workflow requires Python 3.10 or later and the packages listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

## Expected data layout

The MATLAB scripts assume a participant-wise directory layout under `rootDir`.

```text
rootDir/
├─ 01/
│  ├─ sub01s1_raw.edf
│  ├─ sub01s2_raw.edf
│  ├─ sub01n1_raw.edf
│  ├─ sub01n2_raw.edf
│  ├─ spn_silence1event.csv
│  ├─ spn_silence2event.csv
│  ├─ spn_noise1event.csv
│  ├─ spn_noise2event.csv
│  ├─ cnv_silence1event.csv
│  ├─ cnv_silence2event.csv
│  ├─ cnv_noise1event.csv
│  ├─ cnv_noise2event.csv
│  └─ derivatives/
├─ 02/
│  └─ ...
├─ participant.tsv
├─ electrode_metadata.csv
├─ run_qc.csv
└─ speech_accuracy.csv
```

`derivatives/` is created automatically by the MATLAB scripts when needed.

## Before running the scripts

This repository contains example scripts that must be adapted to the local environment before use.

In particular, please edit the following user-defined settings in each MATLAB script before running:
- `rootDir`
- `subList`

You should also confirm that the expected input file names and folder structure match your local dataset.

## Manual steps

This pipeline is not fully automated. Two manual review steps are required.

### Manual ICA review

After `stage1_prepare_ica.m`, open each file named:

```text
*_lp30_goodch_for_manualICA.set
```

Inspect the ICA components in EEGLAB, remove artifact-related components, and save the cleaned file as:

```text
*_goodch_icaclean_manual.set
```

### Manual epoch review

After `stage2a_epoch_and_mark.m`, open each marked epoch file in EEGLAB, review the automatic markings, add any manual markings if needed, and save the reviewed dataset.


This repository contains example scripts that must be adapted to the local environment before use.

In particular, please edit the following user-defined settings in each MATLAB script before running:
- `rootDir`
- `subList`

You should also confirm that the expected input file names and folder structure match your local dataset.

### MATLAB preprocessing

Edit `rootDir` and `subList` at the top of each script before running.

Run the MATLAB scripts in this order:

1. `matlab/stage1_prepare_ica.m`
2. Manual ICA review in EEGLAB
3. `matlab/stage2a_epoch_and_mark.m`
4. Manual epoch review in EEGLAB
5. `matlab/stage2b_finalize_epochs.m`
6. `matlab/make_run_qc_tables.m`
7. `matlab/export_anticipatory_long_noninterp.m`
8. `matlab/export_anticipatory_long_interp.m` (optional, supplementary)

### Python analysis

The Python script uses the non-interpolated long-format file for the primary analyses.

Example:

```bash
python python/run_spn_cnv_analysis.py \
    --input_dir ./data \
    --output_dir ./results \
    --metadata ./data/electrode_metadata.csv \
    --run_qc ./data/run_qc.csv \
    --speech ./data/speech_accuracy.csv
```

The required input is:

- `anticipatory_electrode_long_noninterp.csv`

Optional inputs are:

- `electrode_metadata.csv`
- `run_qc.csv`
- `speech_accuracy.csv`

If `electrode_metadata.csv` is not provided, ROI and hemisphere labels are inferred from channel names. If `run_qc.csv` is not provided, the script proceeds without QC filtering. If `speech_accuracy.csv` is not provided, speech-in-noise analyses are skipped.

## Primary and supplementary datasets

The repository distinguishes between primary and supplementary datasets.

- **Primary analysis** uses **non-interpolated** datasets.
- **Supplementary sensitivity analyses** may use **interpolated** datasets.

This distinction is preserved both in the MATLAB export scripts and in the Python analysis workflow.

## Input files

See `docs/input_file_specifications.md` for full details.

In brief:

- `participant_info.csv`
  - `Participant`
  - `Sex` (optional)

- Event CSV files
  - `latency`
  - `type`
  - `position`

- `electrode_metadata.csv`
  - `Electrode`
  - `ROI` (optional)
  - `Hemisphere` (optional)

- `speech_accuracy.csv`
  - `Participant`
  - `SNR_dB`
  - `Accuracy`
  - `n_trials`
  - `n_correct`

## Outputs

See `docs/output_file_specifications.md` for full details.

Common outputs include:

- `*_trial_log.csv`
- `run_qc.csv`
- `participant_qc.csv`
- `anticipatory_electrode_long_noninterp.csv`
- `anticipatory_electrode_long_interp.csv`
- mixed-effects model result tables
- descriptive tables
- topographic figures and ROI summary figures

## Notes on interpretation

- SPN is the primary target analysis.
- CNV is included as a response-locked control analysis.
- The main Python models use ROI × Hemisphere mixed-effects models.
- Sex-adjusted models are supplementary.
- Electrode-wise analyses are supplementary and use FDR correction.
- Speech-in-noise analyses are exploratory.
- Topographic maps are descriptive and are not used as standalone inferential tests.

## Data availability

The dataset used for the analyses in this repository is publicly available on Zenodo:
[https://doi.org/10.5281/zenodo.19344047](https://doi.org/10.5281/zenodo.19344047)

Please download the dataset from Zenodo before running the preprocessing or analysis scripts.

## License

This repository is distributed under the MIT License. See `LICENSE` for details.
