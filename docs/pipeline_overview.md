# Pipeline overview

This repository provides a semi-automated preprocessing and analysis workflow for SPN and CNV EEG data.

## MATLAB workflow

The MATLAB component uses EEGLAB-based preprocessing in three stages.

### Stage 1

- Load raw EDF data
- Assign standard channel locations
- Import event information from CSV files
- Apply 30 Hz low-pass filtering
- Detect bad channels with PREP
- Run ICA on good EEG channels
- Save datasets for manual ICA inspection

### Stage 2A

- Load manually ICA-cleaned data
- Epoch SPN and CNV events
- Apply threshold-based automatic marking
- Save files for manual epoch review

### Stage 2B

- Load reviewed epoched files
- Reject marked epochs
- Apply baseline correction
- Save non-interpolated primary datasets
- Optionally interpolate previously identified bad channels
- Save interpolated supplementary datasets
- Export run-level trial logs

### Post-processing

- Aggregate run-level trial logs into QC tables
- Export non-interpolated long-format CSV files
- Optionally export interpolated long-format CSV files for sensitivity analyses

## Python workflow

The Python script consumes the long-format non-interpolated export and performs:

- QC filtering based on `run_qc.csv`
- ROI × Hemisphere mixed-effects models for SPN and CNV
- Optional sex-adjusted supplementary models
- Electrode-wise mixed-effects models with FDR correction
- Descriptive topographic plots
- ROI summary figures
- Optional speech-in-noise summaries and exploratory correlations
