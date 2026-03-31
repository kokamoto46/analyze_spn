# Example commands

## Python

```bash
pip install -r requirements.txt

python python/run_spn_cnv_analysis.py \
    --input_dir ./data \
    --output_dir ./results \
    --metadata ./data/electrode_metadata.csv \
    --run_qc ./data/run_qc.csv \
    --speech ./data/speech_accuracy.csv
```

## MATLAB

Open MATLAB, edit `rootDir` and `subList` inside each script, then run:

1. `matlab/stage1_prepare_ica.m`
2. Manual ICA review in EEGLAB
3. `matlab/stage2a_epoch_and_mark.m`
4. Manual epoch review in EEGLAB
5. `matlab/stage2b_finalize_epochs.m`
6. `matlab/make_run_qc_tables.m`
7. `matlab/export_anticipatory_long_noninterp.m`
8. `matlab/export_anticipatory_long_interp.m`
