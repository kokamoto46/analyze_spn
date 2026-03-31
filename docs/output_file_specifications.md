# Output file specifications

## MATLAB outputs

### Stage 1

- `*_raw_loaded.set`
- `*_chanlocs.set`
- `*_events_imported.set`
- `*_event_import_log.csv`
- `*_lp30.set`
- `*_lp30_prep.set`
- `*_bad_channels_log.csv`
- `*_ica_source_goodch.set`
- `*_ica1hz_goodch.set`
- `*_ica1hz_goodch_runica.set`
- `*_lp30_goodch_for_manualICA.set`

### Stage 2A

- `*_spn_ep_goodch_marked_afterICA.set`
- `*_cnv_ep_goodch_marked_afterICA.set`

### Stage 2B

- `*_spn_ep_noninterp_base.set`
- `*_cnv_ep_noninterp_base.set`
- `*_spn_ep_interp_base.set` (optional)
- `*_cnv_ep_interp_base.set` (optional)
- `*_trial_log.csv`

### QC and export

- `run_qc.csv`
- `participant_qc.csv`
- `anticipatory_electrode_long_noninterp.csv`
- `anticipatory_electrode_long_interp.csv`

## Python outputs

Tables are saved to `output_dir/tables/`.

Common outputs include:
- `Table_Primary_SPN_ROI_Hemisphere_LMM.csv`
- `Table_Control_CNV_ROI_Hemisphere_LMM.csv`
- `Supp_Table_SPN_LMM_withSex.csv`
- `Supp_Table_CNV_LMM_withSex.csv`
- `Supp_Table_SPN_ElectrodeWise_LMM_FDR.csv`
- `Supp_Table_CNV_ElectrodeWise_LMM_FDR.csv`
- `Table_Descriptive_ROI_ByCondition_SPN.csv`
- `Table_Descriptive_ROI_ByCondition_CNV.csv`
- `Table1_SpeechAccuracy_BySNR.csv` (optional)
- `Supp_Table_SpeechAccuracy_ParticipantSummary.csv` (optional)
- `Supp_Table_ParticipantSummary_SPN_CNV_Speech.csv` (optional)
- `Supp_Table_Exploratory_SpeechBrain_Relations.csv` (optional)

Figures are saved to `output_dir/figures/`.

Common outputs include:
- `Fig2a_SPN_topomap_in_silence.png`
- `Fig2b_SPN_topomap_in_noise.png`
- `Fig2c_SPN_topomap_difference.png`
- `Fig3_SPN_roi_panels_main.png`
- `Supp_Fig_CNV_roi_panels.png`
- `Supp_Fig_SpeechAccuracy_0dB.png` (optional)

Additional text output:
- `ANALYSIS_NOTES_SPN_CNV_REVISED.txt`
