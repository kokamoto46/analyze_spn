%% =========================================================
%  stage2a_epoch_and_mark.m
%
%  SPN/CNV preprocessing pipeline: Stage 2A
%
%  This script loads the manually ICA-cleaned datasets and performs:
%    1) SPN epoching
%    2) SPN automatic threshold marking
%    3) CNV epoching
%    4) CNV automatic threshold marking
%
%  Output:
%    *_spn_ep_goodch_marked_afterICA.set
%    *_cnv_ep_goodch_marked_afterICA.set
%
%  Next step:
%    Review these epoched datasets manually in EEGLAB and apply any
%    additional epoch marking if needed before running Stage 2B.
%% =========================================================

clear; clc;

%% -----------------------------
% User settings
% ------------------------------
rootDir = 'C:\Users\USER\Documents\MATLAB\eegrecords01';
subList = 1:23;

SPN_event     = 'S2';
SPN_epoch     = [-2.0 0.3];
SPN_thresh_uV = 100;

CNV_event     = 'RESPONSE';
CNV_epoch     = [-1.2 0.2];
CNV_thresh_uV = 100;

blockDefs = {
    's1', 'spn_silence1event.csv', 'cnv_silence1event.csv';
    's2', 'spn_silence2event.csv', 'cnv_silence2event.csv';
    'n1', 'spn_noise1event.csv',   'cnv_noise1event.csv';
    'n2', 'spn_noise2event.csv',   'cnv_noise2event.csv'
};

%% -----------------------------
% Start EEGLAB
% ------------------------------
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; %#ok<ASGLU>

for subNum = subList

    subStr = sprintf('%02d', subNum);
    subFolder = fullfile(rootDir, subStr);

    if ~exist(subFolder, 'dir')
        warning('Subject folder not found: %s', subFolder);
        continue;
    end

    outDir = fullfile(subFolder, 'derivatives');

    for b = 1:size(blockDefs, 1)

        blockCode = blockDefs{b, 1};
        runID = ['sub' subStr blockCode];
        icaCleanFile = fullfile(outDir, [runID '_goodch_icaclean_manual.set']);

        fprintf('\n==============================================\n');
        fprintf('Stage 2A processing: %s\n', runID);
        fprintf('==============================================\n');

        if ~isfile(icaCleanFile)
            warning('ICA-cleaned file not found: %s', icaCleanFile);
            continue;
        end

        try
            %% 1) Load manually ICA-cleaned dataset
            EEG_good_clean = pop_loadset(icaCleanFile);
            EEG_good_clean.setname = [runID '_goodch_icaclean_manual_loaded'];
            EEG_good_clean = eeg_checkset(EEG_good_clean);

            %% 2) SPN epoching + automatic threshold marking
            EEG_spn = pop_epoch(EEG_good_clean, {SPN_event}, SPN_epoch, 'epochinfo', 'yes');
            EEG_spn = eeg_checkset(EEG_spn);

            EEG_spn = pop_eegthresh(EEG_spn, 1, 1:EEG_spn.nbchan, ...
                                    -SPN_thresh_uV, SPN_thresh_uV, ...
                                    SPN_epoch(1), SPN_epoch(2) - 1/EEG_spn.srate, ...
                                    0, 1);

            EEG_spn.setname = [runID '_spn_ep_goodch_marked_afterICA'];
            EEG_spn = eeg_checkset(EEG_spn);
            pop_saveset(EEG_spn, 'filename', [EEG_spn.setname '.set'], 'filepath', outDir);

            %% 3) CNV epoching + automatic threshold marking
            EEG_cnv = pop_epoch(EEG_good_clean, {CNV_event}, CNV_epoch, 'epochinfo', 'yes');
            EEG_cnv = eeg_checkset(EEG_cnv);

            EEG_cnv = pop_eegthresh(EEG_cnv, 1, 1:EEG_cnv.nbchan, ...
                                    -CNV_thresh_uV, CNV_thresh_uV, ...
                                    CNV_epoch(1), CNV_epoch(2) - 1/EEG_cnv.srate, ...
                                    0, 1);

            EEG_cnv.setname = [runID '_cnv_ep_goodch_marked_afterICA'];
            EEG_cnv = eeg_checkset(EEG_cnv);
            pop_saveset(EEG_cnv, 'filename', [EEG_cnv.setname '.set'], 'filepath', outDir);

            fprintf('Stage 2A complete for %s\n', runID);

        catch ME
            warning('Stage 2A failed for %s: %s', runID, ME.message);
            continue;
        end
    end
end

disp(' ');
disp('==============================================');
disp('Stage 2A finished for all available runs.');
disp('Next: manually review the marked epoch files in EEGLAB.');
disp('==============================================');
