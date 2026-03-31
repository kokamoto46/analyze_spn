%% =========================================================
%  stage2b_finalize_epochs.m
%
%  SPN/CNV preprocessing pipeline: Stage 2B
%
%  This script loads the reviewed epoch files and performs:
%    1) Alignment of rejection flags to the current number of trials
%    2) Rejection of marked epochs
%    3) Baseline correction for primary non-interpolated datasets
%    4) Optional bad-channel interpolation for supplementary datasets
%    5) Saving final .set files
%    6) Writing a trial log for each run
%
%  Output:
%    *_spn_ep_noninterp_base.set
%    *_cnv_ep_noninterp_base.set
%    *_spn_ep_interp_base.set          (optional)
%    *_cnv_ep_interp_base.set          (optional)
%    *_trial_log.csv
%% =========================================================

clear; clc;

%% -----------------------------
% User settings
% ------------------------------
rootDir = 'C:\Users\USER\Documents\MATLAB\eegrecords01';
subList = 1:23;

SPN_base_ms = [-1500 -1200];
CNV_base_ms = [-1200 -1000];

makeInterpolatedSupplementary = true;

blockList = {'s1','s2','n1','n2'};

%% -----------------------------
% Start EEGLAB
% ------------------------------
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; %#ok<ASGLU>

for subNum = subList

    subStr = sprintf('%02d', subNum);
    outDir = fullfile(rootDir, subStr, 'derivatives');

    fprintf('\n==================================================\n');
    fprintf('Processing participant %s\n', subStr);
    fprintf('==================================================\n');

    if ~exist(outDir, 'dir')
        warning('No derivatives folder for participant %s', subStr);
        continue;
    end

    for b = 1:numel(blockList)

        blockCode = blockList{b};
        runID = ['sub' subStr blockCode];

        fprintf('\n----------------------------------------------\n');
        fprintf('Stage 2B: %s\n', runID);
        fprintf('----------------------------------------------\n');

        spnMarkedFile = fullfile(outDir, [runID '_spn_ep_goodch_marked_afterICA.set']);
        cnvMarkedFile = fullfile(outDir, [runID '_cnv_ep_goodch_marked_afterICA.set']);
        lpRefFile     = fullfile(outDir, [runID '_lp30.set']);
        badLogFile    = fullfile(outDir, [runID '_bad_channels_log.csv']);

        if ~isfile(spnMarkedFile) || ~isfile(cnvMarkedFile) || ~isfile(lpRefFile) || ~isfile(badLogFile)
            warning('Missing required input files for %s', runID);
            continue;
        end

        try
            %% Load datasets
            EEG_spn = pop_loadset(spnMarkedFile);
            EEG_spn = eeg_checkset(EEG_spn);

            EEG_cnv = pop_loadset(cnvMarkedFile);
            EEG_cnv = eeg_checkset(EEG_cnv);

            EEG_lp = pop_loadset(lpRefFile);
            EEG_lp = eeg_checkset(EEG_lp);

            %% Read bad-channel log
            badLog = readtable(badLogFile);

            if height(badLog) < 1
                error('Bad-channel log is empty.');
            end

            if strlength(string(badLog.bad_labels(1))) == 0
                allBadLabels = {};
            else
                allBadLabels = strsplit(char(badLog.bad_labels(1)), ',');
            end

            allBadLabels = allBadLabels(~cellfun(@isempty, allBadLabels));

            if isempty(allBadLabels)
                allBadIdx = [];
            else
                allLabels = {EEG_lp.chanlocs.labels};
                [tf, idx] = ismember(allBadLabels, allLabels);
                allBadIdx = idx(tf);
            end

            %% SPN
            rejMask_spn = false(1, EEG_spn.trials);

            if isfield(EEG_spn.reject, 'rejmanual') && ~isempty(EEG_spn.reject.rejmanual)
                rejManual_spn = logical(EEG_spn.reject.rejmanual(:))';
                if numel(rejManual_spn) > EEG_spn.trials
                    rejManual_spn = rejManual_spn(1:EEG_spn.trials);
                elseif numel(rejManual_spn) < EEG_spn.trials
                    rejManual_spn = [rejManual_spn false(1, EEG_spn.trials - numel(rejManual_spn))];
                end
                rejMask_spn = rejMask_spn | rejManual_spn;
            end

            if isfield(EEG_spn.reject, 'rejthresh') && ~isempty(EEG_spn.reject.rejthresh)
                rejThresh_spn = logical(EEG_spn.reject.rejthresh(:))';
                if numel(rejThresh_spn) > EEG_spn.trials
                    rejThresh_spn = rejThresh_spn(1:EEG_spn.trials);
                elseif numel(rejThresh_spn) < EEG_spn.trials
                    rejThresh_spn = [rejThresh_spn false(1, EEG_spn.trials - numel(rejThresh_spn))];
                end
                rejMask_spn = rejMask_spn | rejThresh_spn;
            end

            if sum(rejMask_spn) >= EEG_spn.trials
                warning('All SPN trials would be rejected for %s. Skipping run.', runID);
                continue;
            end

            EEG_spn_clean = pop_rejepoch(EEG_spn, rejMask_spn, 0);
            EEG_spn_clean = eeg_checkset(EEG_spn_clean);

            EEG_spn_primary = EEG_spn_clean;
            EEG_spn_primary = pop_rmbase(EEG_spn_primary, SPN_base_ms);
            EEG_spn_primary.setname = [runID '_spn_ep_noninterp_base'];
            EEG_spn_primary = eeg_checkset(EEG_spn_primary);
            pop_saveset(EEG_spn_primary, 'filename', [EEG_spn_primary.setname '.set'], 'filepath', outDir);

            if makeInterpolatedSupplementary
                EEG_spn_interp = EEG_spn_clean;
                if ~isempty(allBadIdx)
                    EEG_spn_interp = eeg_interp(EEG_spn_interp, EEG_lp.chanlocs(allBadIdx), 'spherical');
                end
                EEG_spn_interp = pop_rmbase(EEG_spn_interp, SPN_base_ms);
                EEG_spn_interp.setname = [runID '_spn_ep_interp_base'];
                EEG_spn_interp = eeg_checkset(EEG_spn_interp);
                pop_saveset(EEG_spn_interp, 'filename', [EEG_spn_interp.setname '.set'], 'filepath', outDir);
            end

            %% CNV
            rejMask_cnv = false(1, EEG_cnv.trials);

            if isfield(EEG_cnv.reject, 'rejmanual') && ~isempty(EEG_cnv.reject.rejmanual)
                rejManual_cnv = logical(EEG_cnv.reject.rejmanual(:))';
                if numel(rejManual_cnv) > EEG_cnv.trials
                    rejManual_cnv = rejManual_cnv(1:EEG_cnv.trials);
                elseif numel(rejManual_cnv) < EEG_cnv.trials
                    rejManual_cnv = [rejManual_cnv false(1, EEG_cnv.trials - numel(rejManual_cnv))];
                end
                rejMask_cnv = rejMask_cnv | rejManual_cnv;
            end

            if isfield(EEG_cnv.reject, 'rejthresh') && ~isempty(EEG_cnv.reject.rejthresh)
                rejThresh_cnv = logical(EEG_cnv.reject.rejthresh(:))';
                if numel(rejThresh_cnv) > EEG_cnv.trials
                    rejThresh_cnv = rejThresh_cnv(1:EEG_cnv.trials);
                elseif numel(rejThresh_cnv) < EEG_cnv.trials
                    rejThresh_cnv = [rejThresh_cnv false(1, EEG_cnv.trials - numel(rejThresh_cnv))];
                end
                rejMask_cnv = rejMask_cnv | rejThresh_cnv;
            end

            if sum(rejMask_cnv) >= EEG_cnv.trials
                warning('All CNV trials would be rejected for %s. Skipping run.', runID);
                continue;
            end

            EEG_cnv_clean = pop_rejepoch(EEG_cnv, rejMask_cnv, 0);
            EEG_cnv_clean = eeg_checkset(EEG_cnv_clean);

            EEG_cnv_primary = EEG_cnv_clean;
            EEG_cnv_primary = pop_rmbase(EEG_cnv_primary, CNV_base_ms);
            EEG_cnv_primary.setname = [runID '_cnv_ep_noninterp_base'];
            EEG_cnv_primary = eeg_checkset(EEG_cnv_primary);
            pop_saveset(EEG_cnv_primary, 'filename', [EEG_cnv_primary.setname '.set'], 'filepath', outDir);

            if makeInterpolatedSupplementary
                EEG_cnv_interp = EEG_cnv_clean;
                if ~isempty(allBadIdx)
                    EEG_cnv_interp = eeg_interp(EEG_cnv_interp, EEG_lp.chanlocs(allBadIdx), 'spherical');
                end
                EEG_cnv_interp = pop_rmbase(EEG_cnv_interp, CNV_base_ms);
                EEG_cnv_interp.setname = [runID '_cnv_ep_interp_base'];
                EEG_cnv_interp = eeg_checkset(EEG_cnv_interp);
                pop_saveset(EEG_cnv_interp, 'filename', [EEG_cnv_interp.setname '.set'], 'filepath', outDir);
            end

            %% Save trial log
            trialLog = table;
            trialLog.runID               = string(runID);
            trialLog.participant         = string(subStr);
            trialLog.condition           = string(blockCode(1));
            trialLog.n_bad_channels      = numel(allBadIdx);
            trialLog.bad_channels        = string(strjoin(allBadLabels, ','));

            trialLog.n_spn_epochs_before = EEG_spn.trials;
            trialLog.n_spn_epochs_rej    = sum(rejMask_spn);
            trialLog.n_spn_epochs_final  = EEG_spn_primary.trials;

            trialLog.n_cnv_epochs_before = EEG_cnv.trials;
            trialLog.n_cnv_epochs_rej    = sum(rejMask_cnv);
            trialLog.n_cnv_epochs_final  = EEG_cnv_primary.trials;

            trialLog.flag_bad_ge7        = numel(allBadIdx) >= 7;
            trialLog.exclude_run_primary = numel(allBadIdx) >= 10;

            outCsv = fullfile(outDir, [runID '_trial_log.csv']);
            writetable(trialLog, outCsv);

        catch ME
            fprintf('\nERROR in %s\n', runID);
            fprintf('Message: %s\n', ME.message);
            fprintf('Identifier: %s\n', ME.identifier);
            for k = 1:numel(ME.stack)
                fprintf('  -> %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
            end
        end
    end
end

disp(' ');
disp('==================================================');
disp('Stage 2B finished for all subjects.');
disp('Next: run make_run_qc_tables.m and export scripts.');
disp('==================================================');
