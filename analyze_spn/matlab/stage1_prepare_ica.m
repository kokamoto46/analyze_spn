%% =========================================================
%  stage1_prepare_ica.m
%
%  SPN/CNV preprocessing pipeline: Stage 1
%
%  This script performs the following steps for all participants
%  and all predefined blocks:
%    1) Load raw EDF data
%    2) Assign channel locations
%    3) Import SPN and CNV event CSV files
%    4) Apply low-pass filtering at 30 Hz
%    5) Exclude non-EEG channels
%    6) Run the PREP pipeline to detect bad channels
%    7) Save bad-channel logs
%    8) Create a good-channel dataset for ICA
%    9) High-pass filter at 1 Hz for ICA
%   10) Run ICA
%   11) Transfer ICA weights back to the low-frequency-preserving dataset
%
%  Output:
%    *_lp30_goodch_for_manualICA.set
%
%  Next step:
%    Open each saved dataset in EEGLAB, remove artifact ICs manually,
%    and save the cleaned file as:
%    *_goodch_icaclean_manual.set
%% =========================================================

clear; clc;

%% -----------------------------
% User settings
% ------------------------------
rootDir = 'C:\Users\USER\Documents\MATLAB\eegrecords01';
subList = 1:23;

lineNoiseMethod = 'none';

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
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    for b = 1:size(blockDefs, 1)

        blockCode = blockDefs{b, 1};
        spnEventFileName = blockDefs{b, 2};
        cnvEventFileName = blockDefs{b, 3};
        runID = ['sub' subStr blockCode];

        rawEDFFile = fullfile(subFolder, [runID '_raw.edf']);
        spnEventCSV = fullfile(subFolder, spnEventFileName);
        cnvEventCSV = fullfile(subFolder, cnvEventFileName);

        fprintf('\n==============================================\n');
        fprintf('Stage 1 processing: %s\n', runID);
        fprintf('==============================================\n');

        if ~isfile(rawEDFFile)
            warning('Raw EDF file not found: %s', rawEDFFile);
            continue;
        end

        if ~isfile(spnEventCSV)
            warning('SPN event CSV not found: %s', spnEventCSV);
            continue;
        end

        if ~isfile(cnvEventCSV)
            warning('CNV event CSV not found: %s', cnvEventCSV);
            continue;
        end

        try
            %% 1) Load raw EEG from EDF
            EEG = pop_biosig(rawEDFFile);
            EEG.setname = [runID '_raw_loaded'];
            EEG = eeg_checkset(EEG);
            pop_saveset(EEG, 'filename', [EEG.setname '.set'], 'filepath', outDir);

            %% 2) Assign channel locations
            try
                EEG = pop_chanedit(EEG, 'lookup', 'standard-10-5-cap385.elp');
            catch
                warning('Could not load standard-10-5-cap385.elp automatically for %s.', runID);
            end

            EEG.setname = [runID '_chanlocs'];
            EEG = eeg_checkset(EEG);
            pop_saveset(EEG, 'filename', [EEG.setname '.set'], 'filepath', outDir);

            %% 3) Import SPN and CNV events from CSV
            Tspn = readtable(spnEventCSV);
            Tcnv = readtable(cnvEventCSV);

            requiredCols = {'latency','type','position'};
            if ~all(ismember(requiredCols, Tspn.Properties.VariableNames))
                error('SPN event CSV must contain columns: latency, type, position');
            end
            if ~all(ismember(requiredCols, Tcnv.Properties.VariableNames))
                error('CNV event CSV must contain columns: latency, type, position');
            end

            Tspn = Tspn(~isnan(Tspn.latency), :);
            Tcnv = Tcnv(~isnan(Tcnv.latency), :);

            lat_spn = round(Tspn.latency * EEG.srate) + 1;
            lat_cnv = round(Tcnv.latency * EEG.srate) + 1;

            valid_spn = lat_spn >= 1 & lat_spn <= EEG.pnts;
            valid_cnv = lat_cnv >= 1 & lat_cnv <= EEG.pnts;

            Tspn = Tspn(valid_spn, :);
            Tcnv = Tcnv(valid_cnv, :);
            lat_spn = lat_spn(valid_spn);
            lat_cnv = lat_cnv(valid_cnv);

            Tall = table();
            Tall.latency_sec = [Tspn.latency; Tcnv.latency];
            Tall.latency_samples = [lat_spn; lat_cnv];
            Tall.type = [string(Tspn.type); string(Tcnv.type)];
            Tall.position = [string(Tspn.position); string(Tcnv.position)];
            Tall = sortrows(Tall, 'latency_samples');

            if isempty(Tall)
                error('No valid SPN/CNV events remained after conversion to sample points.');
            end

            EEG.event = struct('type', {}, 'latency', {}, 'urevent', {});
            for i = 1:height(Tall)
                EEG.event(i).type    = char(Tall.position(i));
                EEG.event(i).latency = Tall.latency_samples(i);
                EEG.event(i).urevent = i;
            end

            EEG = eeg_checkset(EEG, 'eventconsistency');
            EEG.setname = [runID '_events_imported'];
            pop_saveset(EEG, 'filename', [EEG.setname '.set'], 'filepath', outDir);

            eventLog = Tall;
            writetable(eventLog, fullfile(outDir, [runID '_event_import_log.csv']));

            %% 4) Low-pass filter at 30 Hz
            EEG_lp = pop_eegfiltnew(EEG, [], 30);
            EEG_lp.setname = [runID '_lp30'];
            EEG_lp = eeg_checkset(EEG_lp);
            pop_saveset(EEG_lp, 'filename', [EEG_lp.setname '.set'], 'filepath', outDir);

            %% 5) Exclude non-EEG channels
            lab = string({EEG_lp.chanlocs.labels});
            hasLoc = ~cellfun(@isempty, {EEG_lp.chanlocs.X});

            excludeLabels = ["Trig","Trigger","TRG","Status","DIN","Stim", ...
                             "EOG","VEOG","HEOG","AUX","ECG"];
            isExt = false(size(lab));

            for k = 1:numel(excludeLabels)
                isExt = isExt | startsWith(lab, excludeLabels(k), 'IgnoreCase', true) ...
                              | strcmpi(lab, excludeLabels(k));
            end

            evalCh = find(hasLoc & ~isExt);

            if isempty(evalCh)
                error('No EEG channels remained after excluding non-EEG channels.');
            end

            %% 6) Run PREP
            P = struct( ...
                'lineNoiseMethod'     , lineNoiseMethod, ...
                'referenceType'       , 'robust', ...
                'evaluationChannels'  , evalCh, ...
                'rereferencedChannels', evalCh, ...
                'cleanupReference'    , false, ...
                'makeSummary'         , false);

            EEG_prep = EEG_lp;
            [EEG_prep,~,~] = prepPipeline(EEG_prep, P);
            EEG_prep.setname = [runID '_lp30_prep'];
            EEG_prep = eeg_checkset(EEG_prep);
            pop_saveset(EEG_prep, 'filename', [EEG_prep.setname '.set'], 'filepath', outDir);

            %% 7) Extract bad channels
            nd = EEG_prep.etc.noiseDetection;

            if isfield(nd, 'interpolatedChannelNumbers')
                interpIdx = nd.interpolatedChannelNumbers;
            else
                interpIdx = [];
            end

            if isfield(nd, 'removedChannelNumbers')
                removedIdx = nd.removedChannelNumbers;
            else
                removedIdx = [];
            end

            if isfield(nd, 'stillNoisyChannelNumbers')
                stillNoisyIdx = nd.stillNoisyChannelNumbers;
            else
                stillNoisyIdx = [];
            end

            allBadIdx = unique([interpIdx(:); removedIdx(:); stillNoisyIdx(:)])';

            if isempty(allBadIdx)
                allBadLabels = {};
            else
                allBadLabels = {EEG_lp.chanlocs(allBadIdx).labels};
            end

            badLog = table;
            badLog.runID      = string(runID);
            badLog.n_bad      = numel(allBadIdx);
            badLog.bad_labels = string(strjoin(allBadLabels, ','));
            badLog.bad_prop   = numel(allBadIdx) / numel(evalCh);

            writetable(badLog, fullfile(outDir, [runID '_bad_channels_log.csv']));

            %% 8) Create ICA source dataset using only good EEG channels
            icaKeepIdx = setdiff(evalCh, allBadIdx);

            if isempty(icaKeepIdx)
                error('No good EEG channels remain for ICA.');
            end

            EEG_icaSrc = pop_select(EEG_lp, 'channel', icaKeepIdx);
            EEG_icaSrc.setname = [runID '_ica_source_goodch'];
            EEG_icaSrc = eeg_checkset(EEG_icaSrc);
            pop_saveset(EEG_icaSrc, 'filename', [EEG_icaSrc.setname '.set'], 'filepath', outDir);

            %% 9) High-pass filter at 1 Hz for ICA
            EEG_ica1 = pop_eegfiltnew(EEG_icaSrc, 1, []);
            EEG_ica1.setname = [runID '_ica1hz_goodch'];
            EEG_ica1 = eeg_checkset(EEG_ica1);
            pop_saveset(EEG_ica1, 'filename', [EEG_ica1.setname '.set'], 'filepath', outDir);

            %% 10) Run ICA
            EEG_ica1 = pop_runica(EEG_ica1, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
            EEG_ica1.setname = [runID '_ica1hz_goodch_runica'];
            EEG_ica1 = eeg_checkset(EEG_ica1);
            pop_saveset(EEG_ica1, 'filename', [EEG_ica1.setname '.set'], 'filepath', outDir);

            %% 11) Transfer ICA weights to low-frequency-preserving dataset
            EEG_good_lp = pop_select(EEG_lp, 'channel', icaKeepIdx);
            EEG_good_lp.setname = [runID '_lp30_goodch_for_manualICA'];

            EEG_good_lp.icaweights  = EEG_ica1.icaweights;
            EEG_good_lp.icasphere   = EEG_ica1.icasphere;
            EEG_good_lp.icawinv     = EEG_ica1.icawinv;
            EEG_good_lp.icachansind = EEG_ica1.icachansind;

            EEG_good_lp = eeg_checkset(EEG_good_lp);
            pop_saveset(EEG_good_lp, 'filename', [EEG_good_lp.setname '.set'], 'filepath', outDir);

            fprintf('Stage 1 complete for %s\n', runID);

        catch ME
            warning('Stage 1 failed for %s: %s', runID, ME.message);
            continue;
        end
    end
end

disp(' ');
disp('==============================================');
disp('Stage 1 finished for all available runs.');
disp('Next: manually remove artifact ICs and save each file as');
disp('*_goodch_icaclean_manual.set');
disp('==============================================');
