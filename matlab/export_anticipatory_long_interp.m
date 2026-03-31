%% =========================================================
%  export_anticipatory_long_interp.m
%
%  Export trial-wise SPN and CNV values from INTERPOLATED datasets
%  for supplementary sensitivity analyses.
%
%  Output:
%    anticipatory_electrode_long_interp.csv
%% =========================================================

clear; clc;

%% -----------------------------
% User settings
% ------------------------------
rootDir = 'C:\Users\USER\Documents\MATLAB\eegrecords01';
subList = 1:23;

participantInfoFile = fullfile(rootDir, 'participant_info.csv');

SPN_window_ms = [-200 0];
CNV_window_ms = [-500 -100];

allRows = table();

%% -----------------------------
% Read participant info if available
% ------------------------------
if isfile(participantInfoFile)
    Pinfo = readtable(participantInfoFile);
    Pinfo.Participant = string(Pinfo.Participant);
else
    Pinfo = table();
end

%% -----------------------------
% Loop over subjects
% ------------------------------
for subNum = subList
    subStr = sprintf('%02d', subNum);
    outDir = fullfile(rootDir, subStr, 'derivatives');

    if ~exist(outDir, 'dir')
        continue;
    end

    %% SPN files
    spnFiles = dir(fullfile(outDir, '*_spn_ep_interp_base.set'));

    for i = 1:numel(spnFiles)
        fullSet = fullfile(spnFiles(i).folder, spnFiles(i).name);

        EEG = pop_loadset(fullSet);
        EEG = eeg_checkset(EEG);

        runID = erase(spnFiles(i).name, '_spn_ep_interp_base.set');

        tok = regexp(runID, '^sub\d{2}([sn])\d$', 'tokens', 'once');
        if isempty(tok)
            condLabel = "unknown";
        elseif strcmp(tok{1}, 's')
            condLabel = "in silence";
        elseif strcmp(tok{1}, 'n')
            condLabel = "in noise";
        else
            condLabel = "unknown";
        end

        participant = string(subStr);

        sexVal = missing;
        if ~isempty(Pinfo)
            idxP = find(Pinfo.Participant == participant, 1);
            if ~isempty(idxP) && ismember('Sex', Pinfo.Properties.VariableNames)
                sexVal = string(Pinfo.Sex(idxP));
            end
        end

        tidx = find(EEG.times >= SPN_window_ms(1) & EEG.times <= SPN_window_ms(2));
        if isempty(tidx)
            warning('No SPN samples found in window for %s', runID);
            continue;
        end

        chanLabels = string({EEG.chanlocs.labels});

        for tr = 1:EEG.trials
            spnVals = mean(EEG.data(:, tidx, tr), 2, 'omitnan');

            T = table();
            T.Participant = repmat(participant, EEG.nbchan, 1);
            T.RunID       = repmat(string(runID), EEG.nbchan, 1);
            T.Trial       = repmat(tr, EEG.nbchan, 1);
            T.Condition   = repmat(condLabel, EEG.nbchan, 1);
            T.Component   = repmat("SPN", EEG.nbchan, 1);
            T.Electrode   = chanLabels(:);
            T.Value_uV    = spnVals(:);
            T.Sex         = repmat(string(sexVal), EEG.nbchan, 1);

            allRows = [allRows; T]; %#ok<AGROW>
        end
    end

    %% CNV files
    cnvFiles = dir(fullfile(outDir, '*_cnv_ep_interp_base.set'));

    for i = 1:numel(cnvFiles)
        fullSet = fullfile(cnvFiles(i).folder, cnvFiles(i).name);

        EEG = pop_loadset(fullSet);
        EEG = eeg_checkset(EEG);

        runID = erase(cnvFiles(i).name, '_cnv_ep_interp_base.set');

        tok = regexp(runID, '^sub\d{2}([sn])\d$', 'tokens', 'once');
        if isempty(tok)
            condLabel = "unknown";
        elseif strcmp(tok{1}, 's')
            condLabel = "in silence";
        elseif strcmp(tok{1}, 'n')
            condLabel = "in noise";
        else
            condLabel = "unknown";
        end

        participant = string(subStr);

        sexVal = missing;
        if ~isempty(Pinfo)
            idxP = find(Pinfo.Participant == participant, 1);
            if ~isempty(idxP) && ismember('Sex', Pinfo.Properties.VariableNames)
                sexVal = string(Pinfo.Sex(idxP));
            end
        end

        tidx = find(EEG.times >= CNV_window_ms(1) & EEG.times <= CNV_window_ms(2));
        if isempty(tidx)
            warning('No CNV samples found in window for %s', runID);
            continue;
        end

        chanLabels = string({EEG.chanlocs.labels});

        for tr = 1:EEG.trials
            cnvVals = mean(EEG.data(:, tidx, tr), 2, 'omitnan');

            T = table();
            T.Participant = repmat(participant, EEG.nbchan, 1);
            T.RunID       = repmat(string(runID), EEG.nbchan, 1);
            T.Trial       = repmat(tr, EEG.nbchan, 1);
            T.Condition   = repmat(condLabel, EEG.nbchan, 1);
            T.Component   = repmat("CNV", EEG.nbchan, 1);
            T.Electrode   = chanLabels(:);
            T.Value_uV    = cnvVals(:);
            T.Sex         = repmat(string(sexVal), EEG.nbchan, 1);

            allRows = [allRows; T]; %#ok<AGROW>
        end
    end
end

%% -----------------------------
% Save
% ------------------------------
if isempty(allRows)
    error('No rows were exported. Check *_spn_ep_interp_base.set and *_cnv_ep_interp_base.set files.');
end

outFile = fullfile(rootDir, 'anticipatory_electrode_long_interp.csv');
writetable(allRows, outFile);

disp(' ');
disp('==============================================');
disp('Saved:');
disp(outFile);
disp('==============================================');
