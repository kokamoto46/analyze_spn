%% =========================================================
%  make_run_qc_tables.m
%
%  Collect trial logs and generate run-level and participant-level
%  quality-control tables for the primary non-interpolated analysis.
%
%  Output:
%    run_qc.csv
%    participant_qc.csv
%% =========================================================

clear; clc;

%% -----------------------------
% User settings
% ------------------------------
rootDir = 'C:\Users\USER\Documents\MATLAB\eegrecords01';
subList = 1:23;

%% -----------------------------
% Collect trial logs
% ------------------------------
allRows = table();

for subNum = subList
    subStr = sprintf('%02d', subNum);
    outDir = fullfile(rootDir, subStr, 'derivatives');

    if ~exist(outDir, 'dir')
        continue;
    end

    files = dir(fullfile(outDir, '*_trial_log.csv'));
    for i = 1:numel(files)
        T = readtable(fullfile(files(i).folder, files(i).name));
        allRows = [allRows; T]; %#ok<AGROW>
    end
end

if isempty(allRows)
    error('No *_trial_log.csv files were found. Run Stage 2B first.');
end

%% -----------------------------
% Normalize types
% ------------------------------
allRows.participant = string(allRows.participant);
allRows.runID = string(allRows.runID);

requiredCols = {'participant','runID','n_bad_channels','exclude_run_primary'};
missingCols = requiredCols(~ismember(requiredCols, allRows.Properties.VariableNames));
if ~isempty(missingCols)
    error('Missing required columns in trial logs: %s', strjoin(missingCols, ', '));
end

%% -----------------------------
% Run-level keep flag
% ------------------------------
allRows.keep_primary_run = ~logical(allRows.exclude_run_primary);

%% -----------------------------
% Participant-level summary
% ------------------------------
[G, participantNames] = findgroups(allRows.participant);
bads = allRows.n_bad_channels;

mean_bad = splitapply(@mean, bads, G);
sd_bad   = splitapply(@std,  bads, G);
min_bad  = splitapply(@min,  bads, G);
max_bad  = splitapply(@max,  bads, G);
n_runs_available = splitapply(@numel, bads, G);

all_ge7 = splitapply(@(x) all(x >= 7), bads, G);
any_ge10 = splitapply(@(x) any(x >= 10), bads, G);

participantQC = table();
participantQC.participant = string(participantNames);
participantQC.mean_bad = mean_bad;
participantQC.sd_bad = sd_bad;
participantQC.min_bad = min_bad;
participantQC.max_bad = max_bad;
participantQC.n_runs_available = n_runs_available;
participantQC.all_runs_ge7 = all_ge7;
participantQC.any_run_ge10 = any_ge10;

participantQC.exclude_primary_participant = participantQC.all_runs_ge7;
participantQC.keep_primary_participant = ~participantQC.exclude_primary_participant;

%% -----------------------------
% Merge participant QC into run-level QC
% ------------------------------
runQC = outerjoin( ...
    allRows, ...
    participantQC(:, {'participant','keep_primary_participant','exclude_primary_participant'}), ...
    'Keys', 'participant', ...
    'MergeKeys', true);

runQC.keep_primary = logical(runQC.keep_primary_run) & logical(runQC.keep_primary_participant);

%% -----------------------------
% Sort and save
% ------------------------------
if ismember('condition', runQC.Properties.VariableNames)
    runQC = sortrows(runQC, {'participant','condition','runID'});
else
    runQC = sortrows(runQC, {'participant','runID'});
end

participantQC = sortrows(participantQC, 'participant');

runQCFile = fullfile(rootDir, 'run_qc.csv');
participantQCFile = fullfile(rootDir, 'participant_qc.csv');

writetable(runQC, runQCFile);
writetable(participantQC, participantQCFile);

disp(' ');
disp('==============================================');
disp('QC tables saved:');
disp(runQCFile);
disp(participantQCFile);
disp('==============================================');
