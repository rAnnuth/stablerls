clear all
clc
clearvars
restoredefaultpath;

filePath = matlab.desktop.editor.getActiveFilename;
[fPath, ~, ~] = fileparts(filePath);
cd(fPath);
addpath(pwd)
mainDir = fullfile('..',filesep(),'..',filesep());
addpath(genpath(fullfile(mainDir,'Script')))
addpath(genpath(fullfile(mainDir,'Submodel')))


Ts = 1e-3;
TestFMU(Ts)

function TestFMU(Ts)
mdl =  'TestFMU';

gitCommit({'dummyFolder'},mdl)
open_system(mdl)
FMUBus
set_param(mdl,'SolverType','Fixed-step')
set_param(mdl,'FixedStep',string(Ts))
exportToFMU2CS(mdl, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');

end