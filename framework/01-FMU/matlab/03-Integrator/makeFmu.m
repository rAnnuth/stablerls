clear all
clc
%clearvars
Simulink.data.dictionary.closeAll('-discard')

Ts = 1e-2;
system = 'rlwatertank';


open_system(system)
makeBus
set_param(system,'SolverType','Fixed-step')
set_param(system,'FixedStep',string(Ts))

set_param([system '/Measurement'],'OutDataTypeStr','Inherit: auto');
busInfo = Simulink.Bus.createObject(system,[system '/Measurement']);
set_param([system '/Measurement'],'OutDataTypeStr',['Bus: ' busInfo.busName]);

fprintf('Starting Build process ... \n')

exportToFMU2CS(system, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');
