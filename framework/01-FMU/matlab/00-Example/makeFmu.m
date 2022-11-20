clear all
clc
clearvars

addpath(genpath('Script'))

Ts = 1e-6;
system = 'TestGrid';


open_system(system)
BusExample
set_param(system,'SolverType','Fixed-step')
set_param(system,'FixedStep',string(Ts))

set_param([system '/Measurement'],'OutDataTypeStr','Inherit: auto');
busInfo = Simulink.Bus.createObject(system,[system '/Measurement']);
set_param([system '/Measurement'],'OutDataTypeStr',['Bus: ' busInfo.busName]);

fprintf('Starting Build process ... \n')

exportToFMU2CS(system, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');
