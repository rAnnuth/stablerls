clear all
clc
Simulink.data.dictionary.closeAll('-discard')

Ts = 1e-6; %sampling time of fmu
system = 'SimulinkModel';

% open system and create port structure
open_system(system)
getports(system)
set_param(system,'DataDictionary','BusSystem.sldd');

% setup the required solver
set_param(system,'SolverType','Fixed-step')
set_param(system,'FixedStep',string(Ts))

% let matlab calculate the output bus
set_param([system '/Measurement'],'OutDataTypeStr','Inherit: auto');
busInfo = Simulink.Bus.createObject(system,[system '/Measurement']);
set_param([system '/Measurement'],'OutDataTypeStr',['Bus: ' busInfo.busName]);

% build FMU
fprintf('Starting Build process ... \n')
exportToFMU2CS(system, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');
