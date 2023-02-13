clear all
clc
Simulink.data.dictionary.closeAll('-discard')

Ts = 1e-6; %sampling time of fmu
system = 'SimulinkModel';

% open system and create port structure
open_system(system)
set_param(system,'DataDictionary','BusSystem.sldd');
getports(system)

% setup the required solver
set_param(system,'SolverType','Fixed-step')
set_param(system,'FixedStep',string(Ts))

% build FMU
fprintf('Starting Build process ... \n')
exportToFMU2CS(system, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');
