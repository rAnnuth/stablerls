%% basic startup
clear all
clc
clearvars
Simulink.data.dictionary.closeAll
restoredefaultpath;
addpath(genpath('Script'))

Simulink.sdi.clear
sdi.Repository.clearRepositoryFile
Simulink.sdi.setArchiveRunLimit(0)
Simulink.sdi.setAutoArchiveMode(false)
    
% simulation config
% Microgrid, AidaDeck
SystemName = 'Microgrid';
SimulationName = [SystemName 'Vessel'];

% see Script/Config.m
Config

%%


if strcmp(SimulationName,'MicrogridVessel')
    MicrogridStart
end


