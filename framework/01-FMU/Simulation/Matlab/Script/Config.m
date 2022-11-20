%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Config file for MainBus%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Path setup
addpath(strjoin({genpath('Submodel'), ...
         ['Grid' filesep() SystemName],...
         ['Grid' filesep() SystemName filesep() 'parameters']},...
         ';'),path)
     
% open simulation   
open_system(SimulationName)
