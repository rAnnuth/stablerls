fp = strsplit(matlab.desktop.editor.getActiveFilename,filesep());
SimTime = [0:1:60*60*24*366];

for seed = [1:36]
    for seed2 = [1:2]
        TallBoyLamp=3.6; WallLamp=7; BedLamp=2*7; SpotLight=6*10.2;
        CabinLight = {TallBoyLamp + WallLamp + BedLamp + SpotLight, 60*30, 1e-3, [0, 0.5, 0.75, 1]};
        dist = {3,...     % 0-6h
            2,...     % 6-9h
            1,...     % 9-12h
            2.5,...   % 12-15h
            1,...     % 15-18h
            0.9,...   % 18-21h
            1};       % 21-24h
        [xProfile] = getProfile(CabinLight,SimTime,seed,dist);
        TotalPower = xProfile;
        
        % TV
        CabinTV = {77, 60*60, 1e-3, [0, 1]};
        dist = {3,... % 0-6h
            1,...     % 6-9h
            1,...     % 9-12h
            2,...     % 12-15h
            2,...     % 15-18h
            0.9,...   % 18-21h
            0.8};     % 21-24h
        [xProfile] = getProfile(CabinTV,SimTime,seed,dist);
        TotalPower = TotalPower + xProfile;
        
        % Hair Dryer
        CabinHD = {1200, 60*5, 1e-3, [0, 0.75, 0.8, 0.9, 1]};
        dist = {300,...     % 0-6h
            300,...     % 6-9h
            150,...     % 9-12h
            300,...     % 12-15h
            300,...     % 15-18h
            150,...     % 18-21h
            300};       % 21-24h
        [xProfile] = getProfile(CabinHD,SimTime,seed,dist);
        TotalPower = TotalPower + xProfile;
        
        % Refridgerator
        CabinRefridgerator = {65, 60*3, 1e-3, [0, 1]};
        dist = {1,...     % 0-6h
            1,...     % 6-9h
            1,...     % 9-12h
            1,...     % 12-15h
            1,...     % 15-18h
            1,...     % 18-21h
            1};       % 21-24h
        [xProfile] = getProfile(CabinRefridgerator,SimTime,seed,dist);
        TotalPower = TotalPower + xProfile;
        
        % Mirror
        CabinMirror = {15+15, 60*30, 1e-3, [0, 0.5, 1]};
        dist = {15,...       % 0-6h
            12.5,...     % 6-9h
            12.5,...     % 9-12h
            15,...       % 12-15h
            12.5,...     % 15-18h
            12.5,...     % 18-21h
            12.5};       % 21-24h
        [xProfile] = getProfile(CabinMirror,SimTime,seed,dist);
        TotalPower = TotalPower + xProfile;
        
        [xTime,xProfile] = cleanProfile(SimTime,TotalPower,1e-3);
        plot(xTime,xProfile)
        CabinConsumer = timetable(xTime, xProfile ,'VariableNames',{'Data'});
        
        %Emergency light? speaker?
        CabinEmergency = {25, 60*60*2, 1e-3, [1]};
        dist = {1,...       % 0-6h
            1,...     % 6-9h
            1,...     % 9-12h
            1,...       % 12-15h
            1,...     % 15-18h
            1,...     % 18-21h
            1};       % 21-24h
        [xProfile] = getProfile(CabinEmergency,SimTime,seed,dist);
        [xTime,xProfile] = cleanProfile(SimTime,xProfile,1e-3);
        CabinEmergency = timetable(xTime, xProfile ,'VariableNames',{'Data'});
       
        save(fullfile(fp{1:end-1},'parameters', [num2str(seed) '-' num2str(seed2) '-' num2str(SimTime(end)) '-' 'Consumer' '.mat']),'CabinConsumer')  
        save(fullfile(fp{1:end-1},'parameters', [num2str(seed) '-' num2str(seed2) '-' num2str(SimTime(end)) '-' 'Emergency' '.mat']),'CabinEmergency')  
    end
        
        % HVAC 4sec an und T 20sec * max 85%
        CabinHVAC = {2 * 1.15e3, 60*60*2, 1e-3, [0, 0.5, 1]};
        dist = {15,...       % 0-6h
            12.5,...     % 6-9h
            12.5,...     % 9-12h
            15,...       % 12-15h
            12.5,...     % 15-18h
            12.5,...     % 18-21h
            12.5};       % 21-24h
        [xProfile] = getProfile(CabinHVAC,SimTime,seed,dist);
        [xTime,xProfile] = cleanProfile(SimTime,xProfile,1e-3);
        CabinHVAC = timetable(xTime, xProfile ,'VariableNames',{'Data'});    
        save(fullfile(fp{1:end-1},'parameters', [num2str(seed) '-' num2str(SimTime(end)) '-' 'HVAC' '.mat']),'CabinHVAC')  
end

%% AC[1:3] Socket[1:6] Laundrette Pantry
% 
% % HVAC 4sec an und T 20sec * max 85%
% CabinHVAC = {2 * 1.15e3, 60*60*2, 1e-3, [0, 0.5, 1]};
% dist = {15,...       % 0-6h
%     12.5,...     % 6-9h
%     12.5,...     % 9-12h
%     15,...       % 12-15h
%     12.5,...     % 15-18h
%     12.5,...     % 18-21h
%     12.5};       % 21-24h
% [xProfile] = getProfile(CabinHVAC,SimTime,seed,dist);
% [xTime,xProfile] = cleanProfile(SimTime,xProfile,1e-3);
% CabinHVAC = timetable(xTime, xProfile ,'VariableNames',{'Data'});    
% inports_dataset{1}.(['Cabin' num2str(seed)]).HVAC.LoadSetpoint.Data = CabinEmergency;
% 
