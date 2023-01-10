function loadParamsInCallerWorkspace(filename,reqvarnames,asSimuParam)
tab = readParamFile(filename);

if exist('reqvarnames','var') && ~isempty(reqvarnames)
    tab = tab(reqvarnames,:);  
end

if ~exist('asSimuParam','var')
    asSimuParam = false;
end

for i=1:numel(tab(:,1))
    if asSimuParam
        val = Simulink.Parameter();    
        val.CoderInfo.StorageClass = 'Model default';
        val.CoderInfo.Alias = tab.Properties.RowNames{i};
        val.Value = tab.value{i};
    else
        val = tab.value{i};
    end
    assignin('caller', tab.Properties.RowNames{i} ,val)
end