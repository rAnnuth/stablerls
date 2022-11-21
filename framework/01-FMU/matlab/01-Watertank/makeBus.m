% function building required bus
% Created by Robert Annuth

function makeBus 
%open data file
DDName = 'BusSystem.sldd';
Simulink.data.dictionary.closeAll
if isfile(DDName)
    DataDict = Simulink.data.dictionary.open(DDName);
else
    DataDict = Simulink.data.dictionary.create(DDName);
end
DataObj = getSection(DataDict,'Design Data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Top Level %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                     

Control = getElems({{'ref'},...
                    {'uctrl'},...
                    });


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataWriter(DataObj,{
                    {Control,'Control'},...
                    })
saveChanges(DataDict);
DataDict.close();
end

function dataWriter(DataObj,Data)

for i = 1 : length(Data)
    try
        iObj = getEntry(DataObj,Data{i}{2});
        deleteEntry(iObj)
    end
    addEntry(DataObj,Data{i}{2},Data{i}{1})
end
end

function out = getElems(Name)
Name=fliplr(Name);
counter = 1;
for i = length(Name) : -1 : 1
    if ~isempty(strfind(Name{i}{1},'['))
        lim = regexp(Name{i}{1},'\d+','match');
        bname = regexp(Name{i}{1},'[a-zA-Z]+','match');
        for i2 = [str2double(lim{1}):str2double(lim{2})]
            xName = [bname{1} num2str(i2)];
            elems(counter) = Simulink.BusElement;
            elems(counter).Name = xName;
            if isequal(size(Name{i}),[1 2])
                elems(counter).DataType = Name{i}{2};
            end       
            counter = counter + 1;
        end    
    else
        elems(counter) = Simulink.BusElement;
        elems(counter).Name = Name{i}{1};
        if isequal(size(Name{i}),[1 2])
            elems(counter).DataType = Name{i}{2};
        end
        counter = counter + 1;        
    end
end
out = Simulink.Bus;
out.Elements = elems;
end
