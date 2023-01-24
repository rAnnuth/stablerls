function getports(path)
% delete and open datastroe
warning ('off','Simulink:BusElPorts:SigHierPropOutputDoesNotMatchInput');
warning ('off','Simulink:Bus:EditTimeBusPropFailureInputPort');


bus = {{1}};
path = 'untitled1';
top_ports = find_system(path, 'SearchDepth', 1, 'BlockType','Inport');
num = 1;
analyzed_port_numbers = [];
for i = 1 : length(top_ports)
    portnumber = str2double(get_param(top_ports{i},'Port'));
    if ismember(analyzed_port_numbers,portnumber)
        continue
    else
        analyzed_port_numbers(end+1) = portnumber;
    end
    %recursive iteration through signal path
    name = get_param(top_ports{i},'Element');
    if ~isempty(name)
        warning(['Unable to set DataType for "%s" due to Matlab limitations. ' ...
            'Please set manually to "%s" or use normal ' ...
            'inports at the top level'], top_ports{i}, ['Bus: bus' char(string(num + 1))])
            [bus, num] = create_input_bus(path, bus, num + 1, portnumber);
    else
        set_param([top_ports{i}],'OutDataTypeStr',['Bus: bus' char(string(num + 1))]);
        port_handle = get_param(top_ports{i},'PortHandles');
        port_line = get_param(port_handle.Outport,'Line');
        handle_dest_block = get_param(port_line,'DstBlockHandle');
        handle_dest_port = get_param(port_line,'DstportHandle');
        dst_type = get_param(handle_dest_block,'BlockType');
        if ~isequal(dst_type,'SubSystem')
            error('Please pack your system inside a SubSystem to generate the bus')
        end
        % save name of signal but with bus
        handle = get(handle_dest_port);
        subsystem_portnumber = handle.PortNumber;
        dst_block = get_param(handle_dest_block,'Name');

        [bus, num] = create_input_bus([path '/' dst_block], bus, num + 1, subsystem_portnumber);

    end

end

% write result to dict
DDName = 'BusSystem.sldd';
Simulink.data.dictionary.closeAll('-discard')
if isfile(DDName)
    delete('BusSystem.sldd')
end
DataDict = Simulink.data.dictionary.create(DDName);

DataObj = getSection(DataDict,'Design Data');

for i = 2 : length(bus)
    dataWriter(DataObj, {{getElems(bus{i}), bus{1}{i}{1}}})
end
saveChanges(DataDict);

end



function [bus, num] = create_input_bus(path, bus, num, portnumber)
%subsystems = find_system(path, 'SearchDepth', 1, 'BlockType','SubSystem');
subsystem_bus = {1};
top_ports = find_system(path, 'SearchDepth', 1, 'BlockType','Inport');

bus_name = ['bus' char(string(num))];


top_ports_out = top_ports;
for i3 = 1 : length(top_ports)
    port_conn = get_param(top_ports{i3}, 'PortConnectivity');
    % only consider inports store this as list and delete afterwards

    if (~isequal(get_param(top_ports{i3}, 'BlockType'), 'Inport') || ...
            ~isequal(str2double(get_param(top_ports{i3}, 'Port')), portnumber) || ...
            isempty(port_conn.DstBlock))
        top_ports_out(strcmp(top_ports_out, top_ports{i3})) = [];
        continue
    end

    % check if destination is subsystem
    port_handle = get_param(top_ports{i3},'PortHandles');
    port_line = get_param(port_handle.Outport,'Line');
    handle_dest_block = get_param(port_line,'DstBlockHandle');
    handle_dest_port = get_param(port_line,'DstportHandle');
    dst_type = get_param(handle_dest_block,'BlockType');
    if isequal(dst_type,'SubSystem')
        % save name of signal but with bus
        handle = get(handle_dest_port);
        subsystem_portnumber = handle.PortNumber;
        name = get_name(top_ports{i3});
        subsystem_bus{end+1} = {name ['Bus: bus' char(string(num+1))]};
        dst_block = get_param(handle_dest_block,'Name');
        if length(string(dst_block)) > 1
            dst_block = dst_block{1};
            warning('Found signal connected to multiple blocks at %s. Verify if all destination blocks take same input bussystem. Different inputs will cause errors!', [path '/' dst_block])
        end
        top_ports_out(strcmp(top_ports_out, top_ports{i3})) = [];
        [bus, num] = create_input_bus([path '/' dst_block], bus, num+1, subsystem_portnumber);
    end
end



if ~isempty(top_ports_out)
    for i4 = 1 : length(top_ports_out)
        name = get_name(top_ports_out{i4});
        subsystem_bus{end+1} = {name};
    end
end
bus{end+1} = subsystem_bus(2:end);
bus{1}{end+1} = {bus_name};
end

%%


function name = get_name(path)
name = get_param(path,'Element');
if isempty(name)
    name = strsplit(path, '/');
    name = name{end};
end
end
%
% dataWriter(DataObj,{{Example,'Example'},...
%                     {Control,'Control'},...
%                     })
% saveChanges(DataDict);

function dataWriter(DataObj,Data)

for i = 1 : length(Data)
    try
        iObj = getEntry(DataObj,Data{i}{2});
        deleteEntry(iObj)
    end
    addEntry(DataObj,Data{i}{2},Data{i}{1})
end
end

function [subsystem_portnumber, dst_block] = get_port_info(port)
port_handle = get_param(port,'PortHandles');
port_line = get_param(port_handle.Outport,'Line');
handle_dest_block = get_param(port_line,'DstBlockHandle');
handle_dest_port = get_param(port_line,'DstportHandle');
dst_block = get_param(handle_dest_block,'Name');
% check if type is block
handle = get(handle_dest_port);
subsystem_portnumber = handle.PortNumber;
end

function portnumber = get_port_number(port)
port_handle = get_param(port,'PortHandles');
handle = get(port_handle.Outport);
portnumber = handle.PortNumber;
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
