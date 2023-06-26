% Author Robert Annuth - robert.annuth@tuhh.de

function getports(model)
% getports Creats input port for simulinkmodels.
%   getports('my_model')
%   the resulting bus structure is stored in "BusSystem.sldd"
%   please add the datastore to the simulink model.
%   Matlab provides no functionality for building the input bus!
%   If something breaks restarting Matlab is (as always) the first step

% since the bus is unknown warning will occur > turning them off
warning ('off','Simulink:BusElPorts:SigHierPropOutputDoesNotMatchInput');
warning ('off','Simulink:Bus:EditTimeBusPropFailureInputPort');
warning ('off','Simulink:utility:slUtilityCompBusCannotUseSignalNameForBusName');
% bus will contain all bus elements found in the system
bus = {{1}};
% path will reflect the position of the iterative bus building process
path = model;

% reset all ports / this if off by default but might help users to debug
% basically this is required if bus elements already have bus elements
% specified on the lower level
if 0
    inports = find_system(path,'LookUnderMasks','on',...
        'FollowLinks','on', 'BlockType','Inport');
    outports = find_system(path,'LookUnderMasks','on',...
        'FollowLinks','on', 'BlockType','Outport');
    ports = cat(1, inports, outports);
    for i = 1 : length(ports)
        set_param(ports{i},'OutDataTypeStr','Inherit: auto');
    end
end

% find all inports
top_ports = find_system(path, 'SearchDepth', 1,'LookUnderMasks','on',...
    'FollowLinks','on', 'BlockType','Inport');

% variable to numerate busses
num = 1;
% keep track of analyzed busses since bus elements will have the same port
% number
analyzed_port_numbers = [];

% looping all ports found
for i = 1 : length(top_ports)
    portnumber = str2double(get_param(top_ports{i},'Port'));
    % only continue if port number is new
    if any(ismember(analyzed_port_numbers,portnumber))
        continue
    else
        analyzed_port_numbers(end+1) = portnumber;
    end

    % we can skip the port if its not connected to a subsystem
    % we have to follow the line to the destination
    port_handle = get_param(top_ports{i},'PortHandles');
    port_line = get_param(port_handle.Outport,'Line');
    handle_dest_block = get_param(port_line,'DstBlockHandle');
    dst_type = get_param(handle_dest_block,'BlockType');
    if length(string(dst_type)) > 1
        dst_type = dst_type{1};
    end
    if ~isequal(dst_type, 'SubSystem')
        continue
    end
    % recursive iteration through signal path
    [bus, num] = create_input_bus(path, bus, num + 1, portnumber);
end

% we need to skip some bus connections since bus elments and
% input ports behave differently
bus = skip_bus_connections(bus, path);

% write result to datastore
DDName = 'BusSystem.sldd';
Simulink.data.dictionary.closeAll('-discard')
% remove existing datastore to avoid conflicts
if isfile(DDName)
    delete('BusSystem.sldd')
end

% create, open and write datastore
DataDict = Simulink.data.dictionary.create(DDName);
DataObj = getSection(DataDict,'Design Data');
for i = 2 : length(bus)
    dataWriter(DataObj, {{getElems(bus{i}), bus{1}{i}{1}}})
end
saveChanges(DataDict);

%%
% now we can continue with the ouptut bus
% let matlab calculate the output bus
top_ports = find_system(model, 'SearchDepth', 1,'LookUnderMasks','on',...
    'FollowLinks','on', 'BlockType','Outport');

% reset bus types to allow matlab the compilation
for i = 1 : length(top_ports)
    set_param(top_ports{i},'OutDataTypeStr','Inherit: auto');
end

% create bus for each port they are automatically stored in the data
% dictionary
for i = 1 : length(top_ports)
    busInfo = Simulink.Bus.createObject(model, top_ports{i});
    % set datatype to outport but only if this is a bus
    if ~isempty(busInfo.busName)
        set_param(top_ports{i},'OutDataTypeStr',['Bus: ' busInfo.busName]);
    end
end

% enabling warnings again
warning ('on','Simulink:BusElPorts:SigHierPropOutputDoesNotMatchInput');
warning ('on','Simulink:Bus:EditTimeBusPropFailureInputPort');
warning ('on','Simulink:utility:slUtilityCompBusCannotUseSignalNameForBusName');

end

function [bus, num] = create_input_bus(path, bus, num, portnumber)
% this function is called recursive while iterating through the bus
%subsystems = find_system(path, 'SearchDepth', 1, 'BlockType','SubSystem');

% variant subsystems behave different and we need to modify the path
% to search inside the correct subsystem
if (isfield(get_param(path, 'ObjectParameters'), 'Variant') ...
        && isequal(get_param(path, 'Variant'), 'on'))
    path = get_param(path,'CompiledActiveChoiceBlock');
end

% if we are at root level we have to change signal types later
is_bdroot = ~any(ismember(path,'/'));

% variable to store the bus at current path in model
subsystem_bus = {1};
top_ports = find_system(path, 'SearchDepth', 1,'LookUnderMasks','on',...
    'FollowLinks','on', 'BlockType','Inport');

% create name since num will change during execution
bus_name = ['bus' char(string(num))];

% variable to keep track of ports considered
top_ports_out = top_ports;
% flag to purge bus later
skip = false;

% iterate all input ports at path
for i3 = 1 : length(top_ports)
    % continue if wrong blocktype, port has wrong portnumber
    % or port is not connected
    port_conn = get_param(top_ports{i3}, 'PortConnectivity');
    if (~isequal(get_param(top_ports{i3}, 'BlockType'), 'Inport') || ...
            ~isequal(str2double(get_param(top_ports{i3}, 'Port')), portnumber) || ...
            isempty(port_conn.DstBlock))
        top_ports_out(strcmp(top_ports_out, top_ports{i3})) = [];
        continue
    end

    % check if destination is subsystem
    % we have to follow the line to the destination
    port_handle = get_param(top_ports{i3},'PortHandles');
    port_line = get_param(port_handle.Outport,'Line');
    handle_dest_block = get_param(port_line,'DstBlockHandle');
    handle_dest_port = get_param(port_line,'DstportHandle');
    dst_type = get_param(handle_dest_block,'BlockType');
    if length(string(dst_type)) > 1
        dst_type = dst_type{1};
    end

    % if destination is a goto block we follow the connection
    if isequal(dst_type,'Goto')
        dst_block = get_param(handle_dest_block,'Name');
        from_blocks = find_system(path, 'SearchDepth', 1,'LookUnderMasks','on',...
            'FollowLinks','on', 'BlockType','From');
        label = get_param([path '/' dst_block],'GotoTag');
        index = cellfun(@(x) isequal(get_param(x,'GotoTag'), label), from_blocks);
        from_blocks = from_blocks(index);
        if length(from_blocks) > 1
            warning('Found signal connected to multiple blocks at %s. ', ...
                'Verify if all destination blocks take same input bussystem. ', ...
                'Different inputs will cause errors!', [path '/' dst_block])
            from_blocks = from_blocks{1};
        end
        port_handle = get_param(from_blocks,'PortHandles');
        port_line = get_param(port_handle{1}.Outport,'Line');
        handle_dest_block = get_param(port_line,'DstBlockHandle');
        handle_dest_port = get_param(port_line,'DstportHandle');
        dst_type = get_param(handle_dest_block,'BlockType');
        if length(string(dst_type)) > 1
            dst_type = dst_type{1};
        end
    end


    % if destination is a subsystem
    if isequal(dst_type,'SubSystem')

        % rename signal type if we are at root level
        if is_bdroot
            % set corret bus type
            set_param(top_ports{i3},'OutDataTypeStr',['Bus: bus' char(string(num+1))]);
        end

        % get name of buselement and port numbers
        handle = get(handle_dest_port);
        subsystem_portnumber = handle.PortNumber;
        [name, bus_port]= get_name(top_ports{i3});

        % skip connection if we have no bus elment port
        if ~bus_port
            skip = true;
        end

        % store information about the bus in cell array
        subsystem_bus{end+1} = {name ['Bus: bus' char(string(num+1))]};

        % determine destination block to modify path
        dst_block = get_param(handle_dest_block,'Name');
        % warning if we connect the signal to multiple systems
        if length(string(dst_block)) > 1
            dst_block = dst_block{1};
            warning('Found signal connected to multiple blocks at %s. ', ...
                'Verify if all destination blocks take same input bussystem. ', ...
                'Different inputs will cause errors!', [path '/' dst_block])
        end
        % remove the bus from port list
        top_ports_out(strcmp(top_ports_out, top_ports{i3})) = [];

        % build the bus for connected subsystem
        [bus, num] = create_input_bus([path '/' dst_block], bus, ...
            num+1, subsystem_portnumber);

        % check if connected block is BusSelector
        % bus selectors are only allowed on the lowest level
        % and should be replaced with bus elements!
    elseif isequal(dst_type,'BusSelector')
        % get information about port and bus selector signal
        [name, bus_port]= get_name(top_ports{i3});
        dst_block = get_param(handle_dest_block,'Name');
        signal = get_param([path '/' dst_block], 'OutputSignalNames');
        signal = {signal{1}(2:end-1)};

        % handle input ports different and create bus elements
        if ~bus_port
            subsystem_bus{end+1} = signal;
        else
            subsystem_bus{end+1} = {name ['Bus: bus' char(string(num+1))]};
            bus{1}{end+1} = {['Bus: bus' char(string(num+1))]};
            bus{end+1} = {signal};
        end

        % remove the bus from port list
        top_ports_out(strcmp(top_ports_out, top_ports{i3})) = [];
        num = num + 1;
    end
end

% all ports remaining in top_ports_out are signals directly
% connnected to blocks
if ~isempty(top_ports_out)
    % iterathe through signals and create bus elements for each
    for i4 = 1 : length(top_ports_out)
        name = get_name(top_ports_out{i4});
        subsystem_bus{end+1} = {name};
    end
end

% finally store the bus of the subsystem in the bus variable
bus{end+1} = subsystem_bus(2:end);

% handle the skip flag if set
if skip
    bus{1}{end+1} = {bus_name, skip};
else
    bus{1}{end+1} = {bus_name};
end
end

%%
function [name, bus_port] = get_name(xpath)
% get_name returns the name of a input port and
% a boolean if its a bus element
name = get_param(xpath,'Element');

% if name is empty its a input port
if isempty(name)
    name = strsplit(xpath, '/');
    name = name{end};
    bus_port = false;
else
    bus_port = true;
end
name = erase(name, ' ');
if any(ismember(name, ' .'))

    error(strjoin([sprintf("Port at %s contains spaces or '.' in name or signal name.\n ", xpath) ...
        "Please rename the port! " ...
        sprintf("Use hilite_system(%s) to find the block", xpath)]))
end
end

function dataWriter(DataObj,Data)
% dataWrite helper function to write data to data store
for i = 1 : length(Data)
    % make sure element is not already in store
    try
        iObj = getEntry(DataObj,Data{i}{2});
        deleteEntry(iObj)
    end
    addEntry(DataObj,Data{i}{2},Data{i}{1})
end
end

function out = getElems(Name)
% getElems convert bus cell array to bus elements

% flip to order bus correctly
Name=fliplr(Name);
counter = 1;
for i = length(Name) : -1 : 1
    % its also possible to specify the bus by hand and use the syntax
    % bus[1:20] to create bus1, bus2... bus20 but this is not used here
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
        % get the elements contained in Name
    else
        elems(counter) = Simulink.BusElement;
        elems(counter).Name = Name{i}{1};
        if isequal(size(Name{i}),[1 2])
            elems(counter).DataType = Name{i}{2};
        end
        counter = counter + 1;
    end
end
% convert the bus elements to a bus
out = Simulink.Bus;
out.Elements = elems;
end

function bus = skip_bus_connections(bus, path)
% input ports behave differently than bus elements. Input ports only pass on
% the bus while bus elements access/ require a own bus element. Since the two
% port types are often mixed in models they have to be handled individually
% skip_bus_connections skips one bus connection if a bus element is connected
%       to a input port inside a subsystem

% if the length is large 1 the skip flag is set
len = cellfun(@(x) length(x), bus{1});
index = (len > 1);

% get the ports of highest level since we dont want to ignore them
top_ports = find_system(path, 'SearchDepth', 1, 'BlockType','Inport');

for i = 2 : length(bus)
    % only contine if skip flag is set
    if (~index(i) || ~iscell(bus{i}))
        continue
    end

    % determine which connections should be replaced
    % busx > source > target (before)
    % busx > target          (after)
    source = ['Bus: ' bus{1}{i}{1}];
    target = bus{i}{1}{2};

    for i2 = 2 : length(bus)
        for i3 = 1 : length(bus{i2})
            % if the length is smaller 2 its no bus
            if length(bus{i2}{i3}) < 2
                continue
            end
            % if we find a bus pointing to source (busx above)
            if isequal(bus{i2}{i3}{2}, source)
                % replace source with target
                bus{i2}{i3}{2} = target;

            end
        end
    end
    % if a top bus points to the source element we have to rename the datatype
    for i4 = 1 : length(top_ports)
        if isequal(get_param(top_ports{i4}, 'OutDataTypeStr'), source)

            % set datatype
            set_param(top_ports{i4},'OutDataTypeStr',target);

        end
    end
end

% remove skipped busses from bus array
bus(index) = [];
bus{1}(index) = [];

% remove duplicate elements
bus_len = length(bus);
i = 2;
while i < bus_len
    index = cellfun(@(x) isequal(x,bus{i}), bus);
    index(i) = 0;
    % if true we have identical busses
    if any(index)
        % remove dublicates
        bus(index) = [];
        % resolve missing links
        removed_busses = bus{1}(index);
        % get a list of all removed busses
        removed_busses = cellfun(@(x) {{['Bus: ' x{1}]}}, removed_busses);
        replacement_bus = bus{1}(i);
        replacement_bus = {{['Bus: ' replacement_bus{1}{1}]}};
        bus{1}(index) = [];
        for i2 = 2 : length(bus)
            for i3 = 1 : length(bus{i2})
                % if the length is smaller 2 its no bus
                if length(bus{i2}{i3}) < 2
                    continue
                end
                % check if this bus accesses a removed bus and replace it

                if any(cellfun(@(x) isequal(x,bus{i2}{i3}(2)), removed_busses))
                    % replace source
                    bus{i2}{i3}{2} = replacement_bus{1}{1};
                end
            end
        end
        % finally check if top port accesses removed bus
        for i2 = 1 : length(top_ports)
            port_datatype = {get_param(top_ports{i2}, 'OutDataTypeStr')};
            if any(cellfun(@(x) isequal(x, port_datatype), removed_busses))
                % change datatype
                set_param(top_ports{i2},'OutDataTypeStr', replacement_bus{1}{1});

            end
        end
    end
    % increment counter
    i = i + 1;
    bus_len = length(bus);
end
end
