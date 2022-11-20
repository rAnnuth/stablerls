function [outTime,profile] = cleanProfile(SimTime,randN,switchTime)
%clean results
iRemove = logical([0 diff(randN)==0]);
timeBackup = SimTime;
SimTime(iRemove) = [];
randN(iRemove) = [];

%convert to rectangular profile
profile = zeros(2*length(randN)-1,1);
outTime = zeros(length(profile),1);

profile(1:2:end) = randN;
outTime(1:2:end) = SimTime;

if length(profile) == 1
%if no switching occurs    
    profile(end+1,1) = profile(end);
    outTime(end+1,1) = timeBackup(end);
else
%if switching occurs add data points to create rectangular profiles
    profile(2:2:end-1) = randN(1:end-1);
    outTime(2:2:end-1) = SimTime(2:end)- switchTime;
    %remove last switching
    profile(end) = [];
    outTime(end-1) = [];    
end

outTime = datetime(2020,1,1) + seconds(outTime);
plot(outTime,profile)
end