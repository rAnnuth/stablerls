function [randN] = getProfile(inpt, SimTime, seed, dist)
% unpack
Power = inpt{1};
timeConst = inpt{2};
switchTime = inpt{3};
edges = inpt{4};

hourTime = datetime(2022,1,1,0,0,SimTime);

% set seed
rng(seed)
randN = zeros(size(SimTime));

% normalize to given range
idx2 = 1;
% fermi dirac
for idx = {[0,6],[6,9],[9,12],[12,15],[15,18],[18,21],[21,24]}
    iSel =  (hourTime.Hour >= idx{1}(1)) & (hourTime.Hour < idx{1}(2));
    randN(iSel) = (1-edge(1)) .* rand(sum(double(iSel),'all'),1)' + edge(1);
    fkt = @(x,p) 2/(exp(2*dist{idx2}*x-1)+1)-0.5;
    randN(iSel) = arrayfun(fkt,randN(iSel));
    idx2 = idx2 + 1;
end

% binsort
for i = [1:numel(edges(1:end-1))]
    if i == 1
        iNums = (randN<=edges(i+1));
    elseif i == numel(edges)-1
        iNums = (randN>=edges(i)) & randN<=edges(i+1);
    else
        iNums = (randN>=edges(i));
    end
    
    avgEdge = (edges(i) + edges(i+1)) / 2;
    idiff = randN - avgEdge;
    idiff(~iNums) = nan;
    inegDiff = idiff < 0;
    iposDiff = idiff >= 0;
    
    randN(inegDiff) = edges(i);
    randN(iposDiff) = edges(i+1);
end
randN(randN > edges(end)) = edges(end);
randN(randN<0) = 0;

% stretch as specified with timeConst
for idx = [1:floor(SimTime(end)/(timeConst))]
   randN([idx+1+(idx-1)*(timeConst-1):idx+idx*(timeConst-1)]) = randN(idx+(idx-1)*(timeConst-1));
end

if ~mod(SimTime(end),timeConst) == 0
    randN([1+idx+idx*(timeConst-1):end]) = randN(1+idx+idx*(timeConst-1));
end

% scale result
randN = randN * Power;

end



