function custPlot(data,varargin)

defaultTitle = '';
valTitle = @(x) isstring(x) || ischar(x);

p = inputParser;
addOptional(p,'title',defaultTitle,valTitle)
parse(p,varargin{:});

% set large values to nan. Simulation error?
figure

data.Data(data.Data>1e6) = nan;
plot(data.Data)
title(p.Results.title)

end

