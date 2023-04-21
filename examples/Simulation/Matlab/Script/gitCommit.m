function gitCommit(ignoreFolder,mdl)
[~,repoStatModAndUntrackedFiles] = system('git status --branch --porcelain');
ignoreFolder = strrep(ignoreFolder,'/','\/');
regex = strcat('([M?]+ (?!',strjoin(ignoreFolder,'|'),')[a-zA-Z\/0-9]+)');
noCommit = regexp(repoStatModAndUntrackedFiles,regex, 'match');

if ~isempty(noCommit)
    disp('Not commited files:')
    disp(noCommit{:})
    error('No clean working directory. Please commit changes!')
end

[~,commitHash] = system('git rev-parse HEAD');
[~,repo] = system('git rev-parse --show-toplevel');
repo = regexp(repo,'\w+', 'match');
repo = repo{end};

fid= fopen('FMUInfo.txt','w');
fprintf(fid, 'Repository: %s\nCommit: %sFMU Name: %s', repo, commitHash, mdl);
fclose(fid);
end