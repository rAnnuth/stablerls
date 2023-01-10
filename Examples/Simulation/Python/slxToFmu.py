import matlab.engine
import sys, os, io

def exportFmu(slxName,dt):
    # start engine
    eng = matlab.engine.start_matlab()						
    slxDir = os.path.dirname(slxName)
    mdl = os.path.splitext(os.path.basename(slxName))[0]
    
    # based on makeFMU.m script
    eng.eval("cd('{}')".format(slxDir),nargout=0)
    eng.eval("addpath(pwd)",nargout=0)
    eng.eval("mainDir = fullfile('..',filesep(),'..',filesep());",nargout=0)
    eng.eval("addpath(genpath(fullfile(mainDir,'Script')))",nargout=0)
    eng.eval("addpath(genpath(fullfile(mainDir,'Submodel')))",nargout=0)
    eng.eval("mdl = '{}';".format(mdl),nargout=0)
    #eng.eval("gitCommit",nargout=0)
    eng.eval("Ts = {};".format(dt),nargout=0)
    eng.eval("open_system(mdl)",nargout=0)
    eng.eval("FMUBus",nargout=0)
    eng.eval("set_param(mdl,'SolverType','Fixed-step')",nargout=0)
    eng.eval("set_param(mdl,'FixedStep',string(Ts))",nargout=0)
    eng.eval("exportToFMU2CS(mdl, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');",nargout=0)
    eng.quit()

if __name__ == '__main__':
    exportFmu(sys.argv[1],sys.argv[2])
