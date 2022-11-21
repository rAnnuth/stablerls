import matlab.engine
import sys, os, io

Section = 'FMU'

def createFMU(cfg):
    # start engine
    eng = matlab.engine.start_matlab('-nosplash -noFigureWindows -r')						
    slxDir = os.path.dirname(cfg[Section]['fmuPath'])
    mdl = os.path.splitext(os.path.basename(cfg[Section]['fmuPath']))[0]
    
    # based on makeFMU.m script
    eng.eval(f"cd('{slxDir}')", nargout=0)
    #eng.eval("addpath(pwd)", nargout=0)
    eng.eval(f"mdl = '{mdl}';" ,nargout=0)
    #eng.eval("gitCommit",nargout=0)
    eng.eval(f"Ts = {cfg[Section]['dt']};" ,nargout=0)
    eng.eval("open_system(mdl)",nargout=0)
    eng.eval("makeBus",nargout=0)
    eng.eval("set_param(mdl,'SolverType','Fixed-step')",nargout=0)
    eng.eval("set_param(mdl,'FixedStep',string(Ts))",nargout=0)

    # Get Measurement Bus
    eng.eval("set_param([mdl '/Measurement'],'OutDataTypeStr','Inherit: auto');",nargout=0)
    eng.eval("busInfo = Simulink.Bus.createObject(mdl ,[mdl '/Measurement']);",nargout=0)
    eng.eval("set_param([mdl '/Measurement'], 'OutDataTypeStr',['Bus: ' busInfo.busName]);",nargout=0)

    eng.eval("exportToFMU2CS(mdl, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');",nargout=0)
    eng.eval("Simulink.data.dictionary.closeAll('-discard')",nargout=0)
    eng.quit()
