# Author Robert Annuth - robert.annuth@tuhh.de
import matlab.engine
import os

# specify section containing FMU path
# ----------------------------------------------------------------------------
section_names = 'FMU'
# ----------------------------------------------------------------------------


def createFMU(cfg):
    """
    see https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    for the installation of matlab engine

    Parameters:
        dict: dictionary containing the keys 'fmuPath', 'dt' within the section specified above
    """
    # get current folder
    script_folder = os.path.dirname(os.path.abspath(__file__))

    # start matlab engine
    eng = matlab.engine.start_matlab('-nosplash -noFigureWindows -r')
    slxDir = os.path.dirname(cfg.get(section_names)['fmuPath'])
    mdl = os.path.splitext(os.path.basename(cfg.get(section_names)['fmuPath']))[0]

    # the code is also available as matlab code /Example/matlab/template/makeFMU.m
    eng.eval(f"cd('{slxDir}')", nargout=0)
    eng.eval(f"mdl = '{mdl}';", nargout=0)
    print(script_folder)
    
    eng.eval(f"addpath('{script_folder}')", nargout=0)

    # dt specifies the step time of the FMU
    eng.eval(f"Ts = {cfg.get(section_names)['dt']};", nargout=0)
    # open the system and create bus structure
    eng.eval("open_system(mdl)", nargout=0)
    eng.eval("getports(mdl)", nargout=0)
    eng.eval("set_param(mdl,'DataDictionary','BusSystem.sldd');", nargout=0)

    # set the solver configuration
    eng.eval("set_param(mdl,'SolverType','Fixed-step')", nargout=0)
    eng.eval("set_param(mdl,'FixedStep',string(Ts))", nargout=0)

    # let matlab create the output bus
    eng.eval(
        "set_param([mdl '/Measurement'],'OutDataTypeStr','Inherit: auto');", nargout=0)
    eng.eval(
        "busInfo = Simulink.Bus.createObject(mdl ,[mdl '/Measurement']);", nargout=0)
    eng.eval(
        "set_param([mdl '/Measurement'], 'OutDataTypeStr',['Bus: ' busInfo.busName]);", nargout=0)

    eng.eval(
        "exportToFMU2CS(mdl, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');", nargout=0)
    eng.eval("Simulink.data.dictionary.closeAll('-discard')", nargout=0)
    eng.quit()
