# Author Robert Annuth - robert.annuth@tuhh.de
import matlab.engine
import os

# specify section containing FMU path
Section = 'FMU'


def createFMU(cfg):
    """
    see https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    for the installation of matlab engine

    Parameters:
        dict: dictionary containing the keys 'fmuPath', 'dt' within the section specified above
    """

    # start matlab engine
    eng = matlab.engine.start_matlab('-nosplash -noFigureWindows -r')
    slxDir = os.path.dirname(cfg[Section]['fmuPath'])
    mdl = os.path.splitext(os.path.basename(cfg[Section]['fmuPath']))[0]

    # the code is also available as matlab code /Example/matlab/template/makeFMU.m
    eng.eval(f"cd('{slxDir}')", nargout=0)
    eng.eval(f"mdl = '{mdl}';", nargout=0)
    # dt specifies the step time of the FMU
    eng.eval(f"Ts = {cfg[Section]['dt']};", nargout=0)
    eng.eval("open_system(mdl)", nargout=0)
    eng.eval("makeBus", nargout=0)
    eng.eval("set_param(mdl,'SolverType','Fixed-step')", nargout=0)
    eng.eval("set_param(mdl,'FixedStep',string(Ts))", nargout=0)

    # Get Measurement Bus
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
