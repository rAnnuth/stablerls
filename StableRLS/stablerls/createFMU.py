# Author Robert Annuth - robert.annuth@tuhh.de
import matlab.engine
import os

# specify section containing FMU path
# ----------------------------------------------------------------------------
section_names = "FMU"
# ----------------------------------------------------------------------------


def createFMU(cfg, simulink_model):
    """See https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    for the installation of matlab engine. The engine is required to run this function.

    The function searches for a simulink model defined in the config dict and
    compiles it into an FMU. The target filename is specified within the config

    Parameters:
    ------
    cfg: dict
        Dictionary containing the keys 'fmuPath', 'dt' within the section specified above
        (default is 'FMU')
    simulink_model: string
        path to the simulink model that should be compiled to a FMU
    """
    # get current folder because we need the 'getports.m' function later
    script_folder = os.path.dirname(os.path.abspath(__file__))

    # start matlab engine
    eng = matlab.engine.start_matlab("-nosplash -noFigureWindows -r")
    slx_dir = os.path.dirname(simulink_model)
    slx_model = os.path.splitext(os.path.basename(simulink_model))[0]
    target_fmu = cfg.get(section_names)["fmuPath"]

    # the code is also available as matlab code /Example/matlab/template/makeFMU.m
    if not slx_dir == '':
        eng.eval(f"cd('{slx_dir}')", nargout=0)
    eng.eval(f"mdl = '{slx_model}';", nargout=0)
    print(script_folder)

    eng.eval(f"addpath('{script_folder}')", nargout=0)

    # dt specifies the step time of the FMU
    eng.eval(f"Ts = {cfg.get(section_names)['dt']};", nargout=0)
    # open the system and create bus structure
    eng.eval("open_system(mdl)", nargout=0)
    eng.eval("set_param(mdl,'DataDictionary','BusSystem.sldd');", nargout=0)
    eng.eval("getports(mdl)", nargout=0)

    # set the solver configuration
    eng.eval("set_param(mdl,'SolverType','Fixed-step')", nargout=0)
    eng.eval("set_param(mdl,'FixedStep',string(Ts))", nargout=0)

    # start the fmu creation process and quit
    eng.eval(
        "exportToFMU2CS(mdl, 'CreateModelAfterGeneratingFMU', 'off', 'AddIcon', 'off');",
        nargout=0,
    )
    eng.eval("Simulink.data.dictionary.closeAll('-discard')", nargout=0)
    eng.quit()

    os.rename(slx_model + '.fmu', target_fmu)
