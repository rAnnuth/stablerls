# Author Robert Annuth - robert.annuth@tuhh.de
import matlab.engine
import os

# specify section containing FMU path
# ----------------------------------------------------------------------------
section_names = "FMU"
# ----------------------------------------------------------------------------


def createFMU(cfg, simulink_model, remove_datastore=True):
    """See https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    for the installation of MATLAB engine. The engine is required to run this function.

    The function searches for a simulink model defined in the config dict and
    compiles it into an FMU. The target filename is specified within the config

    Parameters:
    -----------
    cfg : dict
        Dictionary containing the keys 'FMU_path', 'dt' within the section specified above
        (default is 'FMU')
    simulink_model : string
        path to the Simulink model that should be compiled to a FMU
    remove_datastore : bool
        MATLAB has to create a datastore file during compilation. We dont need it afterward and 
        delete it by default. Unless specified otherwise
    """
    # get current folder because we need the 'getports.m' function later
    script_folder = os.path.dirname(os.path.abspath(__file__))

    # start matlab engine
    eng = matlab.engine.start_matlab("-nosplash -noFigureWindows -r")
    slx_dir = os.path.dirname(simulink_model)
    slx_model = os.path.splitext(os.path.basename(simulink_model))[0]
    target_fmu = cfg.get(section_names)["FMU_path"]

    if not slx_dir == '':
        eng.eval(f"cd('{slx_dir}')", nargout=0)
    eng.eval(f"mdl = '{slx_model}';", nargout=0)

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
    
    # move FMU to the desired location
    try:
        os.rename(os.path.join(slx_dir,slx_model) + '.fmu', target_fmu)
    except FileExistsError:
        os.remove(target_fmu)
        os.rename(os.path.join(slx_dir,slx_model) + '.fmu', target_fmu)

    # remove MATLAB datastore 
    if remove_datastore:
        os.remove(os.path.join(slx_dir, 'BusSystem.sldd'))
