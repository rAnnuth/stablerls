![](src/icon.png)

<h2 align="center">Stable Reinforcement Learning for Simulink</h2>

<p align="center">
<a href="https://stablerls.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/stablerls/badge/?version=latest"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


StableRLS is a software package to use your existing MATLAB Simulink models in Python for reinforcement learning. Basically, your simulation is wrapped in a [Gymnasium](https://gymnasium.farama.org/) environment. The package provides the following features:
- automatic input output signal generation for your model
- automatic compilation of the simulink model to a Functional-Mockup-Unit (FMU) to enable fast simulation
- flexible implementation of postprocessing
- easy to read code

**And the best is** the only thing you need to do is:
- defining a reward function to train your agend

## ToDo
- [ ] Add Action Status, Coverage Status
- [ ] Write paper

## General Information
Reinforcement Learning (RL) is a fast changing and innovative field. The main purpose of this package is to bring the easy to use MATLAB Simulink modelling interface together with the flexible and state of the are Gymnasium interface. So the RL algorithm and learning interface is out of scope of this package. However, we will try to make the interface between the Matlab and Python as easy as possible.

## Installation
The package is currently tested whith Python 3.9.
%TODO For the installing the packe run `pip install stablerls`.

You can also clone this repository and run `pip install -e StableRLS/` from the main main directory. This will also install the main dependencies which are included in `requirements.txt`. For active contribution you should also install the `optional-requirements.txt`, which also include the dependecies to build the documentation.

We decided to exclude the typical machine learning frameworks (PyTorch, Tensorflow) from the requirements, because everyone has their own preferences and we want to keep this package small. However, some of our example are based on PyTorch so you need to run `pip install torch` if you want to run them locally.

# Optional Req!
- Pandoc is required
- sphinx autobuild not working

### Matlab Version
The StableRLS package is able to compile a given MATLAB Simlink model to a FMU. The MATLAB engine package is a requirement for this. Before the MATLAB release R2022b it was inconvenient to install the engine see the [instructions](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). After the release it's possible to install it as pip package. StableRLS will try to install the MATLAB engine as dependency but if your MATLAB version is too old you have to referre to the MATLAB instructions to install the engine.  

## Get Started

## Contribution and other issues
We are researchers in the field of electrical engineering but this package is also useful for all engineers that use MATLAB Simulink as part of their research. If you want to collaborate and develop this tool further simply cerate issues or pull requrest.
In case of issues installing or using this tool you can also create an issue. For more information about contribution and issues have a look at [HOW_TO_CONTRIBUTE][HOW_TO_CONTRIBUTE.md].

## Building the documentation
Run `sphinx-autobuild docs/ docs/build/html` from the main directory and open the documentation `localhost:8000` on port 8000. The page is updated automatically after any file of the documentation is changed.

## More information
To install clean use `pip freeze | grep -v -f requirements.txt - | grep -v '^#' | grep -v '^-e ' | xargs pip uninstall -y` to remove all packages except the one with -e flag. Afterwards `pip install -r requirements.txt`. 

## Make this public
- [ ] [https://joss.theoj.org/about](https://joss.theoj.org/about) Submit