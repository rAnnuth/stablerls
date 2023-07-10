![](src/icon.png)

<h2 align="center">Stable Reinforcement Learning for Simulink</h2>

<p align="center">
<a href="https://stablerls.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/stablerls/badge/?version=latest"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


StableRLS is a software package that lets you use your existing MATLAB Simulink models in Python for reinforcement learning. Basically, your simulation is wrapped in a Python [Gymnasium](https://gymnasium.farama.org/) environment. The package provides the following features:
- automatic generation of input and output signals for your model
- automatic compilation of the Simulink model into a functional-mockup-unit (FMU) to enable fast simulation
- flexible post-processing implementation
- easy-to-read code

**And the best part is** that the only thing you need to do is:
- define a reward function to train your agent


## General Information
Reinforcement Learning (RL) is a rapidly changing and innovative field. The main purpose of this package is to combine the easy-to-use MATLAB Simulink modeling interface with the flexible and state-of-the-art Gymnasium interface. Therefore, the RL algorithm and the learning interface are out of scope for this package. However, we make the interface between Matlab and Python as easy as possible.

## Installation
This package is currently tested with Python 3.9.
To install the package, run `pip install StableRLS`. **You have to install the MATLAB engine seperately** because each MATLAB release has some specific requirements (see below).

You can also clone this repository and run `pip install -e StableRLS/` from the main directory. This will also install the main dependencies, which are listed in `requirements.txt`. To actively contribute, you should also install the `optional-requirements.txt`, which also includes the dependencies for building the documentation, by running `pip install -r optional-requirements.txt`.

We decided to exclude the typical machine learning frameworks (PyTorch, Tensorflow) from the requirements, because everyone has their own preferences and we want to keep this package small. But some of our example are based on PyTorch, so you need to run `pip install torch` if you want to run them locally. This will also be mentioned in the examples. To compile the documentation locally, you need to have Pandoc installed on your computer.


### Matlab Version
The MATLAB engine Python package is a requirement to compile a given MATLAB Simulink model. the MATLAB release R2022b it was inconvenient to install the engine, see the [instructions](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). After the release, it's possible to install the engine as a pip package. StableRLS won't try to install the MATLAB engine as dependency because the pip package only supports the newest MATLAB release. Currently, you can run `pip install matlabengine` if you have MATLAB 2023a installed, if you have MATLAB 2022b installed run `pip install matlabengine==9.13.7`. For other releases refer to the documentation mentioned.

## Get Started
Check out our examples (/examples) or the documentation, which also contains the examples.

## Contribution and other issues
We are researchers in the field of electrical engineering, but this package is also useful for other engineers who use MATLAB Simulink as part of their research. If you would like to contribute to the development of this tool, please create issues or pull requests.
If you have problems installing or using this tool, you can also create an issue. For more information about contributions and issues, take a look at "HOW_TO_CONTRIBUTE.md" located at the top of the repository.

## Building the documentation
Run `sphinx-autobuild docs/ docs/build/html` from the main directory and open the documentation `localhost:8000`. The page is updated automatically after any file in the documentation is changed.

## More information
To install clean use `pip freeze | grep -v -f requirements.txt - | grep -v '^#' | grep -v '^-e ' | xargs pip uninstall -y` to remove all packages except the one with -e flag. Afterwards `pip install -r requirements.txt` or `pip install -r optional-requirements.txt`.

## Make this public
- [ ] [https://joss.theoj.org/about](https://joss.theoj.org/about) Submit
