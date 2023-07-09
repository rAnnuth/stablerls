import setuptools 

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / '..' / "README.md").read_text()

setuptools.setup(
  name='StableRLS',
  version='1.0.3',
  author='Robert Annuth',
  author_email='robert.annuth@tuhh.de',
  packages= setuptools.find_packages(),
  package_data={'stablerls':['*']},
  license='LICENSE.txt',
  description='Simulate Simulink FMUs as Gymnasium environment',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/rAnnuth/stablerls',
  keywords='simulation RL reinforcement learning Simulink matlab FMU',
  install_requires=[
      "numpy",
      "fmpy",
      "gymnasium",
      "pytest-ordering",
  ],
)
