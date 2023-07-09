import setuptools 

setuptools.setup(
  name='StableRLS',
  version='1.0.0',
  author='Robert Annuth',
  author_email='robert.annuth@tuhh.de',
  packages= setuptools.find_packages(),
  license='LICENSE.txt',
  description='Simulate Simulink FMUs as Gymnasium environment',
  url='https://github.com/rAnnuth/stablerls',
  keywords='simulation RL reinforcement learning Simulink matlab FMU',
  install_requires=[
      "numpy",
      "fmpy",
      "gymnasium",
      "pytest-ordering",
  ],
)
