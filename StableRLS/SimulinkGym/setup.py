import setuptools 

setuptools.setup(
  name='StableRLS',
  version='0.0.3',
  author='Robert Annuth',
  author_email='robert.annuth@tuhh.de',
  packages= setuptools.find_packages(),
  license='LICENSE.txt',
  description='Simulate Simulink FMUs as Gymnasium environment',
  install_requires=[
      "numpy",
      "fmpy",
      "gymnasium",
      "matlabengine",
  ],
)
