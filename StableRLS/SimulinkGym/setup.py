import setuptools 

setuptools.setup(
  name='SimulinkGym',
  version='0.0.2',
  author='Robert Annuth',
  author_email='robert.annuth@tuhh.de',
  packages= setuptools.find_packages(),
  license='LICENSE.txt',
  description='Simulate Simulink FMUs as GYM environment',
  install_requires=[
      "gym",
      "numpy",
      "fmpy",
  ],
)
