import setuptools 

setuptools.setup(
  name='fmuSimulation',
  version='0.0.1',
  author='Robert Annuth',
  author_email='robert.annuth@tuhh.de',
  packages= setuptools.find_packages(),
  license='LICENSE.txt',
  description='Simulate FMUs as GYM environment',
  install_requires=[
      "gym",
      "numpy",
  ],
)
