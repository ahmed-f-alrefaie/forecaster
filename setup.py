from setuptools import find_packages
from setuptools import setup

packages = find_packages(exclude=('tests', 'doc'))
provides = ['forecaster', ]

requires = []

install_requires = ['numpy', 'h5py']


setup(name='forecaster',
      author="Ahmed Al-Refaie, Jingjing Chen, David M. Kipping",
      description='Forecaster in package form',
      packages=packages,
      provides=provides,
      requires=requires,
      include_package_data=True,
      install_requires=install_requires)
