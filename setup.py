
from setuptools import find_packages
from distutils.core import setup

packages = find_packages(exclude=('tests', 'doc'))
provides = ['taurex_molecular_emission', ]

requires = []

install_requires = ['taurex', ]

entry_points = {'taurex.plugins': 'molecular_emission = taurex_molecular_emission'}

setup(name='taurex_molecular_emission',
      author="Sam O. M. Wright",
      author_email="samuel.wright.13@ucl.ac.uk",
      license="BSD",
      description='Plugin for molecular specific emissions',
      packages=packages,
      entry_points=entry_points,
      provides=provides,
      requires=requires,
      install_requires=install_requires)