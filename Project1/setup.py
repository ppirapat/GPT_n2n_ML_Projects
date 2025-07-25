# This makes your code installable as a package.
#Python now knows how to find your modules inside src/.

from setuptools import setup, find_packages

setup(
    name='Project1',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)