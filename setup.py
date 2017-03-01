try:
    from setuptools import setup
except:
    from distutils.core import setup
from distutils.extension import Extension

ext1 = Extension('Markov_Models.src._voronoi', ["Markov_Models/src/_voronoi.c"])
ext2 = Extension('Markov_Models.src._estimate', ["Markov_Models/src/_estimate.c"])

setup(
    name = 'Markov Models',
    version = '0.1',
    author = 'Pablo Romano',
    description = 'Python API for Generating Markov Models',
    url = 'https://github.com/pgromano/Markov_Models',

    packages = ['Markov_Models'],
    install_requires=[
        'hmmlearn',
        'msmtools',
        'numpy',
        'pandas',
        'scipy',
        'sklearn'
    ],
    ext_modules=[ext1,ext2],
    zip_safe = False
)
