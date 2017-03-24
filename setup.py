from setuptools import setup, Extension, find_packages
from os.path import join
import numpy

SRC_DIR = 'Markov_Models/analysis/src'
extensions = []

extensions.append(
    Extension(
        'Markov_Models.analysis.src._assignment',
        sources = [join(SRC_DIR, "_assignment.pyx")],
        include_dirs=[numpy.get_include()]),
)

extensions.append(
    Extension(
        'Markov_Models.analysis.src._mle_tmat_prinz',
        sources = [join(SRC_DIR, "_mle_tmat_prinz.pyx")],
        include_dirs=[numpy.get_include()]),
)

setup(
    name = 'Markov_Models',
    author = 'Pablo Romano',
    author_email = 'promano@uoregon.edu',
    description = 'Python API for Generating Markov Models',
    version = '0.1',
    url = 'https://github.com/pgromano/Markov_Models',

    packages = ['Markov_Models'],
    install_requires=[
        'bhmm',
        'hmmlearn',
        'msmtools',
        'numpy',
        'scipy',
        'scikit-learn'
    ],
    ext_modules=extensions,
    zip_safe = False,
)
