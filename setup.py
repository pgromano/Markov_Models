from setuptools import setup, Extension, find_packages
from os.path import join
import numpy

SRC_DIR = 'Markov_Models/estimation'
extensions = []

extensions.append(
    Extension(
        'Markov_Models.estimation._mle_tmat_prinz',
        sources = [join(SRC_DIR, "_mle_tmat_prinz.pyx")],
        include_dirs=[numpy.get_include()]),
)

extensions.append(
    Extension(
        'Markov_Models.estimation._simulate',
        sources = [join(SRC_DIR, "_simulate.pyx")],
        include_dirs=[numpy.get_include()]),
)

setup(
    name = 'Markov_Models',
    author = 'Pablo Romano',
    author_email = 'promano@uoregon.edu',
    description = 'Python API for Generating Markov Models',
    version = '0.3',
    url = 'https://github.com/pgromano/Markov_Models',

    packages = ['Markov_Models'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn'
    ],
    ext_modules=extensions,
    zip_safe = False,
)
