from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from os.path import join
import numpy

SRC_DIR = 'Markov_Models/analysis/src'
extensions = []

setup(ext_modules = cythonize(join(SRC_DIR, "_assignment.pyx")), include_dirs=[numpy.get_include()])
extensions.append(
    Extension(
        'Markov_Models.analysis.src._assignment',
        sources = [join(SRC_DIR, "_assignment.c")],
        include_dirs=[numpy.get_include()]),
)

setup(ext_modules = cythonize(join(SRC_DIR, "_mle_tmat_prinz.pyx")), include_dirs=[numpy.get_include()])
extensions.append(
    Extension(
        'Markov_Models.analysis.src._mle_tmat_prinz',
        sources = [join(SRC_DIR, "_mle_tmat_prinz.c")],
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
        'hmmlearn',
        'msmtools',
        'numpy',
        'scipy',
        'sklearn'
    ],
    ext_modules=extensions,
    zip_safe = False,
)
