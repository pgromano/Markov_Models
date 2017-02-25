from numpy.distutils.core import setup, Extension

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
    zip_safe = False
)
