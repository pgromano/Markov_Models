import Markov_Models as mm

def from_ASCII(files, usecols=None):
    return mm.util.load.from_ASCII(files, usecols=usecols)

def from_CSV(files, usecols=None):
    try:
        return mm.util.load.from_CSV(files, usecols=usecols)
    except:
        return mm.util.load.from_ASCII(files, usecols=usecols)

def from_NPY(files):
    return mm.util.load.from_NPY(files)
