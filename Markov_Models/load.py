import Markov_Models as mm

def from_ASCII(files):
    return mm.util.load.from_ASCII(files)

def from_CSV(files):
    try:
        return mm.util.load.from_CSV(files)
    except:
        return mm.util.load.from_ASCII(files)

def from_NPY(files):
    return mm.util.load.from_NPY(files)
