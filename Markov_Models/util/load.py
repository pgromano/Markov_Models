import numpy as np

def from_ASCII(files):
    n_sets = len(files)
    data = []
    for i in range(n_sets):
        data_store = []
        for f in files[i]:
            data_store.append(np.genfromtxt(f))
        data.append(np.column_stack(data_store))
    return data

try:
    import pandas as pd
    def from_CSV(files):
        n_sets = len(files)
        data = []
        for i in range(n_sets):
            data_store = []
            for f in files[i]:
                data_store.append(pd.read_csv(f, header=None))
            data.append(np.array(pd.concat(data_store, axis=1)))
        return data
except:
    pass
    
def from_NPY(files):
    n_sets = len(files)
    data = []
    for i in range(n_sets):
        data_store = []
        for f in files[i]:
            data_store.append(np.load(f))
        data.append(np.column_stack(data_store))
    return data
