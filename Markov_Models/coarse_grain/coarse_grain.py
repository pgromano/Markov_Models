from .pcca import PCCA
from .pcca_plus import PCCAPlus


def CoarseGrain(model, method='PCCAPlus', **kwargs):
    if method.lower() == 'pcca':
        return PCCA(**kwargs).coarse_grain(model)
    elif method.lower() == 'pccaplus':
        return PCCAPlus(**kwargs).coarse_grain(model)
    else:
        raise ValueError('Method {:s} not currently supported'.format(method))
