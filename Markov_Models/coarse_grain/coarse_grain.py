from .pcca import PCCA
from .pcca_plus import PCCAPlus


def CoarseGrain(model, method='PCCAPlus', **kwargs):
    if method.lower() == 'pcca':
        return PCCA(**kwargs).fit_transform(model)
    elif method.lower() == 'pccaplus':
        return PCCAPlus(**kwargs).fit_transform(model)
    else:
        raise ValueError('Method {:s} not currently supported'.format(method))
