def nonrev_T_matrix(C):
    return C/C.sum(1)[:,None]

def symmetric_T_estimator(C):
    Cs = 0.5*(C+C.T)
    return Cs/Cs.sum(1)[:,None]

def prinz_T_estimator(C):
    pass

def mcmc_T_estimator(C):
    pass
