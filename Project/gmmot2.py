import numpy as np
import scipy.linalg as spl
from scipy.optimize import linprog



def emd(a, b, M, numItermax=100000, log=False, center_dual=True, check_marginals=True):
    # Ensure arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # If empty array given, use uniform distributions
    if len(a) == 0:
        a = np.ones(M.shape[0]) / M.shape[0]
    if len(b) == 0:
        b = np.ones(M.shape[1]) / M.shape[1]

    # Ensure dimensions match
    assert a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1], \
        "Dimension mismatch, check dimensions of M with a and b"

    # Ensure the same mass
    if check_marginals:
        np.testing.assert_almost_equal(a.sum(), b.sum(), err_msg='a and b must have the same sum', decimal=6)

    # Normalize b to match the total sum of a
    b = b * a.sum() / b.sum()

    # Linear programming approach for Earth Mover's Distance (Wasserstein distance)
    n, m = M.shape
    c = M.ravel()  # Cost function (flattened M)
    A_eq = np.zeros((n + m, n * m))

    # Constraints
    for i in range(n):
        A_eq[i, i*m:(i+1)*m] = 1  # Row sums (a)
    for j in range(m):
        A_eq[n+j, j::m] = 1  # Column sums (b)

    b_eq = np.hstack([a, b])  # Combine a and b

    # Solve the transportation problem using linear programming
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs', options={'maxiter': numItermax})

    if log:
        log_data = {
            'cost': res.fun,
            'status': res.message,
            'iterations': res.nit
        }
        return res.x.reshape(n, m), log_data

    return res.x.reshape(n, m)



def average_filter(u,r):
    # uniform filter with a square (2*r+1)x(2*r+1) window
    # u is a 2d image
    # r is the radius for the filter
    
    (nrow, ncol)                                      = u.shape
    big_uint                                          = np.zeros((nrow+2*r+1,ncol+2*r+1))
    big_uint[r+1:nrow+r+1,r+1:ncol+r+1]               = u
    big_uint                                          = np.cumsum(np.cumsum(big_uint,0),1)       # integral image
    
    out = big_uint[2*r+1:nrow+2*r+1,2*r+1:ncol+2*r+1] + big_uint[0:nrow,0:ncol] - big_uint[0:nrow,2*r+1:ncol+2*r+1] - big_uint[2*r+1:nrow+2*r+1,0:ncol]
    out = out/(2*r+1)**2
    
    return out

def guided_filter(u,guide,r,eps):
    C           = average_filter(np.ones(u.shape), r)   # to avoid image edges pb
    mean_u      = average_filter(u, r)/C
    mean_guide  = average_filter(guide, r)/C
    corr_guide  = average_filter(guide*guide, r)/C
    corr_uguide = average_filter(u*guide, r)/C
    var_guide   = corr_guide - mean_guide * mean_guide
    cov_uguide  = corr_uguide - mean_u * mean_guide
    
    alph = cov_uguide / (var_guide + eps)
    beta = mean_u - alph * mean_guide
    
    mean_alph = average_filter(alph, r)/C
    mean_beta = average_filter(beta, r)/C
    
    q = mean_alph * guide + mean_beta
    return q



def GaussianW2(m0,m1,Sigma0,Sigma1):
    # compute the quadratic Wasserstein distance between two Gaussians with means m0 and m1 and covariances Sigma0 and Sigma1
    Sigma00  = spl.sqrtm(Sigma0)
    Sigma010 = spl.sqrtm(Sigma00@Sigma1@Sigma00)
    d        = np.linalg.norm(m0-m1)**2+np.trace(Sigma0+Sigma1-2*Sigma010)
    return d


def GW2(pi0,pi1,mu0,mu1,S0,S1):
    # return the GW2 discrete map and the GW2 distance between two GMM
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    d  = mu0.shape[1]
    S0 = S0.reshape(K0,d,d)
    S1 = S1.reshape(K1,d,d)
    M  = np.zeros((K0,K1))
    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianW2(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    # Then we compute the OT distance or OT map thanks to the OT library
    wstar     = emd(pi0,pi1,M)         # discrete transport plan
    distGW2   = np.sum(wstar*M)
    return wstar,distGW2


def GaussianMap(m0,m1,Sigma0,Sigma1,x):
    # Compute the OT map (evaluated at x) between two Gaussians with means m0 and m1 and covariances Sigma0 and Sigma1 
    # m0 and m1 must be 2D arrays of size 1xd
    # Sigma0 and Sigma1 must be 2D arrays of size dxd
    # x can be a matrix of size n x d,
    # each column of x is a vector to which the function is applied
    d = Sigma0.shape[0]
    m0 = m0.reshape(1,d)
    m1 = m1.reshape(1,d)
    Sigma0 = Sigma0.reshape(d,d)
    Sigma1 = Sigma1.reshape(d,d)
    Sigma  = np.linalg.inv(Sigma0)@spl.sqrtm(Sigma0@Sigma1)
    Tx        = m1+(x-m0)@Sigma
    return Tx