# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 19:44:28 2015

@author: ylkomsamo
"""
import numpy as np
from numpy.dual import svd
from scipy.spatial.distance import pdist, squareform, cdist
import sys
import random
from time import time

if not 'memoizer' in globals():
    memoizer = {}

'''
Computes the product A_1 \otimes \ dots \otimes A_n Y 
    efficiently. O(len(Y)) time complexity and memory requirement.
'''
def kron_mvprod(As, b):
    x = b.copy()
     
    for i in xrange(len(As)):
        N = len(x)
        shape_i = As[i].shape
        X = np.reshape(x, (shape_i[1], N/shape_i[1]))
        Z = np.dot(As[i], X).T
        x = np.ndarray.flatten(Z)
    
    return x
'''
Computes the (unconditional) covariance matrix between two vectors.
'''
def covMatrix(X, Y, theta, symmetric = True, kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1])), \
        dist_f=None):
    if len(np.array(X).shape) == 1:
        _X = np.array([X]).T
    else:
        _X = np.array(X)
        
    if len(np.array(Y).shape) == 1:
        _Y = np.array([Y]).T
    else:
        _Y = np.array(Y)
        
    if dist_f == None:
        if symmetric:
            cM = pdist(_X)
            M = squareform(cM)
            M = kernel(M, theta)
            return M
        else:
            cM = cdist(_X, _Y)
            M = kernel(cM, theta)
            return M
    else:
        if symmetric:
            cM = pdist(_X, dist_f)
            M = squareform(cM)
            M = kernel(M, theta)
            return M
        else:
            cM = cdist(_X, _Y, dist_f)
            M = kernel(cM, theta)
            return M
    return
    
'''
Computes the (unconditional) covariance matrix of a Derivative Gaussian Process 
    at some times X and Y. Even indices correspond to the GP, odd indices correspond
    to its derivative.
    
    For instance, cov[0, 1] = cov(z_{t_0}), z_{t_0}^\prime)
'''     
def covMatrixDeriv(X, Y, theta, kernel, dkerneldx, d2kerneldxdy):
    _cov = np.zeros((2*len(X), 2*len(Y)))
    even_idx_x = [idx for idx in xrange(2*len(X)) if idx%2==0]
    odd_idx_x = [idx for idx in xrange(2*len(X)) if idx%2==1]
    
    even_idx_y = [idx for idx in xrange(2*len(Y)) if idx%2==0]
    odd_idx_y = [idx for idx in xrange(2*len(Y)) if idx%2==1]
    
    cov_11 = covMatrix(X, Y, theta, symmetric=False, kernel=kernel)
    cov_12 = covMatrix(X, Y, theta, symmetric=False, kernel=dkerneldx, dist_f= lambda u, v: v-u)
    cov_22 = covMatrix(X, Y, theta, symmetric=False, kernel=d2kerneldxdy)
    cov_21 = covMatrix(X, Y, theta, symmetric=False, kernel=dkerneldx, dist_f=lambda u, v: u-v)

    _cov[np.ix_(even_idx_x, even_idx_y)] = cov_11
    _cov[np.ix_(even_idx_x, odd_idx_y)] = cov_12
    _cov[np.ix_(odd_idx_x, even_idx_y)] = cov_21
    _cov[np.ix_(odd_idx_x, odd_idx_y)] = cov_22
    
    return _cov

    
    
    
'''
Inverts a positive-definite matrix taking care of conditioning
'''
def inv_cov(cov):
    U, S, V = svd(cov)
    eps = 0.0
    oc = np.max(S)/np.min(S)
    if oc > 1.0:
        nc = np.min([oc, 1e8])
        eps = np.min(S)*(oc-nc)/(nc-1.0)
    
    LI = np.dot(np.diag(1.0/(np.sqrt(np.absolute(S) + eps))), U.T)
    covI= np.dot(LI.T, LI)
    return covI
    
'''
Regularise a covariance matrix
'''
def regularise_cov(cov):
    U, S, V = svd(cov)
    eps = 0.0
    oc = np.max(S)/np.min(S)
    if oc > 1.0:
        nc = np.min([oc, 1e8])
        eps = np.min(S)*(oc-nc)/(nc-1.0)
    return np.dot(U, np.dot(np.diag(S)+eps, V))
    

'''
Computes the (conditional) covariance matrix of the values of a String GP 
    at a collection of times t in [a_{k-1}, a_k], conditional on boundary 
    conditions at a_{k-1} and a_k. Derived from Equation (4).
'''
def sCovMatrix(T, a_k_1, a_k, theta, kernel, dkerneldx, d2kerneldxdy):
    covNewNew11 = covMatrix(T, T, theta, symmetric=True, kernel=kernel)
    covNewNew12 = covMatrix(T, T, theta, symmetric=False, kernel=dkerneldx, dist_f= lambda u, v: v-u)
    covNewNew21 = covMatrix(T, T, theta, symmetric=False, kernel=dkerneldx, dist_f= lambda u, v: u-v)
    covNewNew22 = covMatrix(T, T, theta, symmetric=True, kernel=d2kerneldxdy)
    covNewNew = np.vstack([np.hstack([covNewNew11, covNewNew12]), np.hstack([covNewNew21, covNewNew22])])    
    
    covNewOld11 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=kernel)
    covNewOld12 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=dkerneldx, dist_f=lambda u, v: v-u)
    covNewOld21 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=dkerneldx, dist_f=lambda u, v: u-v)
    covNewOld22 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=d2kerneldxdy)
    covNewOld = np.vstack([np.hstack([covNewOld11, covNewOld12]), np.hstack([covNewOld21, covNewOld22])])

    covOldOld = np.array([\
        [kernel(0.0, theta), kernel(np.abs(a_k-a_k_1), theta), dkerneldx(0.0, theta), dkerneldx(a_k-a_k_1, theta)],\
        [kernel(np.abs(a_k-a_k_1), theta), kernel(0.0, theta), dkerneldx(a_k_1-a_k, theta), dkerneldx(0.0, theta)],\
        [dkerneldx(0.0, theta), dkerneldx(a_k_1-a_k, theta), d2kerneldxdy(0.0, theta), d2kerneldxdy(np.abs(a_k-a_k_1), theta)],\
        [dkerneldx(a_k-a_k_1, theta), dkerneldx(0.0, theta), d2kerneldxdy(np.abs(a_k-a_k_1), theta), d2kerneldxdy(0.0, theta)]
    ])
    covOldOldI = inv_cov(covOldOld)
    ret = covNewNew - np.dot(covNewOld, np.dot(covOldOldI, covNewOld.T))
    
    return ret
    
'''
Computes the (conditional) mean of the values of a String GP at a collection of
    times t in [a_{k-1}, a_k], conditional on boundary conditions at a_{k-1} and a_k.
    Derived from Equation (3).
'''
def sMean(T, a_k_1, a_k, z_k_1, z_k, z_k_1_prime, z_k_prime, theta, kernel, dkerneldx, d2kerneldxdy, m=lambda u: 0, dmdx=lambda u: 0):    
    covNewOld11 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=kernel)
    covNewOld12 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=dkerneldx, dist_f= lambda u, v: v-u)
    covNewOld21 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=dkerneldx, dist_f= lambda u, v: u-v)
    covNewOld22 = covMatrix(T, np.array([a_k_1, a_k]), theta, symmetric=False, kernel=d2kerneldxdy)
    covNewOld = np.vstack([np.hstack([covNewOld11, covNewOld12]), np.hstack([covNewOld21, covNewOld22])])

    covOldOld = np.array([\
        [kernel(0.0, theta), kernel(np.abs(a_k-a_k_1), theta), dkerneldx(0.0, theta), dkerneldx(a_k-a_k_1, theta)],\
        [kernel(np.abs(a_k-a_k_1), theta), kernel(0.0, theta), dkerneldx(a_k_1-a_k, theta), dkerneldx(0.0, theta)],\
        [dkerneldx(0.0, theta), dkerneldx(a_k_1-a_k, theta), d2kerneldxdy(0.0, theta), d2kerneldxdy(np.abs(a_k-a_k_1), theta)],\
        [dkerneldx(a_k-a_k_1, theta), dkerneldx(0.0, theta), d2kerneldxdy(np.abs(a_k-a_k_1), theta), d2kerneldxdy(0.0, theta)]
    ])
    covOldOldI = inv_cov(covOldOld)
    boundaries = np.array([z_k_1-m(a_k_1), z_k-m(a_k), z_k_1_prime-dmdx(a_k_1), z_k_prime-dmdx(a_k)])
    ret = np.dot(covNewOld, np.dot(covOldOldI, boundaries))
    
    return ret

'''
Computes the cross-covariance matrix of a Derivative Gaussian Process at a point a_1,
    and a_2.
'''
def bCrossCovMatrix0(a_1, a_2, theta, kernel, dkerneldx, d2kerneldxdy):
    C = np.array([[kernel(a_1-a_2, theta), dkerneldx(a_2-a_1, theta)],\
                   [dkerneldx(a_1-a_2, theta), d2kerneldxdy(a_1-a_2, theta)]])
    return C
        
'''
Computes the covariance matrix of a Derivative Gaussian Process at a point a_0,
    or equivalently the covariance matrix of a String GP at the start of the 
    index interval. (Equation (6))
'''
def bCovMatrix0(a_0, theta, kernel, dkerneldx, d2kerneldxdy):
    return bCrossCovMatrix0(a_0, a_0, theta, kernel, dkerneldx, d2kerneldxdy)
    
'''
Computes the mean of a Derivative Gaussian Process at a point a_0, or equivalently
    the mean of a String GP at the start of the index interval. (Equation (8))
'''
def bMean0(a_0, m=lambda u: 0, dmdx=lambda u: 0):
    return np.array([m(a_0), dmdx(a_0)])

'''
Computes the covariance matrix of the value of a Derivative String GP at a_k 
    conditional on its value at a_k_1. (Equation (7))
'''
def bCovMatrixk(a_k, a_p, theta, kernel, dkerneldx, d2kerneldxdy):
    Ckk = bCovMatrix0(a_k, theta, kernel, dkerneldx, d2kerneldxdy)
    Ckp = bCrossCovMatrix0(a_k, a_p, theta, kernel, dkerneldx, d2kerneldxdy)
    Cpp = bCovMatrix0(a_p, theta, kernel, dkerneldx, d2kerneldxdy)
    CppI = inv_cov(Cpp)
    C = Ckk - np.dot(Ckp, np.dot(CppI, Ckp.T))
    return C
    

'''
Computes the mean of the value of a Derivative String GP at a_k conditional on 
    its value at a_k_1. (Equation (8))
'''
def bMeank(a_k, a_p, theta, kernel, dkerneldx, d2kerneldxdy, z_p, dz_p, m=lambda u: 0, dmdx=lambda u: 0):
    Mk = bMean0(a_k, m=m, dmdx=dmdx)
    Cpp = bCovMatrix0(a_p, theta, kernel, dkerneldx, d2kerneldxdy)
    CppI = inv_cov(Cpp)
    Ckp = bCrossCovMatrix0(a_k, a_p, theta, kernel, dkerneldx, d2kerneldxdy)
    M = Mk + np.dot(Ckp, np.dot(CppI, np.array([z_p-m(a_p), dz_p-dmdx(a_p)])))
    return M 

#'''
#Computes the factor L = US^{1/2}, where cov = USV is the SVD decomposition of 
#    the covariance matrix cov.
#    
#This function should typically be called in parallel.
#'''
#def SVDFactorise(cov):
#    U, S, V = svd(cov)
#    eps = 0.0
#    oc = np.max(S)/np.min(S)
#    if oc > 1.0:
#        nc = np.min([oc, 1e10])
#        eps = np.min(S)*(oc-nc)/(nc-1.0)
#        
#    L = np.dot(U, np.diag(np.sqrt(S+eps)))
#    return L
    

'''
Inverse hyper-parameters under a sigmoid gaussian prior
'''
def inv_theta(theta, amplitudes):
    n = len(amplitudes)
    res = []
    for i in range(n):
        res += [np.log(theta[i]/(amplitudes[i]-theta[i]))]
        
    return np.array(res)
    

def sigmoid(x):
    if isinstance(x, (frozenset, list, set, tuple, np.ndarray)) and (np.array(x) < -308).all():
        return np.array([sys.float_info.min for _ in x]);  
    elif not isinstance(x, (frozenset, list, set, tuple, np.ndarray)) and x < -308:
        return sys.float_info.min;
        
    return 1.0/(1+np.exp(-x));
    
    

'''
Performs one cycle of Elliptical Slice Sampling
    (Murray et al. 2010)

    The Gaussian prior is assumed to be standard.
'''
def elliptical_slice_sampler(X, logLikelihood, tolerance=0.01):
    _nu = np.random.randn(len(X))
    _u = random.uniform(0.0, 1.0)
    _lyOld = logLikelihood(X) + np.log(_u)
    _a = random.uniform(0.0, 2.0*np.pi)
    _aMin = _a - 2.0*np.pi
    _aMax = _a
        
    while np.abs((_aMax-_aMin)) > tolerance: 
        _X =   X*np.cos(_a) + _nu*np.sin(_a)            
        _lyNew = logLikelihood(_X)
            
        if _lyNew > _lyOld:
            break
        else:
            _a = random.uniform(_aMin, _aMax)
            if _a < 0.0:
                _aMin = _a
            else:
                _aMax = _a
    
    return _X
    

'''
Compute the covariance matrix at the boundary conditions of a uniform String GP kernel
'''
def boundaries_cov(b_times, theta, kernel, dkerneldx, d2kerneldxdy,\
        thetas = None):
    # Compute the covariance matrix at the boundary times
    sorted_b_times = np.sort(b_times)
    
    # Covarience matrix
    cov = np.zeros((2*len(sorted_b_times), 2*len(sorted_b_times)))
    
    for p in xrange(1, len(sorted_b_times)):
        if thetas != None:
            theta_p = thetas[p-1].copy()
        else:
            theta_p = theta.copy()
            
        if p == 1:
            # Handle the case p==1 separately as the first string is a fully
            # fledge GP
            select_0 = [0,1]
            select_1 = [2,3]
            
            # The covariance matrix of the first boundary condition
            a_0 = sorted_b_times[0]
            cov_a_0 = bCovMatrix0(a_0, theta_p, kernel, dkerneldx, d2kerneldxdy)
            
            # The covariance matrix of the second boundary condition
            a_1 = sorted_b_times[1]
            cov_a_1 = bCovMatrix0(a_1, theta_p, kernel, dkerneldx, d2kerneldxdy)
            
            # The cross-covariance matrix between the first and the second boundary
            cov_a_0_a_1 = bCrossCovMatrix0(a_0, a_1, theta_p, kernel, dkerneldx, d2kerneldxdy)      
            
            # Update the big covariance matrix
            cov[np.ix_(select_0, select_0)] = cov_a_0.copy()
            cov[np.ix_(select_1, select_1)] = cov_a_1.copy()
            cov[np.ix_(select_0, select_1)] = cov_a_0_a_1.copy()
            cov[np.ix_(select_1, select_0)] = cov_a_0_a_1.copy().T
            
        else:
            a_p_1 = sorted_b_times[p-1]
            a_p = sorted_b_times[p]
            
            # Covariance matrices under the unconditional cov structure.
            cov_a_p_1 = bCovMatrix0(a_p_1, theta_p, kernel, dkerneldx, d2kerneldxdy)
            cov_a_p = bCovMatrix0(a_p, theta_p, kernel, dkerneldx, d2kerneldxdy)
            cov_a_p_a_p_1 = bCrossCovMatrix0(a_p, a_p_1, theta_p, kernel, dkerneldx, d2kerneldxdy)
            
            # {}^b_k M coefficient
            Mb = np.dot(cov_a_p_a_p_1, np.linalg.inv(cov_a_p_1))
            
            # Covariance at p conditional on values at p-1
            cond_p_1_cov_p = cov_a_p - np.dot(Mb, cov_a_p_a_p_1.T)
            
            # Global covariance matrix of the boundary conditions at the previous time
            select_p_1 = [2*p-2, 2*p-1]
            global_cov_p_1 = cov[np.ix_(select_p_1, select_p_1)]
            select_p = [2*p, 2*p+1]
    
            for q in xrange(p+1):                
                select_q = [2*q, 2*q+1]
                
                if q == p:
                    tmp_cov = cond_p_1_cov_p + np.dot(Mb, np.dot(global_cov_p_1, Mb.T))
                else:
                    global_cov_p_1_q = cov[np.ix_(select_p_1, select_q)]
                    tmp_cov = np.dot(Mb, global_cov_p_1_q)
                
                # Update the auto or cross covariance matrix.
                cov[np.ix_(select_p, select_q)] = tmp_cov.copy()
                cov[np.ix_(select_q, select_p)] = tmp_cov.copy().T
                
    return cov
                
                
        
'''
Compute the covariance matrix of a uniform String GP, with boundary times b_times,
    at string times s_times.
'''
def string_gp_kernel_cov(s_times, b_times, theta, kernel, dkerneldx, d2kerneldxdy,\
        thetas = None):
    
    # 0. Group together the points that belong to the same string
    sorted_s_times = np.sort(s_times)
    sorted_b_times = np.sort(b_times)
    grouped_s_times = []
    grouped_s_idx = []
    
    n = 0
    
    for i in xrange(len(sorted_b_times)-1):
        times = [elt for elt in sorted_s_times if elt > sorted_b_times[i] and elt <= sorted_b_times[i+1]]
        grouped_s_times += [times]
        grouped_s_idx += [range(n, n+2*len(times))]
        n += 2*len(times)
    
    # Variable to return
    global_cov = np.zeros((n, n))
    
    # 1. Compute the covariance matrix at the boundary conditions.
    bound_cov = boundaries_cov(b_times, theta, kernel, dkerneldx, d2kerneldxdy,\
        thetas = thetas)

    # 2. Compute the covariance.
    for p in xrange(1, len(sorted_b_times)):
        idx_l = [2*p-2, 2*p-1, 2*p, 2*p+1]
        times_p = grouped_s_times[p-1]
        a_p_1 = sorted_b_times[p-1]
        a_p = sorted_b_times[p]
        
        if thetas != None:
            theta_p = thetas[p-1].copy()
        else:
            theta_p = theta.copy()
        
        # Unconditional covariance matrix of the p-th String at its boundaries                
        _unc_bound_cov_p = covMatrixDeriv(np.array([a_p_1, a_p]), np.array([a_p_1, a_p]),\
            theta_p, kernel, dkerneldx, d2kerneldxdy)
        inv_unc_bound_cov_p = np.linalg.inv(_unc_bound_cov_p)
        
        for q in xrange(1, len(sorted_b_times)):
            if thetas != None:
                theta_q = thetas[q-1].copy()
            else:
                theta_q = theta.copy()
                
            idx_c = [2*q-2, 2*q-1, 2*q, 2*q+1]
            b_cov = bound_cov[np.ix_(idx_l, idx_c)]
            times_q = grouped_s_times[q-1]
            
            # If there is no string time on this string just move on
            if len(times_q) == 0:
                continue
            
            a_q_1 = sorted_b_times[q-1]
            a_q = sorted_b_times[q]
            
            # Unconditional covariance matrix of the q-th String at its boundaries   
            _unc_bound_cov_q = covMatrixDeriv(np.array([a_q_1, a_q]), np.array([a_q_1, a_q]),\
                theta_q, kernel, dkerneldx, d2kerneldxdy)
            inv_unc_bound_cov_q = np.linalg.inv(_unc_bound_cov_q)
            
                
            if p==q:

                # Unconditional covariance matrix of the j-th String GP at the string times times_p
                _unc_times_cov = covMatrixDeriv(times_p, times_p, theta_p, kernel, dkerneldx,\
                    d2kerneldxdy)
                
                # Unconditional cross-covariance matrix of the j-th String GP 
                #   between the String times and the boundary times.
                _unc_times_bound_cov = covMatrixDeriv(times_p, np.array([a_p_1, a_p]),\
                    theta_p, kernel, dkerneldx, d2kerneldxdy)
                                
                # Covariance matrix of the values of the j-th String GP at the string times s_times,
                #   conditional on its values at the boundary conditions.
                cond_cov_p_q = _unc_times_cov - np.dot(_unc_times_bound_cov, np.dot(inv_unc_bound_cov_p, _unc_times_bound_cov.T))
            else:
                cond_cov_p_q = np.zeros((2*len(times_p), 2*len(times_q)))
                
                
            _unc_times_p_bound_cov = covMatrixDeriv(times_p, np.array([a_p_1, a_p]), theta_p, kernel,\
                dkerneldx, d2kerneldxdy)
                
            _unc_times_q_bound_cov = covMatrixDeriv(times_q, np.array([a_q_1, a_q]), theta_q, kernel,\
                dkerneldx, d2kerneldxdy)
                
            lambda_p = np.dot(_unc_times_p_bound_cov, inv_unc_bound_cov_p)
            lambda_q = np.dot(_unc_times_q_bound_cov, inv_unc_bound_cov_q)
            
            # Global covariance matrix between times_p and times_q
            cov_p_q = cond_cov_p_q + np.dot(lambda_p, np.dot(b_cov, lambda_q.T))
            
            global_cov[np.ix_(grouped_s_idx[p-1], grouped_s_idx[q-1])] = cov_p_q.copy()
                       
            
    return global_cov
                    
'''
Computes the inverse and the determinant of a covariance matrix in one go, using
    SVD.
    Returns a structure containing the following keys:
        inv: the inverse of the covariance matrix,
        L: the pseudo-cholesky factor US^0.5,
        det: the determinant of the covariance matrix.
'''
def SVDFactorise(cov):
    U, S, V = svd(cov)
    eps = 0.0
    oc = np.max(S)/np.min(S)
    if oc > 1.0:
        nc = np.min([oc, 1e8])
        eps = np.min(S)*(oc-nc)/(nc-1.0)

    L = np.dot(U, np.diag(np.sqrt(S+eps)))        
    LI = np.dot(np.diag(1.0/(np.sqrt(np.absolute(S) + eps))), U.T)
    covI= np.dot(LI.T, LI)
    
    res = {}
    res['inv'] = covI.copy()
    res['L'] = L.copy()    
    res['det'] = np.prod(S+eps)
    res['LI'] = LI.copy()
    res['eigen_vals'] = S+eps
    res['u'] = U.copy()
    return res 

'''
Computes the hyper-parameters and the noise variance of the GP regression model
    under i.i.d Gaussian noise.
'''
def gp_regression_calibrate(X, Y, hyper_type = 'SE', x_0 = np.array([1.0, 1.0, 1.0 ]),\
    penalty_center=0.0):
        
    from scipy.optimize import fmin_bfgs
    from numpy.core.umath_tests import inner1d
    
    if hyper_type == 'MA32':
        kernel = lambda u, theta: theta[0]*theta[0]*(1+(np.sqrt(3.0)/theta[1])*\
            np.abs(u))*np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
        # Derivative of the kernel with respect to the input length scale
        kernel_d2 = lambda u, theta: theta[0]*theta[0]*(3.0/(theta[1]**3)*u*u)*\
            np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
    else:
        kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))
        # Derivative of the kernel with respect to the input length scale
        kernel_d2 = lambda u, theta: kernel(u, theta)*u*u/(theta[1]*theta[1]*theta[1])
        

    
    def log_marginal(x):
        noise_var = x[0]*x[0]
        theta = np.abs(x[1:])
    
        cov = covMatrix(X, X, theta, symmetric=True, kernel=kernel) + noise_var*np.eye(len(X))
        svd_factor = SVDFactorise(cov)
        cov_i = svd_factor['inv']
        cov_det = svd_factor['det']
        res = np.log(cov_det)+np.dot(Y, np.dot(cov_i, Y))
        
        if penalty_center != None:
            res += np.dot(x-np.array([penalty_center]*len(x)),x-np.array([penalty_center]*len(x)))
        
        return res
        
        
    def grad_log_marginal(x):
        noise_var = x[0]*x[0]
        theta = np.abs(x[1:])
        # p.114 Rassmussen and Williams
        cov = covMatrix(X, X, theta, symmetric=True, kernel=kernel)
        K =  cov + noise_var*np.eye(len(X))
        svd_factor = SVDFactorise(K)
        K_i = svd_factor['inv']
        alpha = np.dot(K_i, Y)
        trace_factor = np.outer(alpha, alpha) - K_i
        # d covariance matrix / d noise std
        dcov_0 = 2.0*np.abs(x[0])*np.eye(len(X))
        # d covariance matrix / d output scale
        theta_d1 = np.array([2.0*np.abs(theta[0])]+ list(theta[1:]))
        dcov_1 = covMatrix(X, X, theta_d1, symmetric=True, kernel=kernel)
        # d covariance matrix / d input scale
        dcov_2 = covMatrix(X, X, theta, symmetric=True, kernel=kernel_d2)
        
        res = -np.array([np.sum(inner1d(trace_factor, dcov_0.T)),\
            np.sum(inner1d(trace_factor, dcov_1.T)),
            np.sum(inner1d(trace_factor, dcov_2.T))])
            
        if penalty_center != None:
            res += 2.0*(x-np.array([penalty_center]*len(x)))


        return res
        
    # Attempt 1: warm-up/smart initialisation
    x_opt = fmin_bfgs(log_marginal, x_0, fprime=grad_log_marginal)
    # Attempt 2: max from smart initialisation
    x_opt = fmin_bfgs(log_marginal, np.abs(x_opt), fprime=grad_log_marginal)
    
    return (x_opt[0]*x_opt[0], np.abs(x_opt[1:]))
    



'''
Computes the hyper-parameters and the noise variance of the GP regression model
    under i.i.d Gaussian noise over training inputs forming a grid.
'''
def image_gp_regression_calibrate(X, Y, hyper_type = 'SM', x_0 = np.array([1.0, 1.0, 1.0 ]),\
    penalty_center=0.0):
        
    from scipy.optimize import fmin_bfgs
    from numpy.core.umath_tests import inner1d
    
    if hyper_type == 'MA32':
        kernel = lambda u, theta: theta[0]*theta[0]*(1+(np.sqrt(3.0)/theta[1])*\
            np.abs(u))*np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
        # Derivative of the kernel with respect to the input length scale
        kernel_d2 = lambda u, theta: theta[0]*theta[0]*(3.0/(theta[1]**3)*u*u)*\
            np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
    else:
        kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))
        # Derivative of the kernel with respect to the input length scale
        kernel_d2 = lambda u, theta: kernel(u, theta)*u*u/(theta[1]*theta[1]*theta[1])
        

    # Spectral mixture kernel
    sm_kernel_single_single = lambda u, theta: theta[0]*theta[0]*np.exp(-2.0*np.pi*np.pi*u*u*\
            theta[1]*theta[1])*np.cos(2.0*np.pi*u*theta[2])
            
    def sm_kernel(u, theta):
        theta_reshaped = np.reshape(theta, (len(theta)/3, 3))
        res = None
        for j in xrange(len(theta)/3):
            if res == None:
                res = sm_kernel_single_single(u, theta_reshaped[j])
            else:
                res += sm_kernel_single_single(u, theta_reshaped[j])
        return res
    
    # SE kernel
    se_kernel = lambda u, theta: theta[0]*theta[0]*(1+(np.sqrt(3.0)/theta[1])*\
            np.abs(u))*np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
    # Derivative of the kernel with respect to the input length scale
    se_kernel_d2 = lambda u, theta: theta[0]*theta[0]*(3.0/(theta[1]**3)*u*u)*\
            np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
    
    # Matern 3/2
    ma_32_kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    # Derivative of the kernel with respect to the input length scale
    ma_32_kernel_d2 = lambda u, theta: kernel(u, theta)*u*u/(theta[1]*theta[1]*theta[1])
    
    if hyper_type == 'SM':
        kernel = sm_kernel
            
    elif hyper_type == 'MA32':
        kernel = ma_32_kernel
    else:
        # Default to the SE kernel
        kernel = se_kernel

    def log_marginal(x):
        noise_var = x[0]*x[0]
        theta = np.abs(x[1:])
        grid_x = np.sort(list(set([_[0] for _ in X])))
        grid_y = np.sort(list(set([_[1] for _ in X])))
        cov_x = covMatrix(grid_x, grid_x, theta, symmetric=True, kernel=kernel)
        cov_y = covMatrix(grid_y, grid_y, theta, symmetric=True, kernel=kernel)
        svd_factor_x = SVDFactorise(cov_x)
        svd_factor_y = SVDFactorise(cov_y)
        eigen_vals_x = svd_factor_x['eigen_vals']
        eigen_vals_y = svd_factor_y['eigen_vals']
        u_x = svd_factor_x['u']
        u_y = svd_factor_y['u']
        
        res = 0.0
        eigen_vals = np.array([noise_var+ eigen_vals_x[i]*eigen_vals_x[j]\
            for i in xrange(len(eigen_vals_x)) for j in xrange(len(eigen_vals_y))])
                
        log_det = np.sum([np.log(_) for _ in eigen_vals])
        res += log_det
        
        kron_dot_prod = kron_mvprod([u_x, u_y], Y)
        res += np.dot(kron_dot_prod, np.multiply(1.0/eigen_vals, kron_dot_prod))
        
        if penalty_center != None:
            res += np.dot(x-np.array([penalty_center]*len(x)),x-np.array([penalty_center]*len(x)))
                        
        return res
      
      
    # Attempt 1: warm-up/smart initialisation
    x_opt = fmin_bfgs(log_marginal, x_0)
    
    return (x_opt[0]*x_opt[0], np.abs(x_opt[1:]))
    



def popular_kernels_distortion():
    theta = np.array([1.0, 0.5, 1.0])
    theta_rq_5 = np.array([1.0, 0.5, 5.0])
    s_times = np.arange(0.0001, 1.0, 0.01)
    
    t = time()
    import pylab as plt
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True,)
    ticks = [0.5]
    
    #################
    #   SE Kernel   # 
    #################
    se_kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    se_dkerneldx = lambda u, theta: -((theta[0]/theta[1])**2)*u*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    se_d2kerneldxdy = lambda u, theta: ((theta[0]/theta[1])**2)*(1.0-(u/theta[1])**2)*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    
    res_0 = covMatrix(s_times, s_times, theta, symmetric=True, kernel=se_kernel)
    
    # 2 Strings 
    b_times = np.array([0.0, 0.5, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, se_kernel, se_dkerneldx, se_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_2 = res[np.ix_(evn, evn)]
    
    # 4 Strings 
    b_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, se_kernel, se_dkerneldx, se_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_4 = res[np.ix_(evn, evn)]
    
    # 8 Strings 
    b_times = np.array([i/8.0 for i in range(9)])
    res = string_gp_kernel_cov(s_times, b_times, theta, se_kernel, se_dkerneldx, se_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_8 = res[np.ix_(evn, evn)]
    
    # 16 Strings 
    b_times = np.array([i/16.0 for i in range(17)])
    res = string_gp_kernel_cov(s_times, b_times, theta, se_kernel, se_dkerneldx, se_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_16 = res[np.ix_(evn, evn)]
    
    X, Y = plt.meshgrid(s_times, s_times)
    
    vmin = min(abs(res_0).min(), abs(res_2).min(), abs(res_4).min(), abs(res_8).min(), abs(res_16).min())
    vmax = max(abs(res_0).max(), abs(res_2).max(), abs(res_4).max(), abs(res_8).max(), abs(res_16).max())
    
    p = axes[0, 0].pcolor(X, Y, res_0, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(r"$K=1$")
    axes[0, 0].set_ylabel('SE', rotation=90)
    axes[0, 0].yaxis.set_ticks(ticks)
    axes[0, 0].xaxis.set_ticks(ticks)
    p = axes[0, 1].pcolor(X, Y, res_2, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(r"$K=2$")
    p = axes[0, 2].pcolor(X, Y, res_4, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(r"$K=4$")
    p = axes[0, 3].pcolor(X, Y, res_8, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[0, 3].set_title(r"$K=8$")
    p = axes[0, 4].pcolor(X, Y, res_16, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[0, 4].set_title(r"$K=16$")
    #cb = fig.colorbar(p, ax=axes[0, 5])
    
    print '"""""""""""""""""""""""""'
    print 'ABSOLUTE DEVIATION SE'
    print '"""""""""""""""""""""""""'
    print ''
    print 'K=2', "%.2f"%np.amin(np.abs(res_0-res_2)), "%.2f"%np.mean(np.abs(res_0-res_2)), "%.2f"%np.amax(np.abs(res_0-res_2))
    print 'K=4', "%.2f"%np.amin(np.abs(res_0-res_4)), "%.2f"%np.mean(np.abs(res_0-res_4)), "%.2f"%np.amax(np.abs(res_0-res_4))
    print 'K=8', "%.2f"%np.amin(np.abs(res_0-res_8)), "%.2f"%np.mean(np.abs(res_0-res_8)), "%.2f"%np.amax(np.abs(res_0-res_8))
    print 'K=16', "%.2f"%np.amin(np.abs(res_0-res_16)), "%.2f"%np.mean(np.abs(res_0-res_16)), "%.2f"%np.amax(np.abs(res_0-res_16))
    
    #################
    #   RQ 1 Kernel # 
    #################
    #kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    #dkerneldx = lambda u, theta: -((theta[0]/theta[1])**2)*u*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    #d2kerneldxdy = lambda u, theta: ((theta[0]/theta[1])**2)*(1.0-(u/theta[1])**2)*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    
    rq_kernel = lambda u, theta: theta[0]*theta[0]*((1.0+u*u/(2*theta[2]*theta[1]*theta[1]))**(-theta[2]))
    rq_dkerneldx = lambda u, theta: -((theta[0]/theta[1])**2)*u*((1.0+u*u/(2*theta[2]*theta[1]*theta[1]))**(-theta[2]-1.0))
    rq_d2kerneldxdy = lambda u, theta: ((theta[0]/theta[1])**2)*((1.0+u*u/(2*theta[2]*theta[1]*theta[1]))**(-theta[2]-2.0))*\
        (1.0+((u/theta[1])**2)*(-1.0-0.5/theta[2]))
    
    res_0 = covMatrix(s_times, s_times, theta, symmetric=True, kernel=rq_kernel)
    
    # 2 Strings 
    b_times = np.array([0.0, 0.5, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_2 = res[np.ix_(evn, evn)]
    
    # 4 Strings 
    b_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_4 = res[np.ix_(evn, evn)]
    
    # 8 Strings 
    b_times = np.array([i/8.0 for i in range(9)])
    res = string_gp_kernel_cov(s_times, b_times, theta, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_8 = res[np.ix_(evn, evn)]
    
    # 16 Strings 
    b_times = np.array([i/16.0 for i in range(17)])
    res = string_gp_kernel_cov(s_times, b_times, theta, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_16 = res[np.ix_(evn, evn)]
    
    X, Y = plt.meshgrid(s_times, s_times)
    
    vmin = min(abs(res_0).min(), abs(res_2).min(), abs(res_4).min(), abs(res_8).min(), abs(res_16).min())
    vmax = max(abs(res_0).max(), abs(res_2).max(), abs(res_4).max(), abs(res_8).max(), abs(res_16).max())
    
    p = axes[1, 0].pcolor(X, Y, res_0, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[1, 0].set_ylabel('RQ 1', rotation=90)
    axes[1, 0].yaxis.set_ticks(ticks)
    axes[1, 0].xaxis.set_ticks(ticks)
    p = axes[1, 1].pcolor(X, Y, res_2, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[1, 2].pcolor(X, Y, res_4, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[1, 3].pcolor(X, Y, res_8, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[1, 4].pcolor(X, Y, res_16, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    
    
    print '"""""""""""""""""""""""""'
    print 'ABSOLUTE DEVIATION RQ1'
    print '"""""""""""""""""""""""""'
    print ''
    print 'K=2', "%.2f"%np.amin(np.abs(res_0-res_2)), "%.2f"%np.mean(np.abs(res_0-res_2)), "%.2f"%np.amax(np.abs(res_0-res_2))
    print 'K=4', "%.2f"%np.amin(np.abs(res_0-res_4)), "%.2f"%np.mean(np.abs(res_0-res_4)), "%.2f"%np.amax(np.abs(res_0-res_4))
    print 'K=8', "%.2f"%np.amin(np.abs(res_0-res_8)), "%.2f"%np.mean(np.abs(res_0-res_8)), "%.2f"%np.amax(np.abs(res_0-res_8))
    print 'K=16', "%.2f"%np.amin(np.abs(res_0-res_16)), "%.2f"%np.mean(np.abs(res_0-res_16)), "%.2f"%np.amax(np.abs(res_0-res_16))
    
    #################
    #   RQ 2 Kernel # 
    #################
    
    rq_kernel = lambda u, theta: theta[0]*theta[0]*((1.0+u*u/(2*theta[2]*theta[1]*theta[1]))**(-theta[2]))
    rq_dkerneldx = lambda u, theta: -((theta[0]/theta[1])**2)*u*((1.0+u*u/(2*theta[2]*theta[1]*theta[1]))**(-theta[2]-1.0))
    rq_d2kerneldxdy = lambda u, theta: ((theta[0]/theta[1])**2)*((1.0+u*u/(2*theta[2]*theta[1]*theta[1]))**(-theta[2]-2.0))*\
        (1.0+((u/theta[1])**2)*(-1.0-0.5/theta[2]))
    
    res_0 = covMatrix(s_times, s_times, theta_rq_5, symmetric=True, kernel=rq_kernel)
    
    # 2 Strings 
    b_times = np.array([0.0, 0.5, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta_rq_5, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_2 = res[np.ix_(evn, evn)]
    
    # 4 Strings 
    b_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta_rq_5, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_4 = res[np.ix_(evn, evn)]
    
    # 8 Strings 
    b_times = np.array([i/8.0 for i in range(9)])
    res = string_gp_kernel_cov(s_times, b_times, theta_rq_5, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_8 = res[np.ix_(evn, evn)]
    
    # 16 Strings 
    b_times = np.array([i/16.0 for i in range(17)])
    res = string_gp_kernel_cov(s_times, b_times, theta_rq_5, rq_kernel, rq_dkerneldx, rq_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_16 = res[np.ix_(evn, evn)]
    
    X, Y = plt.meshgrid(s_times, s_times)
    
    vmin = min(abs(res_0).min(), abs(res_2).min(), abs(res_4).min(), abs(res_8).min(), abs(res_16).min())
    vmax = max(abs(res_0).max(), abs(res_2).max(), abs(res_4).max(), abs(res_8).max(), abs(res_16).max())
    
    p = axes[2, 0].pcolor(X, Y, res_0, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[2, 0].set_ylabel('RQ 5', rotation=90)
    axes[2, 0].yaxis.set_ticks(ticks)
    axes[2, 0].xaxis.set_ticks(ticks)
    p = axes[2, 1].pcolor(X, Y, res_2, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[2, 2].pcolor(X, Y, res_4, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[2, 3].pcolor(X, Y, res_8, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[2, 4].pcolor(X, Y, res_16, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    
    

    print '"""""""""""""""""""""""""'
    print 'ABSOLUTE DEVIATION RQ5'
    print '"""""""""""""""""""""""""'
    print ''
    print 'K=2', "%.2f"%np.amin(np.abs(res_0-res_2)), "%.2f"%np.mean(np.abs(res_0-res_2)), "%.2f"%np.amax(np.abs(res_0-res_2))
    print 'K=4', "%.2f"%np.amin(np.abs(res_0-res_4)), "%.2f"%np.mean(np.abs(res_0-res_4)), "%.2f"%np.amax(np.abs(res_0-res_4))
    print 'K=8', "%.2f"%np.amin(np.abs(res_0-res_8)), "%.2f"%np.mean(np.abs(res_0-res_8)), "%.2f"%np.amax(np.abs(res_0-res_8))
    print 'K=16', "%.2f"%np.amin(np.abs(res_0-res_16)), "%.2f"%np.mean(np.abs(res_0-res_16)), "%.2f"%np.amax(np.abs(res_0-res_16))
    
    #########################
    #   Matern 3/2 Kernel   # 
    #########################
    
    
    #kernel = lambda u, theta: theta[0]*theta[0]*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    #dkerneldx = lambda u, theta: -((theta[0]/theta[1])**2)*u*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    #d2kerneldxdy = lambda u, theta: ((theta[0]/theta[1])**2)*(1.0-(u/theta[1])**2)*np.exp(-0.5*u*u/(theta[1]*theta[1]))
    
    
    ma_32_kernel = lambda u, theta: theta[0]*theta[0]*(1+(np.sqrt(3.0)/theta[1])*np.abs(u))*np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
    ma_32_dkerneldx = lambda u, theta: -3.0*theta[0]*theta[0]/(theta[1]*theta[1])*u*np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
    ma_32_d2kerneldxdy = lambda u, theta: 3.0*theta[0]*theta[0]/(theta[1]*theta[1])*(1.0-(np.sqrt(3.0)/theta[1])*np.abs(u))*\
        np.exp(-(np.sqrt(3.0)/theta[1])*np.abs(u))
    
    res_0 = covMatrix(s_times, s_times, theta, symmetric=True, kernel=ma_32_kernel)
    
    # 2 Strings 
    b_times = np.array([0.0, 0.5, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_32_kernel, ma_32_dkerneldx, ma_32_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_2 = res[np.ix_(evn, evn)]
    
    # 4 Strings 
    b_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_32_kernel, ma_32_dkerneldx, ma_32_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_4 = res[np.ix_(evn, evn)]
    
    # 8 Strings 
    b_times = np.array([i/8.0 for i in range(9)])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_32_kernel, ma_32_dkerneldx, ma_32_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_8 = res[np.ix_(evn, evn)]
    
    # 16 Strings 
    b_times = np.array([i/16.0 for i in range(17)])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_32_kernel, ma_32_dkerneldx, ma_32_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_16 = res[np.ix_(evn, evn)]
    
    X, Y = plt.meshgrid(s_times, s_times)
    
    vmin = min(abs(res_0).min(), abs(res_2).min(), abs(res_4).min(), abs(res_8).min(), abs(res_16).min())
    vmax = max(abs(res_0).max(), abs(res_2).max(), abs(res_4).max(), abs(res_8).max(), abs(res_16).max())
    
    p = axes[3, 0].pcolor(X, Y, res_0, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[3, 0].set_ylabel('MA 3/2', rotation=90)
    axes[3, 0].yaxis.set_ticks(ticks)
    axes[3, 0].xaxis.set_ticks(ticks)
    p = axes[3, 1].pcolor(X, Y, res_2, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[3, 2].pcolor(X, Y, res_4, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[3, 3].pcolor(X, Y, res_8, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[3, 4].pcolor(X, Y, res_16, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)

    print '"""""""""""""""""""""""""""'
    print 'ABSOLUTE DEVIATION MA32'
    print '""""""""""""""""""""""""""""'
    print ''
    print 'K=2', "%.2f"%np.amin(np.abs(res_0-res_2)), "%.2f"%np.mean(np.abs(res_0-res_2)), "%.2f"%np.amax(np.abs(res_0-res_2))
    print 'K=4', "%.2f"%np.amin(np.abs(res_0-res_4)), "%.2f"%np.mean(np.abs(res_0-res_4)), "%.2f"%np.amax(np.abs(res_0-res_4))
    print 'K=8', "%.2f"%np.amin(np.abs(res_0-res_8)), "%.2f"%np.mean(np.abs(res_0-res_8)), "%.2f"%np.amax(np.abs(res_0-res_8))
    print 'K=16', "%.2f"%np.amin(np.abs(res_0-res_16)), "%.2f"%np.mean(np.abs(res_0-res_16)), "%.2f"%np.amax(np.abs(res_0-res_16))
    
    #########################
    #   Matern 5/2 Kernel   # 
    #########################
    
    ma_52_kernel = lambda u, theta: theta[0]*theta[0]*(1.0 + (np.sqrt(5.0)/theta[1])*np.abs(u) +\
        (5.0/(3.0*theta[1]*theta[1]))*u*u)*np.exp(-(np.sqrt(5.0)/theta[1])*np.abs(u))
        
    ma_52_dkerneldx = lambda u, theta: theta[0]*theta[0]*u*(-5.0/(3.0*theta[1]*theta[1])\
        -(5.0*np.sqrt(5.0))/(3.0*theta[1]*theta[1]*theta[1])*np.abs(u))*\
        np.exp(-(np.sqrt(5.0)/theta[1])*np.abs(u))
        
    ma_52_d2kerneldxdy = lambda u, theta: theta[0]*theta[0]*(5.0/(3.0*theta[1]*theta[1]) +\
        5.0/(3.0*theta[1]*theta[1])*(np.sqrt(5.0)/theta[1])*np.abs(u) -25.0/(3.0*theta[1]**4)*u*u)*\
       np.exp(-(np.sqrt(5.0)/theta[1])*np.abs(u))        
    
    res_0 = covMatrix(s_times, s_times, theta, symmetric=True, kernel=ma_52_kernel)
    
    # 2 Strings 
    b_times = np.array([0.0, 0.5, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_52_kernel, ma_52_dkerneldx, ma_52_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_2 = res[np.ix_(evn, evn)]
    
    # 4 Strings 
    b_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_52_kernel, ma_52_dkerneldx, ma_52_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_4 = res[np.ix_(evn, evn)]
    
    # 8 Strings 
    b_times = np.array([i/8.0 for i in range(9)])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_52_kernel, ma_52_dkerneldx, ma_52_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_8 = res[np.ix_(evn, evn)]
    
    # 16 Strings 
    b_times = np.array([i/16.0 for i in range(17)])
    res = string_gp_kernel_cov(s_times, b_times, theta, ma_52_kernel, ma_52_dkerneldx, ma_52_d2kerneldxdy, key= '')
    n = res.shape[0]
    evn = [_ for _ in xrange(n) if _%2 ==0]
    res_16 = res[np.ix_(evn, evn)]
    
    X, Y = plt.meshgrid(s_times, s_times)
    
    vmin = min(abs(res_0).min(), abs(res_2).min(), abs(res_4).min(), abs(res_8).min(), abs(res_16).min())
    vmax = max(abs(res_0).max(), abs(res_2).max(), abs(res_4).max(), abs(res_8).max(), abs(res_16).max())
    
    p = axes[4, 0].pcolor(X, Y, res_0, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    axes[4, 0].set_ylabel('MA 5/2', rotation=90)
    axes[4, 0].yaxis.set_ticks(ticks)
    axes[4, 0].xaxis.set_ticks(ticks)
    p = axes[4, 1].pcolor(X, Y, res_2, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[4, 2].pcolor(X, Y, res_4, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[4, 3].pcolor(X, Y, res_8, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    p = axes[4, 4].pcolor(X, Y, res_16, cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)

    print '"""""""""""""""""""""""""""'
    print 'ABSOLUTE DEVIATION MA52'
    print '""""""""""""""""""""""""""""'
    print ''
    print 'K=2', "%.2f"%np.amin(np.abs(res_0-res_2)), "%.2f"%np.mean(np.abs(res_0-res_2)), "%.2f"%np.amax(np.abs(res_0-res_2))
    print 'K=4', "%.2f"%np.amin(np.abs(res_0-res_4)), "%.2f"%np.mean(np.abs(res_0-res_4)), "%.2f"%np.amax(np.abs(res_0-res_4))
    print 'K=8', "%.2f"%np.amin(np.abs(res_0-res_8)), "%.2f"%np.mean(np.abs(res_0-res_8)), "%.2f"%np.amax(np.abs(res_0-res_8))
    print 'K=16', "%.2f"%np.amin(np.abs(res_0-res_16)), "%.2f"%np.mean(np.abs(res_0-res_16)), "%.2f"%np.amax(np.abs(res_0-res_16))
    
    import matplotlib as mpl
    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    cb = fig.colorbar(p, cax=cax, **kw)
    
    #from matplotlib.backends.backend_pdf import PdfPages
    #pp = PdfPages('uniform_string_gp_kernels.pdf')
    #plt.savefig(pp, format='pdf')
    #pp.close()
    
    plt.savefig('uniform_string_gp_kernels.png')
    plt.savefig('uniform_string_gp_kernels.eps', format='eps')
    print time() -t

#popular_kernels_distortion()


    






