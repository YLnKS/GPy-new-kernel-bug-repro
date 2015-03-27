# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:25:17 2015

@author: ylkomsamo
"""
import numpy as np
from string_gp_utilities import string_gp_kernel_cov
from numpy.core.umath_tests import inner1d
from GPy.kern import Kern
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp
import sys

'''
Generic kernel for String GP indexed on undimensional input spaces.
'''
class StringKern(Kern):
    def __init__(self, b_times, thetas, uncond_kernel, uncond_dkerneldx,\
            uncond_d2kerneldxdy, name='StringSpectMixt'):
        super(StringKern, self).__init__(1, None, name)
        
        self.b_times = np.unique(b_times) # Boundary times (uniquely sorted)
        # Number of strings
        self.n_strings = len(self.b_times)-1
        self.uncond_kernel = uncond_kernel #  Unconditional string kernel
        self.uncond_dkerneldx = uncond_dkerneldx
        self.uncond_d2kerneldxdy = uncond_d2kerneldxdy
        
        # String hyper-parameters
        #   This class SHOULD NOT be implemented directly. The constructor of 
        #   any derived class should take care of initialising the string hyper-
        #   parameters at random and appropriately, and linking them to the model 
        #   object.
        
            
    def K(self, X, X2):
        """
        Compute the kernel function.

        :param X: the first set of inputs to the kernel
        :param X2: (optional) the second set of arguments to the kernel. If X2
                   is None, this is passed throgh to the 'part' object, which
                   handLes this as X2 == X.
        """ 
        if X2 == None:
            X2 = X
            s_times = np.array(sorted(np.ndarray.flatten(X)))
            ind_x = np.searchsorted(s_times, np.ndarray.flatten(X))
            ind_x2 = ind_x
        else:
            s_times = np.array(sorted(set(list(np.ndarray.flatten(X)) + list(np.ndarray.flatten(X2)))))
            ind_x = np.searchsorted(s_times, np.ndarray.flatten(X))
            ind_x2 = np.searchsorted(s_times, np.ndarray.flatten(X2))
        
        
        # Covariance matrix the derivative string GP
        deriv_sgp_cov = string_gp_kernel_cov(s_times, self.b_times, None, self.uncond_kernel,\
            self.uncond_dkerneldx, self.uncond_d2kerneldxdy, thetas=self._get_thetas())
        
        evn = [_ for _ in xrange(deriv_sgp_cov.shape[0]) if _%2==0]
        sgp_cov = deriv_sgp_cov[np.ix_(evn, evn)]
        
        return sgp_cov[np.ix_(ind_x, ind_x2)]
        
    '''
    Method that uses the linked parameters to form a matrix of unconditional
        string hyper-parameters. There should be as many rows as strings.
    '''
    def _get_thetas(self):
        # Depends on the type of string. 
        #   Should be implemented in the derived class.
        raise NotImplementedError
        
    '''
    Diagonal of the covariance matrix
    '''
    def Kdiag(self, X):
        return np.diag(self.K(X, X))
        
        
    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Given the derivative of the objective with respect to the diagonal of
        the covariance matrix, compute the derivative wrt the parameters of
        this kernel and stor in the <parameter>.gradient field.

        See also update_gradients_full
        """
        # Depends on the type of string. 
        #   Should be implemented in the derived class.
        raise NotImplementedError
        
    def update_gradients_full(self, dL_dK, X, X2):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        # Depends on the type of string. 
        #   Should be implemented in the derived class.
        raise NotImplementedError
        
        
'''
Some utility functions
'''
def sm_kernel(u, theta):
    return sum([theta[3*i]*theta[3*i]*np.exp(-2.0*np.pi*np.pi*(u/theta[1+3*i])**2)*\
        np.cos(2.0*np.pi*u*theta[2+3*i]) for i in xrange(len(theta)/3)])  
        
AFactor = lambda u, params: (-2.0*np.pi*params[2]*np.sin(2.0*np.pi*u*params[2])
    -((2.0*np.pi/params[1])**2)*u*np.cos(2.0*np.pi*u*params[2]))
    
def dsm_kernel_dx(u, theta):
    return sum([theta[3*i]*theta[3*i]*np.exp(-2.0*np.pi*np.pi*(u/theta[1+3*i])**2)*AFactor(u, [theta[3*i], theta[1+3*i], theta[2+3*i]])  for i in xrange(len(theta)/3)]) 

dAFactordy = lambda u, params: ((2.0*np.pi*params[2])**2 + 4.0*np.pi/(params[1]*params[1]))*\
    np.cos(2.0*np.pi*u*params[2])-2.0*np.pi*params[2]*((2.0*np.pi/params[1])**2)*u*np.sin(2.0*np.pi*u*params[2])

def dsm_kernl_dxdy(u, theta):
    return sum([theta[3*i]*theta[3*i]*np.exp(-2.0*np.pi*np.pi*(u/theta[1+3*i])**2)*\
        (-((2.0*np.pi/theta[1+3*i])**2)*u*AFactor(u, [theta[3*i], theta[1+3*i], theta[2+3*i]]) 
            + dAFactordy(u, [theta[3*i], theta[1+3*i], theta[2+3*i]])) for i in xrange(len(theta)/3)]) 

'''
String Spectral Mixture kernel
'''
class StringSM(StringKern):
    def __init__(self, b_times, n_mixture_elts, thetas=None, name='StringSM', constrained_opt=False, verbose=False):
        super(StringSM, self).__init__(b_times, thetas, sm_kernel, dsm_kernel_dx,\
            dsm_kernl_dxdy, name=name)
            
        self.verbose = verbose
        self.constrained_opt=constrained_opt
        self.n_mixture_elts = n_mixture_elts
                
        # Initialise the parameters and link them to the model.
        for i in xrange(self.n_strings):
            if thetas == None:
                random_init_freq = np.sort(np.random.uniform(0.0, 128.0, n_mixture_elts))
                mixture_params = np.array([1.0]*(3*n_mixture_elts))
                # Initialise the standard deviation uniformly at random.
                mixture_params[0::3] = np.random.uniform(1.0/255.0, 1.0, n_mixture_elts)**2
                # Initialise the lengthscale uniformly at random.
                mixture_params[1::3] = np.random.uniform(1.0/510.0, 1.0, n_mixture_elts)
                # When constrained_opt is set to true, we enforce an order in the frequencies
                #   by optimizing over positive frequency increments.
                #   i.e. \mu_0 = d\mu_0, ..., \mu_k = \mu_{k-1} + d\mu_k, where d\mu_k > 0.
                if not self.constrained_opt:
                    mixture_params[2::3] = random_init_freq.copy()
                else:
                    mixture_params[2::3] = np.array([random_init_freq[0]]+ list(random_init_freq[1:]-random_init_freq[:-1]))
                    
            else:
                # When 'thetas' is provided, the i-th row should contain the hyper-paramters of the i-th string.
                mixture_params = thetas[i,:].copy()
            
                    
            # Check the number of elements in the mixture
            assert len(mixture_params)%3 == 0, "Each element in the mixture should have 3 parameters."
            assert len(mixture_params) == 3*n_mixture_elts, "The number of mixture parameters should be 3 times the number of mixture elements"
        
            params = []
            for j in xrange(self.n_mixture_elts):
                # Weight of the i-th element in the mixture
                setattr(self, 'variance_' + str(i) + '_' + str(j), Param('variance_' + str(i) + '_' + str(j), mixture_params[3*j], Logexp()))
                params += [getattr(self, 'variance_' + str(i) + '_' + str(j))]
                
                # lengthscale of the i-th element in the mixture
                setattr(self, 'lengthscale_' + str(i) + '_' + str(j), Param('lengthscale_' + str(i)  + '_' + str(j), mixture_params[3*j+1], Logexp()))
                params += [getattr(self, 'lengthscale_' + str(i) + '_' + str(j))]
                
                if not self.constrained_opt:
                    # Frequency of the i-th element in the mixture
                    setattr(self, 'frequency_' + str(i) + '_' + str(j), Param('frequency_' + str(i)  + '_' + str(j), mixture_params[3*j+2], Logexp()))
                    params += [getattr(self, 'frequency_' + str(i) + '_' + str(j))]
                else:
                    # Frequency increment of the i-th element in the mixture
                    setattr(self, 'frequency_inc_' + str(i) + '_' + str(j), Param('frequency_inc_' + str(i) + '_' + str(j), mixture_params[3*j+2], Logexp()))
                    params += [getattr(self, 'frequency_inc_' + str(i) + '_' + str(j))]
                    
            self.link_parameters(*params)
        
    
    '''
    Method that uses the linked parameters to form a matrix of unconditional
        string hyper-parameters. There should be as many rows as strings.
    '''
    def _get_thetas(self):
        thetas = np.empty((self.n_strings, 3*self.n_mixture_elts))
        for i in xrange(self.n_strings):
            str_theta = np.array([0.0]*(3*self.n_mixture_elts))
            str_theta[0::3] = np.array([np.sqrt(getattr(self, 'variance_' + str(i) + '_' + str(j))[0])\
                for j in xrange(self.n_mixture_elts)])
            str_theta[1::3] = np.array([getattr(self, 'lengthscale_' + str(i) + '_' + str(j))[0]\
                for j in xrange(self.n_mixture_elts)])
            if self.constrained_opt:
                str_theta[2::3] = np.cumsum([getattr(self, 'frequency_inc_' + str(i) + '_' + str(j))[0]\
                    for j in xrange(self.n_mixture_elts)])
            else:
                str_theta[2::3] = np.array([getattr(self, 'frequency_' + str(i) + '_' + str(j))[0]\
                    for j in xrange(self.n_mixture_elts)])
            thetas[i,:] = str_theta.copy()
            
        return thetas
        
    
        
    def update_gradients_full(self, dL_dK, X, X2):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        h = 1.0e-6
        if X2 == None:
            X2 = X
            s_times = np.array(sorted(np.ndarray.flatten(X)))
            ind_x = np.searchsorted(s_times, np.ndarray.flatten(X))
            ind_x2 = ind_x
        else:
            s_times = np.array(sorted(set(list(np.ndarray.flatten(X)) + list(np.ndarray.flatten(X2)))))
            ind_x = np.searchsorted(s_times, np.ndarray.flatten(X))
            ind_x2 = np.searchsorted(s_times, np.ndarray.flatten(X2))
        
        current_thetas = self._get_thetas()
        current_cov = self.K(X, X2)
        evn = [_ for _ in xrange(2*len(s_times)) if _%2==0]
        if self.verbose:
            print '------------------------'
            print 'In update_gradients_full'
            print 'Parameters --- Variance: ', current_thetas[0,0], 'Lengthscale: ', current_thetas[0,1], 'Frequency: ', current_thetas[0,2]
            sys.stdout.flush()

        for i in xrange(self.n_strings):
            for j in xrange(self.n_mixture_elts):
                # Derivative of the covariance matrix with respect to the variance of the j-th 
                #   mixture component of the i-th string.
                thetas_variance_h = current_thetas.copy()
                thetas_variance_h[i,3*j] = np.sqrt(thetas_variance_h[i,3*j]**2 + h)
                deriv_sgp_cov_var_h = string_gp_kernel_cov(s_times, self.b_times, None, self.uncond_kernel,\
                    self.uncond_dkerneldx, self.uncond_d2kerneldxdy, thetas=thetas_variance_h)
                sgp_cov_var_h = deriv_sgp_cov_var_h[np.ix_(evn, evn)]
                cov_var_h = sgp_cov_var_h[np.ix_(ind_x, ind_x2)]         
                
                dcov_dvar = (1.0/h)*(cov_var_h-current_cov)
                setattr(getattr(self, 'variance_' + str(i) + '_' + str(j)), 'gradient', \
                    np.sum(inner1d(dL_dK, dcov_dvar.T)))
                    
                if self.verbose:
                    print 'dObjective_dvariance_' + str(i) + '_' + str(j), ':    ',  getattr(self, 'variance_' + str(i) + '_' + str(j)).gradient[0]
                
                # Derivative of the covariance matrix with respect to the lengthscale of the j-th 
                #   mixture component of the i-th string.
                thetas_len_h = current_thetas.copy()
                thetas_len_h[i,1+3*j] = thetas_len_h[i,1+3*j]+h
                deriv_sgp_cov_len_h = string_gp_kernel_cov(s_times, self.b_times, None, self.uncond_kernel,\
                    self.uncond_dkerneldx, self.uncond_d2kerneldxdy, thetas=thetas_len_h)
                sgp_cov_len_h = deriv_sgp_cov_len_h[np.ix_(evn, evn)]
                cov_len_h = sgp_cov_len_h[np.ix_(ind_x, ind_x2)]
                
                dcov_dlen = (1.0/h)*(cov_len_h-current_cov)
                setattr(getattr(self, 'lengthscale_' + str(i) + '_' + str(j)), 'gradient', \
                    np.sum(inner1d(dL_dK, dcov_dlen.T)))
                    
                if self.verbose:
                    print 'dObjective_dlengthscale_' + str(i) + '_' + str(j), ':    ',  getattr(self, 'lengthscale_' + str(i) + '_' + str(j)).gradient[0]
                    
                if not self.constrained_opt:
                    # Derivative of the covariance matrix with respect to the frequency of the j-th 
                    #   mixture component of the i-th string.
                    thetas_freq_h = current_thetas.copy()
                    thetas_freq_h[i,2+3*j] = thetas_freq_h[i,2+3*j]+h
                    deriv_sgp_cov_freq_h = string_gp_kernel_cov(s_times, self.b_times, None, self.uncond_kernel,\
                        self.uncond_dkerneldx, self.uncond_d2kerneldxdy, thetas=thetas_freq_h)
                    sgp_cov_freq_h = deriv_sgp_cov_freq_h[np.ix_(evn, evn)]
                    cov_freq_h = sgp_cov_freq_h[np.ix_(ind_x, ind_x2)]
                    
                    dcov_dfreq = (1.0/h)*(cov_freq_h-current_cov)
                    setattr(getattr(self, 'frequency_' + str(i) + '_' + str(j)), 'gradient', \
                        np.sum(inner1d(dL_dK, dcov_dfreq.T)))
                    
                    if self.verbose:
                        print 'dObjective_dfrequency_' + str(i) + '_' + str(j), ':    ', getattr(self, 'frequency_' + str(i) + '_' + str(j)).gradient[0]
                        
                else:
                    # Derivative of the covariance matrix with respect to the j-th frequency increment in the  
                    #   spectral mixture of the i-th string.
                    thetas_freq_inc_h = current_thetas.copy()
                    thetas_freq_inc_h[i,2+3*j::3] = thetas_freq_inc_h[i,2+3*j::3]+h
                    deriv_sgp_cov_freq_inc_h = string_gp_kernel_cov(s_times, self.b_times, None, self.uncond_kernel,\
                        self.uncond_dkerneldx, self.uncond_d2kerneldxdy, thetas=thetas_freq_inc_h)
                    sgp_cov_freq_inc_h = deriv_sgp_cov_freq_inc_h[np.ix_(evn, evn)]
                    cov_freq_inc_h = sgp_cov_freq_inc_h[np.ix_(ind_x, ind_x2)]
                    
                    dcov_dfreq_inc = (1.0/h)*(cov_freq_inc_h-current_cov)
                    setattr(getattr(self, 'frequency_inc_' + str(i) + '_' + str(j)), 'gradient', \
                        np.sum(inner1d(dL_dK, dcov_dfreq_inc.T)))
                        
                    if self.verbose:
                        print 'dObjective_dfrequency_increment_' + str(i) + '_' + str(j), ':    ', getattr(self, 'frequency_inc_' + str(i) + '_' + str(j)).gradient[0]
        
        return
                    
                