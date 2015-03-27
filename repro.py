# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:50:32 2015

@author: ylkomsamo
"""
import Image
import numpy as np
from string_kernels import StringSM
from GPy.models.gp_kronecker_gaussian_regression import GPKroneckerGaussianRegression

###############################################################################
####                      PREPARING THE DATA                               ####
###############################################################################

fl = 'image.jpg'
im = Image.open(fl)
n, m = im.size
ratio = 1.0/8.0
start = int(n*(0.5-ratio/2.0))
end = int(n*(0.5+ratio/2.0))
# Training and test configurations
training_idx = range(0, start) + range(end, n)
train_grid_x = 1.0*np.array([training_idx]).T/n
train_grid_y = train_grid_x
test_grid_x = 1.0*np.array([range(n)]).T/n
test_grid_y = test_grid_x

bands = im.convert('RGB').split()
training_source = [None, None, None]
training_mats = [None, None, None]
test_source = [None, None, None]

for i in xrange(3):
    im_array = np.asarray(bands[i])
    
    # Crop and save cropped image
    training_im_array = im_array.copy()
    training_im_array[start:end,:] = np.zeros(training_im_array[start:end,:].shape)
    training_im_array[:,start:end] = np.zeros(training_im_array[:,start:end].shape)
    training_source[i] = Image.fromarray(training_im_array)
    
    # Save test image
    test_im_array = im_array.copy()
    test_im_array[np.ix_(training_idx, training_idx)] = np.zeros(\
        test_im_array[np.ix_(training_idx, training_idx)].shape)
    test_source[i] = Image.fromarray(test_im_array)
    
    # Training outputs
    train_mat_y = im_array[np.ix_(training_idx, training_idx)]/255.0
    training_mats[i] = train_mat_y.copy()
    
###############################################################################
####                     DONE PREPARING THE DATA                           ####
###############################################################################




###############################################################################
####                        ACTUAL REPRO                                   ####
###############################################################################

kernel = StringSM(np.array([-0.001, 0.5, 1.001]), 1, constrained_opt=False)
train_mat_y = training_mats[0]      
# Create simple GP Model.  
m = GPKroneckerGaussianRegression(train_grid_x, train_grid_y, train_mat_y,\
    kernel.copy(), kernel.copy(), noise_var=0.0001)

# No need to check m.kern2 as it is the same as m.kern1.
# First let us check the gradients.
print '-------------------------------'
print 'Checking gradient of the kernel'
print '-------------------------------'
m.kern1.checkgrad(1)

# Now let's optimize.
# Every time the machinery hits update_gradients_full(self, dL_dK, X, X2)
#   the values of the parameters are printed as well as the values of the
#   derative of the marginal likelihood with respect to each parameter.
#   Note that the parameter values barely change, regardless of the high 
#   magnitude of the gradient.
m.kern1.verbose = True
m.kern2.verbose = True
m.optimize('scg', messages=0)
