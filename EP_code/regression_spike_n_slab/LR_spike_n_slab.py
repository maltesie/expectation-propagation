# spike_n_slab_LR.py
#
# Implementation of Expectation Propagation with Spike and Slab priors for
# linear regression. As described in:
# Expectation propagation in linear regression models with spike-and-slab
# priors, José Miguel Hernández-Lobato, Daniel Hernández-Lobato and
# Alberto Suárez, 2014.
#
# This simple example should be run on a train / test partition of the
# adenocarcinoma data set.
#
# Author: Alan Aberdeen

# Import libs
import numpy as np
import os


# Run the EP example procedure
def main():

    # Current working directory
    cwd = os.getcwd()

    # Set the configuration variables
    X = np.loadtxt(cwd + '/data/Xtrain1.txt')   # Design matrix for the regression problem.
    Y = np.loadtxt(cwd + '/data/Ytrain1.txt')   # Target vector for the regression problem.
    beta = 1                                    # Noise precision.
    p0 = 0.5                                    # Prior probability that a feature is relevant for solving the regression problem.
    v = 1                                       # Variance of the slab.

    # Save dimensions of design matrix
    n = X.shape[0]
    d = X.shape[1]

    # Pre-compute the
    #tXX = np.multiply(np.transpose(X), X)
    #tXY = np.multiply(np.transpose(X), Y)

    # Initialise the approximations to be non-informative
    # First factor, of the form (mHat, vHat) tuple
    f1hat = (np.full((1, d), 0), np.full((1, d), np.infty))

    # Second factor, of the form (mHat, vHat, phiHat)
    f2hat = (np.full((1, d), 0), np.full((1, d), np.infty), np.full((1, d), 0))

    # Third factor, of form (phiHat)
    f3hat = np.full((1, d), 0)

    # Initial free parameter approximations
    m = np.full((1, d), 0)
    v = np.full((1, d), np.infty)
    phi = np.full((1, d), 0)
    p = np.full((1, d), np.nan)

    # For ease of use, save an object with each of the parameters as
    # an attribute
    params = {
        'f1hat': f1hat,
        'f2hat': f2hat,
        'f3hat': f3hat,
        'm': m,
        'v': v,
        'phi': phi,
        'p': p
    }

    # Initialise convergence flag
    converged = False

    # Initialise damping factor. This helps to avoid the EP algorithm
    # oscillating without ever converging. The original EP update operation
    # (without damping) is recovered in the limit damping = 1. For damping = 0,
    # the approximate factor would not be adjusted in each EP step.
    damping = 0.99

    # Iterate EP algorithm
    for iteration in range(1000):

        if converged:
            break

        # Retain a copy of the old parameters
        paramsOld = params

        # Refine the approximation for the likelihoods

    alskjdf = 1


main()
