# classify_spike_n_slab.py
#
# Implementation of Expectation Propagation with Spike and Slab priors for
# classification. As described in:
# Expectation Propagation for Microarray Data Classification
# Daniel Hernandez Lobato, Jose Miguel Hernandez Lobato, and Alberto Suarez
# Pattern Recognition Letters, 31, pp. 1618-1626, 2010.
#
# The focus here is on a 'Toy' problem to illustrate the method. The goal is to
# use EP to estimate the posterior for the coefficient vector w in the linear
# regression problem. For this artificial situation we know the true w has
# in-fact been sampled from a spike-and-slab prior distribution and therefore
# the use of a such a prior is appropriate.
#
# Author: Alan Aberdeen

# Import libs
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from math import sqrt, pi
from scipy.stats import norm


# EP example procedure
def ep(X, Y, pprior):

    # Inputs:
    #   X:      Matrix of gene expression data.
    #   Y:      Binary labels in 1, -1 codification
    #   pprior: Prior parameter for the probability of an attribute taking part
    #           in the classification.
    #
    # Outputs:
    #   out:    An object with the following entries
    #       mean:   Mean of the Gaussian approximation for the parameters of the
    #               separating hyperplane.
    #       var:    Covariance matrix of the Gaussian approximation for the
    #               parameters of the separating hyperplane.
    #       p:      Vector with the probability of each attribute taking part in
    #               the classification.

    # Add a constant value of 1 to guarantee that the separating hyperplane goes
    # through the origin.
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Store number of training instances and attributes
    n = X.shape[0]
    d = X.shape[1]

    # Scale the attributes of each instance by Y
    # *** Not perfectly sure why we do this? ***
    Y = np.expand_dims(Y, axis=1)
    X = X * np.tile(Y, (1, d))

    # Initialise the variances of the spike and the slab
    s0 = 0
    s1 = 1

    # Initialise our current approximation to be equal to that of the prior
    # Fist set all the terms that in involved in the calculation of each factor
    a = np.ones(d)
    b = np.ones(d)
    v = np.full(((n+1), d), np.inf)     # n for the data, added row for spike and slab prior
    m = np.zeros(((n+1), d))
    s = np.ones((d + n + 1))

    # Initialise approximation of params that define the posterior
    mu = np.zeros(d)
    nu = np.repeat((pprior * s1 + (1-pprior) * s0), d)
    p = np.repeat(pprior, d)
    v[n, :] = nu

    # Initialise convergence flag and store original params to check conv point
    muBackup = mu
    nuBackup = nu
    pBackup = p
    converged = False

    # Iterate EP algorithm
    for iteration in range(100):

        if converged:
            break

        # For each of the likelihood terms
        for i in range(n):

            # Save the old posterior distribution to variables but check for
            # negative variance issues and skip this iteration if found.
            nuOld = 1/((1/nu) - 1/v[i, :])
            if nuOld.any() < 0:
                break
            muOld = mu + (nuOld * 1/v[i, :] * (mu - m[i, :]))

            # Compute axillary variables used in the update for posterior terms
            xiVwxi = np.sum(X[i, :]**2 * nuOld)
            muOldxi = np.sum(muOld * X[i, :])
            z = muOldxi / sqrt(xiVwxi + 1)
            alpha = (1 / (sqrt(xiVwxi + 1))) * norm.pdf(z) / norm.cdf(z)

            # Update posterior params
            mu = muOld + alpha * nuOld * X[i, :]
            nu = nuOld - alpha * (sum(X[i, :] * mu) + alpha) / (xiVwxi + 1) * nuOld**2 * X[i, :]**2

            # Compute and store the new likelihood approximation
            # Be careful with dividing by zero. Here I hacked around the issue
            # by simply replacing those zero values with something very small
            # but there are almost certainly more clever ways of doing it.
            divisor = 1/nu - 1/nuOld
            divisor[divisor == 0] = 1e-20
            v[i, :] = 1 / divisor
            m[i, :] = muOld + alpha * (v[i, :] + nuOld) * X[i, :]
            s[i] = np.log(norm.cdf(z)) + 0.5 * (sum(np.log(1 + nuOld * v[i, :]**(-1)))) + (0.5 * sum((m[i, :] - muOld)**2 / (v[i, :] + nuOld)))

        # Process spike and slab prior terms
        # TODO Again be careful for negative variances.
        # I believe this is the heads up that we were given in the email
        # about equation 42 in the paper being misleading. Can just set
        # pOld = pprior as below instead...
        pOld = pprior
        nuOld = 1 / (1/nu - 1/v[n, :])
        muOld = mu + nuOld * 1/v[n, :] * (mu - m[n, :])

        # Compute necessary intermediary constants
        G1 = norm.pdf(0, muOld, np.sqrt(nuOld + s1))
        G0 = norm.pdf(0, muOld, np.sqrt(nuOld + s0))

        Z = pOld*G1 + (1 - pOld)*G0

        c1 = 1/Z * (pOld * G1 * -muOld/(nuOld + s1) + (1 - pOld)*G0 * -muOld/(nuOld + s0))
        c2 = 1/2 * 1/Z * (pOld * G1 * ((muOld**2 / ((nuOld + s1)**2)) - (1 / (nuOld + s1))) +
                         (1-pOld) * G0 * ((muOld**2 / ((nuOld + s0)**2)) - (1 / (nuOld + s0))))
        c3 = c1**2 - 2*c2

        # Compute new posterior params
        nu = nuOld - (c3 * (nuOld**2))
        mu = muOld + c1*nuOld
        p = pOld * G1 / (pOld*G1 + ((1-pOld)*G0))

        # Update terms approximations
        v[n] = 1/c3 - nuOld
        m[n] = muOld + c1*(v[n] + nuOld)
        a = p / pOld
        b = (1 - p) / (1 - pOld)

        # Scale factor s
        s[n: (n + d)] = np.log(Z) + 0.5 * np.log(1 + nuOld * 1/v[n]) + 0.5 * c1**2 / c3

        # Check for convergence
        maxDiff = max(max(abs(nu - nuBackup)), max(abs(mu - muBackup)), max(abs(p - pBackup)))
        print(maxDiff)

        if maxDiff < 1e-5:
            converged = True

        # Save the values for the next convergence check
        muBackup = mu
        nuBackup = nu
        pBackup = p


# Generate training and test data splits
def generate_data(Xpath, Ypath):

    # Inputs:
    #   Xpath:  Path to .txt file where the attributes are stored
    #   Ypath:  Path to .txt file where the labels are stored, (-1 and 1 values)
    #
    # Outputs:
    #   train:  (Xtrain, Ytrain) A random split of 2/3 of the data, with zero
    #           mean and unit standard deviation.
    #   test:   (Xtest, Ytest) A random split of the alt corresponding 1/3 data

    # Load data from text files
    X = np.loadtxt(Xpath)
    Y = np.loadtxt(Ypath)

    # Generate specific train and test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=0)

    # Scale to zero mean and unit s.d
    Xtrain = preprocessing.scale(Xtrain)
    Ytrain = preprocessing.scale(Ytrain)

    # Combine sets
    train = (Xtrain, Ytrain)
    test = (Xtest, Ytest)

    return train, test


# Run the required functionality for simulation.
def main():

    # Current working directory
    cwd = os.getcwd()

    # Generate data splits
    #Xpath = cwd + '/data/adenocarcinoma/X.txt'
    #Ypath = cwd + '/data/adenocarcinoma/Y.txt'
    #train, test = generate_data(Xpath, Ypath)

    # For testing purposes utilise the same splits that the paper provides.
    X = np.loadtxt(cwd + '/data/adenocarcinoma/train_X.csv', delimiter=",")
    Y = np.loadtxt(cwd + '/data/adenocarcinoma/train_Y.csv', delimiter=",")
    pprior = 32 / X.shape[1]

    # Run ep
    ep(X, Y, pprior)


main()
