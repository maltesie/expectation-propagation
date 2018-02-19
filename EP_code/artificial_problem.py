import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.stats import norm
from matplotlib import cm

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
            return v, m ,a, b

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
        #print(maxDiff, maxDiff < 1e-5)

        if maxDiff < 1e-5: return v, m ,a, b, nu, mu, p

        # Save the values for the next convergence check
        muBackup = mu
        nuBackup = nu
        pBackup = p
        
        
def artificial_data(n, d, relevant_each, class_split, add_value):
    X = np.random.rand(n,d)
    X[:class_split,:relevant_each] += add_value
    X[class_split:,-relevant_each:] += add_value
    Y = np.ones(n)
    Y[:class_split] = -1
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=0)

    # Scale to zero mean and unit s.d
    Xtrain = preprocessing.scale(Xtrain)
    Ytrain = preprocessing.scale(Ytrain)

    return Xtrain, Ytrain

def run_ep(n, d, class_split, relevant_per_class, add_relevant, plot_posterior=True):

    
    X, Y = artificial_data(n, d, relevant_per_class, class_split, add_relevant)
    pprior = 40 / X.shape[1]
    try: v, m, a, b, nu, mu, p = ep(X, Y, pprior)
    except: return False
    print(m[-1].sum(), v[-1].sum())
    index_class_1 = np.nonzero(m[-1]>m[-1].mean()+3*m[-1].std())[0]
    index_class_2 = np.nonzero(m[-1]<m[-1].mean()-3*m[-1].std())[0]
    
    check_index_1 = np.array([i for i in range(relevant_per_class)])
    check_index_2 = np.array([d-i for i in range(relevant_per_class,0,-1)])
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('nu')
    fig.set_size_inches(8.5, 4.5)
    ax1.plot(p, 'r-', markersize=5)
    
    try:
        if (index_class_1 == check_index_1).all() and (index_class_2 == check_index_2).all():
            return True
        else: return False
    except:
        return False

def create_heatmap(nb_runs):
    n = 50
    d = 1000
    class_split = 25

    rels = [1, 2, 3, 5, 10, 15]
    adds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    heatmap = np.zeros((len(rels),len(adds)))
    for i,relevant_per_class in enumerate(rels):
        for j,add_relevant in enumerate(adds):
            for _ in range(nb_runs):
                success = run_ep(n, d, class_split, relevant_per_class, add_relevant, False)
                if success: heatmap[i,j]+=1.
                
    heatmap/=nb_runs
    
    fig, ax = plt.subplots()
    data = heatmap
    cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title('success rate')
    ax.set_xlabel('value added to noise')
    ax.set_ylabel('relevant dimensions')
    ax.set_xticklabels([0]+adds)
    ax.set_yticklabels([0]+rels)
    fig.colorbar(cax, ticks=[1/nb_runs*i for i in range(nb_runs+1)])
    plt.show()
    
#create_heatmap(5)

run_ep(50, 1000, 25, 2, 1.5)
