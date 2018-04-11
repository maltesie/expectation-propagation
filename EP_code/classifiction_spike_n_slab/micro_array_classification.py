import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import sqrt, pi
from matplotlib import cm
from scipy.stats import norm, sem

# EP example procedure
class EP:
    
    def __init__(self, pprior):
        
        self.pprior = pprior

    def fit(self, X, Y):
        pprior = self.pprior
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        n = X.shape[0]
        d = X.shape[1]

        Y = np.expand_dims(Y, axis=1)
        X = X * np.tile(Y, (1, d))

        s0 = 0
        s1 = 1

        a = np.ones(d)
        b = np.ones(d)
        v = np.full(((n+1), d), np.inf)     # n for the data, added row for spike and slab prior
        m = np.zeros(((n+1), d))
        s = np.ones((d + n + 1))

        mu = np.zeros(d)
        nu = np.repeat((pprior * s1 + (1-pprior) * s0), d)
        p = np.repeat(pprior, d)
        v[n, :] = nu

        muBackup = mu
        nuBackup = nu
        pBackup = p
    
        # Iterate EP algorithm
        for iteration in range(200):
    
            # For each of the likelihood terms
            for i in range(n):
    
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

                divisor = 1/nu - 1/nuOld
                divisor[divisor == 0] = 1e-20
                v[i, :] = 1 / divisor
                m[i, :] = muOld + alpha * (v[i, :] + nuOld) * X[i, :]
                s[i] = np.log(norm.cdf(z)) + 0.5 * (sum(np.log(1 + nuOld * v[i, :]**(-1)))) + (0.5 * sum((m[i, :] - muOld)**2 / (v[i, :] + nuOld)))
    
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
    
            if maxDiff < 1e-5: 
                self.v, self.m, self.a, self.b, self.nu, self.mu, self.p = v, m ,a, b, nu, mu, p
                return True
    
            # Save the values for the next convergence check
            muBackup = mu
            nuBackup = nu
            pBackup = p
        
        return False
            
    def score(self, Xtest, Ytest, normalize_labels):
        P = norm.cdf(np.sum(Xtest * self.mu[:-1].reshape(1,-1), axis=1) / np.sqrt(np.sum(Xtest * self.nu[:-1].reshape(1,-1) * Xtest, axis=1) + 1))
        if normalize_labels:
            labels = np.unique(Ytest)
            P[P>0.5]=labels[0]
            P[P<=0.5]=labels[1]
        else:
            P[P>0.5]=1#labels[0]
            P[P<=0.5]=-1#labels[1]
        score = (P == Ytest).sum()/len(Ytest)
        return score
        
def load_data(dataset, normalize_labels=False):
    print(dataset + ' dataset')
    X = np.loadtxt('./data/' + dataset + '/X.txt')
    Y = np.loadtxt('./data/' + dataset + '/Y.txt')

    # Scale to zero mean and unit s.d
    Xtrain = preprocessing.scale(X)
    if normalize_labels: Ytrain = preprocessing.scale(Y)
    else: Ytrain = Y
    return Xtrain, Ytrain


def artificial_data(n=70, d=2000, relevant_each=25, class_split=35, add_value=0.4, normalize_labels=False):
    print('artificial dataseet')
    X = np.random.rand(n,d)
    X[:class_split,:relevant_each] += add_value
    X[class_split:,-relevant_each:] += add_value
    Y = np.ones(n)
    Y[class_split:] = -1

    # Scale to zero mean and unit s.d
    Xtrain = preprocessing.scale(X)
    if normalize_labels: Ytrain = preprocessing.scale(Y)
    else: Ytrain = Y
    
    return Xtrain, Ytrain


def classification_error(dataset, cross_n=20, normalize_labels=False):
    
    if dataset == 'artificial': X, Y = artificial_data(normalize_labels=normalize_labels)
    else: X, Y = load_data(dataset=dataset, normalize_labels=normalize_labels)
    
    clf = EP(32/X.shape[1])
    scores = []
    
    for i in range(cross_n):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=i)
        if clf.fit(Xtrain, Ytrain):
            scores.append(clf.score(Xtest, Ytest, normalize_labels=normalize_labels))
        else: print(i, 'didnt converge')
    
    return 100*(1-np.array(scores))
    
def heatmap(dataset):
    X, Y = load_data(dataset)
    clf = EP(32/X.shape[1]) 
    clf.fit(X, Y)
    P_ind = np.argsort(clf.p[:-1])[-16:]
    heatmap = X[:,P_ind].T
    Y_ind = np.nonzero(Y==1)[0]
    h_ind = np.argsort(np.mean(heatmap[:,Y_ind], axis=1))
    heatmap = heatmap[h_ind]
    data = np.hstack((heatmap[:,Y==1],heatmap[:,Y==-1]))
    fig, ax = plt.subplots()
    cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm, aspect='auto')
    #ax.set_title('expression level', fontsize=20)
    ax.set_xlabel('patients', fontsize=20)
    ax.set_ylabel('genes', fontsize=20)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=16)
    fig.savefig(dataset + '.png', dpi = 400)
    plt.show()


heatmap('leukaemia')

for dataset in ['adenocarcinoma', 'brain_A', 'brain_B', 'brain_C', 'breastER_standard', 'breastLN_standard', 'colon', 'down_syndrome', 'leukaemia', 'lymphoma', 'metastasis', 'mutation', 'ovarian', 'srbct']:
    scores = classification_error(dataset)
    print("{0} $\pm$ {1}".format(str(round(np.mean(scores),2)), str(round(np.std(scores),2))))


