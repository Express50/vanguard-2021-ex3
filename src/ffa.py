# implementation of EM factor analysis as in Ghahramani & Hinton Matlab version

import numpy as np

X = np.loadtxt(open("../data/exams_imputed.csv", "rb"), delimiter=",")
N, D = X.shape
K = 2
iter = 100
tol = 0.0001

X = X - np.ones((N, 1)) * np.mean(X, axis=0)
XX = np.matmul(X.T, X) / N
diagXX = np.diag(XX)

covX = np.cov(X.T)
scale = np.linalg.det(covX) ** (1 / D)
Lambda = np.random.randn(D, K) * np.math.sqrt(scale / K)
Phi = np.diag(covX)

I = np.identity(K)

lik = 0
LL = []
const = -D / 2 * np.math.log(2 * np.math.pi)

for i in range(iter):
    # E Step
    Phid = np.diag(1 / Phi)
    Lambda_Phid = np.matmul(Phid, Lambda)
    left_mat = np.matmul(Lambda_Phid, np.linalg.inv(I + np.matmul(Lambda.T, Lambda_Phid)))
    MM = Phid - np.matmul(left_mat, Lambda_Phid.T)
    dM = np.math.sqrt(np.linalg.det(MM))
    beta = np.matmul(Lambda.T, MM)
    XXbeta = np.matmul(XX, beta.T)
    EZZ = I - np.matmul(beta, Lambda) + np.matmul(beta, XXbeta)

    # compute LL
    old_lik = lik
    lik = N * const + N * np.math.log(dM) - 0.5 * N * np.sum(np.diag(np.matmul(MM, XX)))
    print(f'cycle {i} lik {lik}\n')
    LL.append(lik)

    # M step
    Lambda = np.matmul(XXbeta, np.linalg.inv(EZZ))
    Phi = diagXX - np.diag(np.matmul(Lambda, XXbeta.T))

    if i <= 2:
        lik_base = lik
    elif lik < old_lik:
        print('VIOLATION')
    elif ((lik - lik_base) < (1 + tol) * (old_lik - lik_base)) or np.isneginf(lik):
        break