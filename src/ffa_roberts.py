# implementation of EM factor analysis for missing data as in Roberts 2014

# N = observations
# K = size of observation

import numpy as np

Z = np.loadtxt(open("../data/exams_imputed.csv", "rb"), delimiter=",")
N, K = Z.shape
P = 2
iter = 100
tol = 0.0001

# X = X - np.ones((N, 1)) * np.mean(X, axis=0)
mu = np.mean(Z, axis=0)
ZZ = np.matmul(Z.T, Z) / N
diagZZ = np.diag(ZZ)

covZ = np.cov(Z.T)
scale = np.linalg.det(covZ) ** (1 / K)
Lambda = np.random.randn(K, P) * np.math.sqrt(scale / P)
Sigma = np.diag(covZ)

# build Y by computing y_t = H_t*z_t for each observation t
H = [np.identity(K)] * N # TODO: replace with real missingness structure
Y = []
for t in range(N):
    # y_t = np.matmul(H[t], Z[t])
    y_t = H[t] @ Z[t]
    Y.append(y_t)

Y = np.array(Y)

I = np.identity(P)

lik = 0
LL = []
const = -K / 2 * np.math.log(2 * np.math.pi)

for i in range(iter):
    # E Step
    # R_det = np.diag(1 / R)
    # LR_det = np.matmul(R_det, Lambda)
    # left_mat = np.matmul(LR_det, np.linalg.inv(I + np.matmul(Lambda.T, LR_det)))
    # MM = R_det - np.matmul(left_mat, LR_det.T)
    R = Lambda @ Lambda.T + np.diag(Sigma)
    Lambda_R = Lambda.T @ np.linalg.inv(R)
    EX = Lambda_R @ (Y - np.ones((N, 1)) * mu).T
    EXtilde = np.vstack([EX, np.ones(N)])
    EXtilde_Z = EXtilde @ Z

    EXX = I - Lambda_R @ Lambda
    # add EXX as square submatrix of EXXtilde
    EXXtilde = np.hstack([EXX, np.zeros((P, 1))])
    EXXtilde = np.vstack([EXXtilde, np.zeros(P+1)])
    # dM = np.math.sqrt(np.linalg.det(MM))
    # beta = np.matmul(Lambda.T, MM)
    # XXbeta = np.matmul(ZZ, beta.T)
    # EZZ = I - np.matmul(beta, Lambda) + np.matmul(beta, XXbeta)

    MM = np.linalg.inv(R)
    dM = np.math.sqrt(np.linalg.det(MM))

    # compute LL
    old_lik = lik
    lik = N * const + N * np.math.log(dM) - 0.5 * N * np.sum(np.diag(np.matmul(MM, ZZ)))

    print(f'cycle {i} lik {lik}\n')
    LL.append(lik)

    # M step
    Lambda = np.matmul(EXtilde_Z, np.linalg.inv(EXXtilde))
    R = diagZZ - np.diag(np.matmul(Lambda, EXtilde_Z))

    if i <= 2:
        lik_base = lik
    elif lik < old_lik:
        print('VIOLATION')
    elif ((lik - lik_base) < (1 + tol) * (old_lik - lik_base)) or np.isneginf(lik):
        break