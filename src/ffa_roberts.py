# implementation of EM factor analysis for missing data as in Roberts 2014

# N = observations
# K = size of observation
# P = factors

import numpy as np

Z = np.loadtxt(open("../data/exams_imputed.csv", "rb"), delimiter=",")
N, K = Z.shape
P = 1
iter = 300
tol = 10 ** -6


# parameters
mu = np.expand_dims(np.array([40.82, 51.91, 51.82, 49.32, 46.82]), axis=-1) # initial values from Roberts
lam = np.ones((K, P)) # \lambda
Lam = np.hstack((lam, mu)) # \Lambda = [\lambda, \mu]
Sigma = np.identity(K)

I_p = np.identity(P)

# build Y by computing y_t = H_t*z_t for each observation t
H = [np.identity(K)] * N # TODO: replace with real missingness structure
Y = []
for t in range(N):
    y_t = H[t] @ Z[t]
    Y.append(y_t)

Y = np.array(Y)


lik = 0
LL = []

for i in range(iter):
    # E Step
    # compute R^-1 = (\Lambda\Lambda^T + \Sigma)^-1
    Sigma_inv = np.diag(1 / np.diag(Sigma))
    S_lam = Sigma_inv @ lam
    Q = np.linalg.inv(I_p + lam.T @ S_lam)
    R = Sigma + lam @ lam.T
    R_inv = Sigma_inv - S_lam @ Q @ S_lam.T # Woodbury identity

    Z_cent = Z - mu.T # centered Z (data)
    
    # compute LL
    old_lik = lik
    lik = \
        - 0.5 * N * np.math.log(np.linalg.det(R)) \
        - 0.5 * N * K * np.math.log(2 * np.math.pi) \
        - 0.5 * np.sum(np.diag(R_inv @ Z_cent.T @ Z_cent))

    beta = lam.T @ R_inv
    Ex_z = beta @ Z_cent.T
    Exx_z = I_p - beta @ lam + Ex_z @ Ex_z.T
    # first moment of \tilde{x}|z
    Extilde_z = np.vstack( (Ex_z, np.ones(N)) )
    # variance of \tilde{x}|z
    Vxtilde_z = np.hstack( (I_p - beta @ lam, np.zeros((P, 1))) )
    Vxtilde_z = np.vstack( (Vxtilde_z, np.zeros(P+1)) )
    # second moment of \tilde{x}|z = Var + E\tilde{x}|z ^ 2
    Exxtilde_z = Vxtilde_z + Extilde_z @ Extilde_z.T


    print(f'cycle {i} lik {lik:.15f}\n')
    LL.append(lik)

    # M step
    Lam = Z.T @ Extilde_z.T @ np.linalg.inv(Exxtilde_z)
    Sigma = np.diag(np.diag(Z.T @ Z - Lam @ Extilde_z @ Z))
    mu = np.expand_dims(Lam[:, -1], axis=-1)

    if i <= 2:
        lik_base = lik
    elif lik < old_lik:
        print('VIOLATION')
    elif ((lik - lik_base) < (1 + tol) * (old_lik - lik_base)) or np.isneginf(lik):
        break

print(Lam)
print(Sigma)
print(mu)