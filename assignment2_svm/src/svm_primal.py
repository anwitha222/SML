import numpy as np
from cvxopt import matrix, spmatrix, sparse, solvers

class SVM_PrimalLinearQP:

    def __init__(self, C=1.0, tol=1e-9):
        self.C = float(C)
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        #X: (n, d), y: (n,) with {0,1} or {-1,+1}
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y = np.where(y == 0, -1.0, 1.0)

        n, d = X.shape
        m = d + 1 + n  # vars: w (d), b (1), xi (n)

        #objective: (1/2) z^T P z + q^T z
        #P: identity on w-block only (d x d)
        P = spmatrix(1.0, list(range(d)), list(range(d)), (m, m))
        #q: zeros for w and b, C for each xi
        q = np.zeros(m, dtype=np.float64)
        q[d + 1:] = self.C
        q = matrix(q, tc='d')

        #inequalities: G z <= h
        #Margin constraints: -y_i*(w^T x_i + b) - xi_i <= -1
        #build G1 (n x m) explicitly as a sparse matrix.
        #w-block entries
        rows_aw = np.repeat(np.arange(n), d)
        cols_aw = np.tile(np.arange(d), n)
        vals_aw = (-y[:, None] * X).ravel(order='C').astype(np.double)

        #b column
        rows_ab = np.arange(n)
        cols_ab = np.full(n, d, dtype=int)
        vals_ab = (-y).astype(np.double)

        #xi block (diagonal starting at column d+1)
        rows_axi = np.arange(n)
        cols_axi = d + 1 + np.arange(n)
        vals_axi = -np.ones(n, dtype=np.double)

        #combine triplets
        G1_rows = np.concatenate([rows_aw, rows_ab, rows_axi]).astype(int)
        G1_cols = np.concatenate([cols_aw, cols_ab, cols_axi]).astype(int)
        G1_vals = np.concatenate([vals_aw, vals_ab, vals_axi]).astype(np.double)

        G1 = spmatrix(G1_vals, G1_rows.tolist(), G1_cols.tolist(), (n, m))
        h1 = matrix(-np.ones(n), tc='d')

        #slack nonnegativity: -xi_i <= 0
        rows2 = np.arange(n)
        cols2 = d + 1 + np.arange(n)
        vals2 = -np.ones(n, dtype=np.double)
        G2 = spmatrix(vals2, rows2.tolist(), cols2.tolist(), (n, m))
        h2 = matrix(np.zeros(n), tc='d')

        #stack vertically: shape (2n, m)
        G = sparse([[G1], [G2]])
        h = matrix(
            np.hstack([np.asarray(h1).ravel(), np.asarray(h2).ravel()]),
            tc='d'
        )

        #solver options
        solvers.options['show_progress'] = True
        solvers.options['maxiters'] = 100
        solvers.options['abstol']   = 1e-7
        solvers.options['reltol']   = 1e-6
        solvers.options['feastol']  = 1e-7

        #solve QP
        sol = solvers.qp(P, q, G, h)
        z = np.array(sol['x']).ravel()

        #extract w, b
        self.w = z[:d]
        self.b = z[d]

        return self

    def decision_function(self, X):
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)
