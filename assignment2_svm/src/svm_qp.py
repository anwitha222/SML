import numpy as np
from cvxopt import matrix, solvers

class SVM_QP:
    #soft margin SVM using QP (dual formulation) with kernels
    #y must be between -1 and +1

        def __init__(self, C=1.0, kernel=("linear", {}), tol=1e-6):

            self.C = float(C)
            self.kernel = kernel
            self.tol = tol
            self.alphas = None
            self.support_idx = None
            self.Xsv = None
            self. ysv = None
            self.b = None

        def _compute_kernel(self, X, Y):
            #helper to compute kernel matrix between X and Y
            name, params = self.kernel
            if name == "linear":
                from .kernels import linear_kernel
                return linear_kernel(X, Y)
            elif name == "rbf":
                from .kernels import rbf_kernel
                return rbf_kernel(X, Y, params.get("gamma", 0.01))
            elif name == "poly":
                from .kernels import poly_kernel
                return poly_kernel(X, Y, params.get("degree", 3), params.get("coef0", 1.0))
            else:
                raise ValueError(f"Unknown kernel: {name}")
            
        def fit(self, X, y):
            #train the SVM on training data

            #store training data 
            self.X_train = X.copy()

            #ensure labels are within -1,1
            y = y.astype(float)
            y = np.where(y == 0, -1.0, 1.0)
            self.y_train_pm = y

            n = X.shape[0]

            #kernel matrix
            K = self._compute_kernel(X, X) 

            # cvxopt formulation: minimize (1/2)x^T P x + q^T x
            # where x is alpha
            P = matrix(np.outer(y, y) * K, tc='d')
            q = matrix(-np.ones(n), tc='d')

            # inequality constraints: G x <= h encodes 0 <= alpha_i <= C
            G_top = np.diag(-np.ones(n))#-I for alpha >= 0  -> -alpha <= 0
            h_top = np.zeros(n)
            G_bot = np.diag(np.ones(n))#I for alpha <= C  ->  alpha <= C
            h_bot = np.ones(n) * self.C
            G = matrix(np.vstack([G_top, G_bot]), tc='d')
            h = matrix(np.concatenate([h_top, h_bot]), tc='d')

           #equality constraint: A x = b  where A = y^T, b = 0
            A = matrix(y.reshape(1, -1), tc='d')
            b = matrix(0.0, tc='d')

            #show every line 
            solvers.options['show_progress'] = True

            #fix runs taking ages add tolerences
            solvers.options['maxiters'] = 50      #hard cap on iterations
            solvers.options['abstol'] = 1e-7      
            solvers.options['reltol'] = 1e-6
            solvers.options['feastol'] = 1e-7

            #solve QP
            sol = solvers.qp(P, q, G, h, A, b)
            alphas = np.array(sol['x']).reshape(-1)

            #support vectors: alpha_i > tol
            sv_mask = alphas > self.tol
            self.alphas = alphas[sv_mask]
            self.support_idx = np.where(sv_mask)[0]
            self.Xsv = X[self.support_idx]
            self.ysv = y[self.support_idx]

            #compute bias b using KKT: for any 0 < alpha_i < C,
            #y_i ( sum_j alpha_j y_j K_ij + b ) = 1  ->  b = y_i - sum_j alpha_j y_j K_ij
            #prefer "free" SVs (strictly inside margin); if none, use all SVs.
            free_sv = (alphas > self.tol) & (alphas < self.C - self.tol)
            idxs = np.where(free_sv)[0] if np.any(free_sv) else self.support_idx

            K_sub = K[np.ix_(idxs, self.support_idx)]  # K for those idxs vs true SVs
            decision_no_b = (self.alphas * self.ysv) @ K_sub.T
            b_vals = self.y_train_pm[idxs] - decision_no_b
            self.b = float(np.mean(b_vals)) if b_vals.size else 0.0

            return self  #sklearn-style
            

        
        def decision_function(self, X):
            #compute the decision function values for X
            if self.alphas is None:
                raise RuntimeError("model not fitted. call fit(X, y) fitst")
            
            K_sv = self._compute_kernel(self.Xsv, X) 
            scores = (self.alphas * self.ysv)@K_sv + self.b
            return scores.ravel()
        
        def predict(self, X):
            scores = self.decision_function(X)
            return (scores >= 0).astype(int)