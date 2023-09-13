import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

try:
    from mosek.fusion import *
except:
    print("Warning: Could not load module named mosek.fusion")

try:
    import cvxpy as cvx
except:
    print("Warning: Could not load module named cvxpy")

def log_mvn_pdf(X, atoms, prec, prec_type, homoscedastic):
    """
    given:
       an n x d observation ndarray X,
       an m x d ndarray of atoms,
       an ndarray of precision matrices, whose shape 
           ndarray of shape 
                    (n, d, d) if heteroscedastic & prec_type==general  
                    (n, d)    if heteroscedastic & prec_type==diagonal
                    (d, d)    if homoscedastic & prec_type==general
                    (d,)      if homoscedastic & prec_type==diagonal
    return:
       an n x m matrix of log-probabilities A, where
       A[i, j] = log MVN(X[i] | atoms[j], prec[i]),
       and MVN(X | μ, Ω) is the density of a multivariate 
       normal with mean μ and precision Ω evaluated at X.
    """
    n, d = X.shape
    
    ## the (squared) mahalanobis distance M between all pairs
    ## and the log-determinant ld of the precision matrices
    if prec_type=='general' and not homoscedastic:
        diffs = (X[:, np.newaxis, :] - atoms[np.newaxis, :, :])
        M = np.einsum('ijk,ikl,ijl->ij',diffs,prec,diffs)
        ld = np.linalg.slogdet(prec)[1][:, np.newaxis]
    elif prec_type=='diagonal' and not homoscedastic:
        diffs = (X[:, np.newaxis, :] - atoms[np.newaxis, :, :])
        M = np.einsum('ik,ijk->ij',prec,diffs**2)
        ld = np.sum(np.log(prec), 1)[:, np.newaxis]
    elif prec_type=='general' and homoscedastic:
        M = (np.einsum('ik,kl,il->i',X,prec,X)[:, np.newaxis]
             + np.einsum('jk,kl,jl->j',atoms,prec,atoms)[np.newaxis, :]
             - 2*np.einsum('ik,kl,jl->ij',X,prec,atoms))
        ld = np.linalg.slogdet(prec)[1]
    elif prec_type=='diagonal' and homoscedastic:
        M = ((X**2 @ prec)[:, np.newaxis] 
             + (atoms**2 @ prec)[np.newaxis, :] 
             - 2*np.einsum('ik,k,jk->ij',X,prec,atoms))
        ld = np.sum(np.log(prec))
    else:
        raise Exception("prec_type must be 'general' or 'diagonal'")
    return(-(M + d*np.log(2*np.pi) - ld) / 2)

def mvn_pdf(X, atoms, prec, prec_type, homoscedastic, 
            n_chunks=1, log_prob_thresh=-float('Inf'), row_condition=False):
    """
    given:
       an n x d observation ndarray X,
       an m x d ndarray of atoms,
       an ndarray of precision matrices, of shape 
                    (n, d, d) if heteroscedastic & prec_type==general  
                    (n, d)    if heteroscedastic & prec_type==diagonal
                    (d, d)    if homoscedastic & prec_type==general
                    (d,)      if homoscedastic & prec_type==diagonal
    additional options:
       n_chunks, int default 1
           compute the kernel in chunks
       log_prob_thresh, float default -∞
           set all log probabilities under log_prob_thresh
           to -∞ (so the resulting probability is exactly zero)
       row_condition, bool default False
           if True, divide each probability by the max in each row,
           so the maximum entry in each row is guaranteed to be one
    return:
       an n x m sparse array (csr_matrix) of probabilities K, where
       K[i, j] = MVN(X[i] | atoms[j], prec[i]),
       and MVN(X | μ, Ω) is the density of a multivariate 
       normal with mean μ and precision Ω evaluated at X.
    """
    n = X.shape[0]
    ind_list = np.array_split(np.arange(n), n_chunks)
    res = []
    for inds in ind_list:
        A = log_mvn_pdf(X[inds], atoms, 
                        prec if homoscedastic else prec[inds], 
                        prec_type, homoscedastic)
        if row_condition:
            A = A - np.max(A, 1)[:, np.newaxis]
        A[A < log_prob_thresh] = -float('Inf')
        res.append(sparse.csr_matrix(np.exp(A)))
    return(sparse.vstack(res))


def solve_weights_cvx(K, **solver_params):
    """
    given:
       an n x m kernel K csr_matrix of 
       probabilities, scaled by row
    return:
       the weights w_1,…,w_m maximizing 
        sum_{i=1}^n log(Aw)_i    
       using cvxpy
    """
    n,m = K.shape
    A = K.toarray()

    w = cvx.Variable(m)
    constraints = [w >= 0, cvx.sum(w) == 1]
    obj = cvx.Maximize(cvx.sum(cvx.log(A @ w)))

    prob = cvx.Problem(obj, constraints)
    prob.solve(**solver_params)

    return(w.value)

def solve_weights_mosek(K, use_sparse=True, **solver_params):
    """
    given:
       an n x m kernel K csr_matrix of 
       probabilities, scaled by row
    return:
       the weights w_1,…,w_m maximizing 
        sum_{i=1}^n log(Aw)_i    
       via an exponential cone program in mosek
    """
    n,m = K.shape
    M = Model()

    t = M.variable(n)
    u = M.variable(n, Domain.greaterThan(0.0))
    w = M.variable(m, Domain.inRange(0.0, 1.0))

    ## exponential cone constraints
    for i in range(n):
        ## constrains t[i] ≤ log(u[i]) and u[i] ≥ 0
        M.constraint(Expr.hstack(u.index(i), 1, t.index(i)), 
                     Domain.inPExpCone())
    M.constraint(Expr.sum(w), Domain.equalsTo(1.0))
    if use_sparse:
        K = K.tocoo()
        A = Matrix.sparse(K.row, K.col, K.data)
    else:
        A = Matrix.sparse(K.todense())
    M.constraint(Expr.sub(Expr.mul(A, w), u), Domain.equalsTo(0.0))

    ## Set the objective function to sum(t)
    M.objective("obj", ObjectiveSense.Maximize, Expr.sum(t))
    
    for k,v in solver_params.items():
        M.setSolverParam(k, v)
    
    ## Solve
    M.solve()
    
    return w.level()

class GLMixture:
    """
    
    Kiefer-Wolfowitz nonparametric maximum likelihood estimation 
    (NPMLE) for Gaussian location mixtures in general dimensions,
    supporting general heteroscedastic errors. 


    ----------------------------OPTIONS-----------------------------
    
    prec_type : string, default 'general'
        Type of covariance matrix. Either 'general' or 'diagonal'.
            
    homoscedastic : bool, default False
    
    atoms_init : array, default None
        Specify a set of atoms to use in the discretization.
    
    --------------------------ATTRIBUTES----------------------------
    
    d : int (ambient dimension)
    
    m : int (number of atoms)
    
    n : int (number of training obs)
    
    weights : ndarray of shape (m,)
    
    atoms : ndarray of shape (m, d)
    
    XTrain : ndarray of shape (n, d)
    
    PrecTrain : ndarray of shape 
                    (n, d, d) if heteroscedastic & general  
                    (n, d)    if heteroscedastic & diagonal
                    (d, d)    if homoscedastic & general
                    (d,)      if homoscedastic & diagonal

    """
    
    def __init__(self, prec_type='general', 
                 homoscedastic=False, atoms_init=None):
        self.prec_type = prec_type
        self.homoscedastic = homoscedastic
        self.atoms_init = atoms_init
        
    def get_params(self):
        return(self.atoms, self.weights)
    
    def set_params(self, atoms, weights):
        self.atoms, self.weights = atoms, weights
        
    def initialize_atoms_subsample(self, X, n_atoms):
        self.atoms_init = X[np.random.choice(X.shape[0], 
                                             size=n_atoms, 
                                             replace=False)]
    
    def fit(self, X, prec, max_iter_em=10, weight_thresh=0., 
            n_chunks=1, log_prob_thresh=-float('Inf'), row_condition=False, 
            score_every=1, solver='cvxpy', **solver_params):
        """
        given:
           an n x d observation ndarray X,
           an ndarray of precision matrices, of shape 
                        (n, d, d) if heteroscedastic & prec_type==general  
                        (n, d)    if heteroscedastic & prec_type==diagonal
                        (d, d)    if homoscedastic & prec_type==general
                        (d,)      if homoscedastic & prec_type==diagonal
           max_iter_em : int, default 10
               number of iterations of EM
           weight_thresh : float, default 0
               threshold value for discretized NPMLE weights
           n_chunks : int, default 1
               compute the kernel in chunks
           log_prob_thresh : float, default -∞
               set all log probabilities under log_prob_thresh
               to -∞ (so the resulting probability is exactly zero)
           row_condition : bool, default False
               if True, divide each probability by the max in each row,
               so the maximum entry in each row is guaranteed to be one
           score_every : int, default 1
               number of EM iterations between computing avg log-likelihood
               if None, `fit` method will not call the `score` method
           solver_params : 
               additional parameters for mosek conic interior solver

        
        Approximate the NPMLE by 
           (1) computing the discretized NPMLE over a set atoms_init
           (2) running EM to improve atoms and weights jointly
        Updates self.atoms and self.weights
         
        return:
           a list of average log-likelihoods across EM iterations
        """
        self.n, self.d = X.shape
        self.XTrain = X
        self.PrecTrain = prec
        
        ## initialize atoms
        if self.atoms_init is None:
            print('Selecting all data points as atoms: done.')
            self.atoms_init = X
        
        ## compute sparse kernel matrix
        print('Computing kernel matrix:', end=' ')
        K = mvn_pdf(X, self.atoms_init, prec, self.prec_type, 
                    self.homoscedastic, n_chunks, 
                    log_prob_thresh, row_condition)
        print('done.')
        
        ## solve for the optimal weights given locations atoms_init
        print('Solving for discretized NPMLE:', end=' ')
        if solver=='mosek':
            w = solve_weights_mosek(K, 
                                   use_sparse=(log_prob_thresh > -float('Inf')), 
                                   **solver_params)
        elif solver=='cvxpy':
            w = solve_weights_cvx(K, # no use_sparse option
                                  **solver_params)
        print('done.')
        
        ## threshold weights
        atoms = self.atoms_init[w > weight_thresh]
        weights = w[w > weight_thresh]
        weights /= sum(weights)
        self.set_params(atoms, weights)
        if score_every is not None: llik = [self.score(X, prec)]
        
        if self.prec_type=='diagonal' and not self.homoscedastic:
            precX = np.multiply(prec, X)
        
        ## optimize over weights and locations using EM 
        for _ in range(max_iter_em):
            ## E-step : compute responsibilities
            B = mvn_pdf(X, atoms, prec, self.prec_type, 
                        self.homoscedastic, n_chunks, 
                        log_prob_thresh, row_condition=False)
            resp = B.multiply(weights)
            resp = normalize(resp, norm='l1', axis=1, copy=False)

            ## M-step : update cluster centers
            Nk = np.array(resp.sum(0).T)
            weights = np.ravel(Nk / np.sum(Nk))
            if self.homoscedastic:
                atoms = np.array((resp.transpose().dot(X)) / Nk)
            else:
                if self.prec_type=='general':
                    resp = np.array(resp.todense())
                    Z = np.einsum('ij,ikl->jkl', resp, prec)
                    N = np.einsum('ij,ikl,il->jk', resp, prec, X)
                    for j in range(atoms.shape[0]):
                        atoms[j] = np.linalg.inv(Z[j]) @ N[j]
                elif self.prec_type=='diagonal':
                    num = resp.transpose().dot(precX)
                    dnm = resp.transpose().dot(prec)
                    atoms = np.array(num/dnm)
                    
            if (score_every is not None) and (_ % score_every == 0): 
                print('Running EM: iteration %s / %s' %(_, max_iter_em), 
                      end='\r', flush=True)
                self.set_params(atoms, weights)
                llik.append(self.score(X, prec))
       
        if (score_every is not None): 
            print(' ' * (30 + 2*len(str(max_iter_em))), end='\r', flush=True)    
            print('Running EM: done.', end='\r', flush=True)
            return llik
        
    def score(self, X, prec, n_chunks=1, 
              log_prob_thresh=-float('Inf')):
        """
        given:
           an n x d observation matrix X,
           an array of precision matrices, whose shape depends
           on prec_type and homoscedastic
        return:
           the average log-likelihood across observations X_i.
        """
        K = mvn_pdf(X, self.atoms, prec, self.prec_type, 
                    self.homoscedastic, n_chunks, 
                    log_prob_thresh, row_condition=False)
        return(np.mean(np.log(K.dot(self.weights))))
        
    def posterior_mean(self, X, prec, row_condition=True):
        """
        given:
           an n x d observation matrix X,
           an array of precision matrices, whose shape depends
           on prec_type and homoscedastic
        return:
           the Bayes estimate (posterior mean) for each X_i in 
           the (multivariate, heteroscedastic) normal means
           model, using the prior defined by atoms and weights. 
       """
        a, w = self.get_params()
        A = mvn_pdf(X, a, prec, self.prec_type, self.homoscedastic, 
                row_condition=row_condition)
        return(((A.dot(a * w[:, np.newaxis])) 
               / np.ravel(A.dot(w))[:, np.newaxis]))
