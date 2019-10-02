import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
import cvxpy


def likelihood_wass(x, X_train, eps):
    N = X_train.shape[0]
    dist = np.abs(X_train - x)
    # dist = dist ** 2
    if dist.ndim == 2:
        dist = np.sum(dist, axis=1)
    ind = np.argsort(dist)
    dist = dist[ind]
    c_dist = np.cumsum(dist)
    z = np.zeros(N)
    if c_dist[-1] < N * eps:
        p_out = 1
    else:
        index = np.sum(c_dist < N * eps)
        z[:index] = 1.0 / N
        d_full = 0 if index == 0 else c_dist[index - 1]
        z[index] = (N * eps - d_full) / (c_dist[index] - d_full) / N
        p_out = np.sum(z)
    return p_out


def likelihood_moment(x, X_train):
    mu = np.array([np.mean(X_train, axis=0)]).ravel()
    mu_vec = mu[:, np.newaxis]
    cov = np.cov(X_train.T)
    top = np.concatenate((cov + mu_vec @ mu_vec.T, mu_vec), axis=1)
    bottom = np.concatenate((mu_vec.T, np.ones((1,1))), axis=1)
    Omega = np.concatenate((top, bottom), axis=0)
    x_vec = np.append(x, 1)
    try:
        sol = np.linalg.solve(Omega, x_vec)
        return 1 / np.abs(x_vec @ sol)
    except np.linalg.LinAlgError:
        f = lambda v: v @ Omega @ v
        g = lambda v: 2 * v @ Omega
        eq_cons = {"type": "eq",
                   "fun": lambda v: x_vec @ v - 1,
                   "jac": lambda v: x_vec}
        res = minimize(f, x_vec / np.linalg.norm(x_vec, 2), method="SLSQP", jac=g,
                       constraints=eq_cons, options={"disp": False})
        return np.maximum(np.abs(res.fun), 1e-8)


def likelihood_kl(x, X_train, eps):
    X, v = np.unique(X_train, return_counts=True, axis=0)
    v = v / X_train.shape[0]
    diff = np.abs(X - x)
    if diff.ndim == 2:
        diff = diff.sum(axis=1)
    n = v.size
    c = np.zeros(n)
    c[diff < 1e-8] = 1
    if c.sum() == 0:
        return 1 - np.exp(-eps)
    else:
        y_i = cvxpy.Variable(shape=n)
        v_i = cvxpy.Parameter(shape=n)
        c_i = cvxpy.Parameter(shape=n)
        v_i.value = v
        c_i.value = c
        R = cvxpy.kl_div(v_i, y_i) + v_i - y_i
        constraints = [y_i >= 0.0, cvxpy.sum(y_i) == 1.0, cvxpy.sum(R) <= eps]
        objective = cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(y_i, c_i)))
        prob = cvxpy.Problem(objective, constraints)
        try:
            prob.solve()
        except:
            prob.solve(solver="CVXOPT")
        return np.minimum(np.abs(objective.value), 1)


def likelihood_exp(x, X_train, eps):
    diff = np.abs(X_train - x)
    if diff.ndim == 2:
        diff = np.mean(diff, axis=1)
    diff = np.exp(- eps * diff)
    return np.mean(diff)


class NPBC(BaseEstimator, ClassifierMixin):
    """Non Parametric Sampling Based variational Bayes Approach to Classification"""
    def __init__(
            self,
            eps=0,
            method="wass",
            adaptive=True,
            verbose=False
    ):
        """Copy params to object properties"""
        self.eps = eps
        self.method = method
        self.adaptive = adaptive
        self.verbose = verbose
        self.priors = []
        self.eps_ = []

    def fit(self, X, y):
        """Fit the NPBC to the training data"""
        methods = ("wasserstein", "moment", "kl", "pearson", "hellinger", "tv", "exp")
        if self.method is not None and self.method.lower() not in methods:
            raise ValueError(
                "method must be either in {} or None; got (method={})".format(
                    methods, self.method))

        n_samples, self.n_features_ = X.shape
        self.labels_, self.n_samples_ = np.unique(y, return_counts=True)
        self.n_class_ = self.labels_.size
        self.eps_ = np.array([self.eps]).ravel()
        if self.eps_.size == 1:
            self.eps_ = self.eps_[0] * np.ones(self.n_class_)
        if self.adaptive:
            self.eps_ *= np.sqrt(self.n_features_)
        self.priors = self.n_samples_ / n_samples
        self.y_train = y
        self.X_train = X
        return self

    def predict(self, X):
        """ Predict the labels of a set of features """
        prob = self.predict_proba(X)
        return self.labels_[prob.argmax(1)]

    def predict_proba(self, X):
        """ Estimate the probability for the each class """
        X = check_array(X, dtype=np.float64)
        prob = []
        for n_sample, x in enumerate(X):
            prob.append(self.minimum_kl(x))
        return np.array(prob)

    def minimum_kl(self, x):
        p = self.likelihood_estimator(x)
        n = len(p)
        q = cvxpy.Variable(shape=n)
        lp = cvxpy.Parameter(shape=n)
        pi = cvxpy.Parameter(shape=n)
        lp.value = np.log(p + 1e-8)
        pi.value = self.priors
        constraints = [q >= 0.0, cvxpy.sum(q) == 1.0]
        R = cvxpy.kl_div(q, pi) + q - pi - cvxpy.multiply(q, lp)
        objective = cvxpy.Minimize(cvxpy.sum(R))
        prob = cvxpy.Problem(objective, constraints)
        try:
            prob.solve()
        except:
            prob.solve(solver="CVXOPT")
        return q.value

    def likelihood_estimator(self, x):
        p = np.zeros(self.n_class_)
        for n_c, label in enumerate(self.labels_):
            mask = (self.y_train == label)
            X_oneclass = self.X_train[mask, :]
            if self.method.lower() == "wasserstein":
                p[n_c] = likelihood_wass(x, X_oneclass, self.eps_[n_c])
            elif self.method.lower() == "moment":
                p[n_c] = likelihood_moment(x, X_oneclass)
            elif self.method.lower() == "kl":
                p[n_c] = likelihood_kl(x, X_oneclass, self.eps_[n_c])
            elif self.method.lower() == "hellinger":
                p[n_c] = 1 - (1 - self.eps_[n_c] ) ** 2
            elif self.method.lower() == "pearson":
                p[n_c] = 1 - (1 / (1 + self.eps_[n_c]))
            elif self.method.lower() == "tv":
                p[n_c] = self.eps_[n_c] / 2
            elif self.method.lower() == "exp":
                p[n_c] = likelihood_exp(x, X_oneclass, self.eps_[n_c])
        return p


# def likelihood_wass(x, X_train, eps, verbose=False):
#     N = X_train.shape[0]
#     n = x.size
#     res = X_train - x
#     if n == 1:
#         dist = np.abs(res)
#     else:
#         dist = np.sum(res ** 2, axis=1)
#     P = picos.Problem()
#     T = P.add_variable("T", N)
#     P.add_constraint(T >= 0)
#     p_emp = np.ones(N) / N
#     P.add_constraint(T <= p_emp)
#     P.add_constraint(picos.sum([dist[i] * T[i] for i in range(N)]) <= eps)
#     obj = picos.sum([T[i] for i in range(N)])
#     P.maximize(obj, verbose=verbose, solver="cvxopt")
#     return obj.value


# def likelihood_moment(x, X_train, verbose=False):
#     # first implementation
#     mu = np.array([np.mean(X_train, axis=0)]).ravel()
#     cov = np.cov(X_train.T)
#     x_vec = np.append(x, 1)
#     x_vec = x_vec[:, np.newaxis]
#     mu_vec = mu[:, np.newaxis]
#     P = picos.Problem()
#     d = int(np.sqrt(cov.size))
#     M = P.add_variable("M", (d + 1, d + 1), vtype="symmetric")
#     P.add_constraint(x_vec.T * M * x_vec >= 1)
#     P.add_constraint(M >> 0)
#     top = np.concatenate((cov + mu_vec * mu_vec.T, mu_vec), axis=1)
#     bottom = np.concatenate((mu_vec.T, np.ones([1, 1])), axis=1)
#     Pi = np.concatenate((top, bottom), axis=0)
#     obj = picos.trace(Pi * M)
#     P.minimize(obj, verbose=verbose, solver="cvxopt")
#     p_opt = obj.value
#     return p_opt


