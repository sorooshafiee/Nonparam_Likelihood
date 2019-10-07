import os
import argparse
import numpy as np
import cvxpy
from scipy.stats import entropy
from NPBC import likelihood_wass, likelihood_moment, likelihood_kl


cwd = os.getcwd()
DIR_CSV = os.path.join(cwd, "csv")
np.random.seed(1000)
parser = argparse.ArgumentParser(description='toy example')
parser.add_argument('--method', default='exp', type=str,
                    help='type of ambiguity set')
parser.add_argument('--eps', default=[1], nargs='+', type=float,
                    help='radius of ambiguity set')
parser.add_argument('--repeat', default=100, type=int,
                    help='number of repetition')
parser.add_argument('--ntrial', default=10, type=int,
                    help='number of trials m')
parser.add_argument('--theta', default=0.6, type=float,
                    help='parameter theta')
parser.add_argument('--alpha', default=1, type=float,
                    help='parameter alpha')
parser.add_argument('--beta', default=1, type=float,
                    help='parameter beta')
parser.add_argument('--discrete', default=19, type=int,
                    help='number of discretization points')
args = parser.parse_args()


def compute_beta(a, b, Theta):
    q = []
    for theta in Theta:
        pdf = theta ** (a - 1)
        pdf *= (1 - theta) ** (b - 1)
        q.append(pdf)
    return np.array(q) / sum(q)


def estimate_p(x, X_train, method, eps):
    if method == 'wasserstein':
        return likelihood_wass(x, X_train, eps)
    elif method == 'moment':
        return likelihood_moment(x, X_train)
    elif method == 'kl':
        return likelihood_kl(x, X_train, eps)
    else:
        diff = np.abs(X_train - x)
        # diff = diff ** 2
        if diff.ndim == 2:
            diff = np.mean(diff, axis=1)
        diff = np.exp(- eps * diff)
        return np.mean(diff)


def minimum_kl(p, Theta):
    n = len(p)
    q = cvxpy.Variable(shape=n)
    lp = cvxpy.Parameter(shape=n)
    pi = cvxpy.Parameter(shape=n)
    lp.value = np.log(p)
    pi.value = compute_beta(args.alpha, args.beta, Theta)
    constraints = [q >= 0.0, cvxpy.sum(q) == 1.0]
    R = cvxpy.kl_div(q, pi) - cvxpy.multiply(q, lp)
    objective = cvxpy.Minimize(cvxpy.sum(R))
    prob = cvxpy.Problem(objective, constraints)
    try:
        prob.solve()
    except:
        prob.solve(solver="CVXOPT")
    return np.abs(q.value)


def main():
    all_N = np.hstack([np.arange(1, 10) * 1e0,
                       np.arange(1, 11) * 1e1])
    fname = args.method
    if args.method in ('wasserstein', 'kl', 'exp'):
        for eps_ in args.eps:
            fname += '_{}'.format(eps_)
    fname = os.path.join(DIR_CSV, fname + ".csv")
    if args.method == 'moment':
        print('training with {} method'.format(args.method,))
    else:
        print('training with {} method and epsilon {}'.format(args.method, args.eps))
    if os.path.isfile(fname):
        print('the model is already trained.')
    else:
        Theta = np.linspace(0, 1, args.discrete + 2)[1:-1]
        eps = np.array([args.eps]).ravel()
        if eps.size == 1:
            eps = eps[0] * np.ones(args.discrete)
        result = []
        for r in range(args.repeat):
            result_1 = []
            result_2 = []
            result_3 = []
            result_4 = []
            err_1 = []
            err_2 = []
            x = np.random.binomial(args.ntrial, args.theta, 1)
            a = np.asscalar(x) + args.alpha
            b = args.ntrial - np.asscalar(x) + args.beta
            q = compute_beta(a, b, Theta)
            X_hat = []
            for theta in Theta:
                X_hat.append(np.random.binomial(args.ntrial, theta, int(all_N.max())))
            for N in all_N.astype(int):
                p = []
                if N == 1 and args.method == 'moment':
                    pass
                else:
                    for ind, theta in enumerate(Theta):
                        p.append(estimate_p(x, X_hat[ind][0:N], args.method, eps[ind]))
                    q_elbo = minimum_kl(np.abs(p), Theta)
                    q_bayes = np.abs(p) / np.sum(np.abs(p))
                    result_1.append(entropy(q, q_elbo))
                    result_2.append(entropy(q_elbo, q))
                    result_3.append(entropy(q, q_bayes))
                    result_4.append(entropy(q_bayes, q))
                    err_1.append(np.abs(Theta[np.argmax(q_elbo)] - args.theta))
                    err_2.append(np.abs(Theta[np.argmax(q_bayes)] - args.theta))
            result.append(result_1 + result_2 + result_3 + result_4 + err_1 + err_2)
        result = np.array(result)
        np.savetxt(fname, result, fmt="%0.5f", delimiter=",")


if __name__ == '__main__':
    main()