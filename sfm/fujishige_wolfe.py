""" This is an implementation of the Fujishige-Wolfe minimum norm algorithm for
finding the minimal minimiser of a submodular function.

The Fujishige-Wolfe algorithm returns the component-wise minimal submodular
minimiser and is called by the function sfm(n,f). Here the inputs are:
    - n is the dimension of the ground set, which is assumed to be {0,...,n-1}.
    - f is assumed to be a normalised submodular set function that takes as
    input a Python set S (subset of {0,...,n-1}) and normalised means that we
    assume have f(set[]) = 0.

Potential future additions:
    - add support for non-normalised submodular functions (by normalising)
    - speed up the affine_minimiser(S) function using ideas from Wolfe's
    original paper on his minimum norm algorithm.
    - various other speed optimisations

References:
1. Wolfe P. Finding the nearest point in a polytope. Math Program. 1976;
11(1):128-49.
2. Fujishige S, Hayashi T, Isotani S. The Minimum-Norm-Point Algorithm Applied
to Submodular Function Minimization and Linear Programming. RIMS Prepr 1571
[Internet]. 2006;1-19.
http://www.kurims.kyoto-u.ac.jp/preprint/file/RIMS1571.pdf
3. Chakrabarty D, Jain P, Kothari P. Provable Submodular Minimization via
Fujishige-Wolfe's Algorithm. ArXiv e-prints. 2014;
https://arxiv.org/abs/1411.0095
"""

import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def naive_sfm(n, f):
    minimiser = []
    minimum = f(minimiser)
    for S in powerset(range(n)):
        f_S = f(S)
        if f_S < minimum:
            minimiser = S
            minimum = f_S
    return minimiser

# Define constants
Z1 = 10e-12
Z2 = 10e-10
Z3 = 10e-10

def greedy(w, f):
    """
    Implements the Greedy algorithm that optimises linear programs over the
    base polyhedron.
    Input: vector w, submodular set function f.
    Output: vector x minimising <x,w> over base polyhedron B_f defined by f.
    """
    n = len(w)
    # Find a linear ordering so that w is in ascending order
    lin_ordering = np.argsort(w, kind='mergesort')
    # Initialise the point
    x = np.empty(n, dtype=float)
    f_vals = [f(lin_ordering[:i]) for i in range(n+1)]
    for i in range(n):
        x[lin_ordering[i]] = f_vals[i+1] - f_vals[i]
    return x

def is_in(v, A, eps):
    """Returns true if v is in A up to a tolerance of eps."""
    return np.any(np.all(np.absolute(A-v) < eps, axis=1))

def affine_minimiser(S):
    """Find affine minimiser of affine hull spanned by the vectors in S. Naive
    implementation using standard numpy methods!
    """
    m, n = S.shape
    Q = S.transpose()
    # Compute S^T * S
    M = np.matmul(S, Q)
    
    # Add ones to the left of the result and add row on top starting with a
    # single zero and otherwise ones
    M = np.concatenate([np.ones((m, 1)), M], axis=1)
    e = np.hstack([np.zeros((1, 1)), np.ones((1, m))])
    M = np.concatenate([e, M], axis=0)

    # Compute vector 'on the right' for the solver and solve linear system
    v = np.hstack([np.ones((1)), np.zeros((m))])
    w = np.linalg.solve(M, v)[1:]
    return w, np.dot(Q, w)

def memoize(f):
    memo = {}
    def helper(S):
        hash_S= tuple(S)
        if hash_S not in memo:
            memo[hash_S] = f(S)
        return memo[hash_S]
    return helper

def fw_sfm(n, f):
    """Implementation of the Fujishige-Wolfe minimum norm algorithm for SFM.
    Maintains a vector x explicitly and as a minimal affine combination given
    by a list of tuples (lambda, vector).

    The programming logic mainly follows the paper of Chakrabarty et al. and
    the original paper by Wolfe.
    """
    # NB: x,y,z are vectors while a,b,c are vectors of barycentric
    # coefficients.
    
    # decorate f with memoize
    f = memoize(f)

    # Initialise
    x = greedy(np.zeros(n, dtype=float), f)
    S = x.reshape((1,n))
    a = np.array([1], dtype=float)

    counter = 0
    # Major cycle
    while True:
        q = greedy(x, f)
        if is_in(q, S, Z2):
            break  # x is min norm point
        # Determine max dot(p,p) over columns p in S
        max_dot = max(np.diag(np.matmul(S,S.transpose())))
        corr_factor = max(np.dot(q,q), max_dot)
        # Check if dot(x,q) >= dot(x,x) with allowance for roundoff errors
        if np.dot(x, q) >= np.dot(x, x) - Z1*corr_factor:
            break  # x is min norm point
        else:
            S = np.vstack((S,q))
            a = np.hstack((a,np.zeros(1)))

        # Minor cycle
        minor_counter = 0
        while True:
            # Get affine minimiser y and its barycentric coefficients b wrt S
            b, y = affine_minimiser(S)
            if all(b >= 0):  # y is in conv(S)
                a, x = b, y  # update x to y (and bary coeffs)
                break
            else:  # y is not in conv(S)
                # Compute intersection z of x-y line segment with boundary
                # of conv(S).
                # Compute all indices for which a[i] - b[i] > 0
                ind = np.nonzero(a-b > Z2)
                theta = np.divide(a[ind], (a-b)[ind], dtype=float).min()
                x = theta*y + (1-theta)*x
                a = theta*b + (1-theta)*a
                # Update affine combination representation to express z and
                # only keep v in S if its barycentric coeff wrt y is > Z2.
                indices = np.nonzero(a <= Z2)[0]
                S = np.delete(S, indices, axis=0)
                a = np.delete(a, indices)
                x = np.matmul(S.transpose(), a)

    # Compute and return minimal minimising set from the minimum norm point x
    min_minimiser = {i for i in range(n) if x[i] <= -Z2}
    return min_minimiser
