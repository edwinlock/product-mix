# -*- coding: utf-8 -*-
""" This module is an implementation of the Product-Mix auction originally developed by Paul Klemperer (cf. https://www.nuffield.ox.ac.uk/users/klemperer/productmix.pdf).

Specifically, we present an implementation of the algorithms developed in the working paper (BGKL) by Elizabeth Baldwin, Paul Goldberg,
Paul Klemperer and Edwin Lock available on the ArXiv at https://arxiv.org/abs/1909.07313. The algorithms solve the Product-Mix Auction; that is, they find a competitive (Walrasian) equilibrium. Computing the equilibrium can be separated into two parts:

1) Find the component-wise minimal market-clearing price using a steepest
descent approach. Both long-step methods described in the paper are
implemented.

2) Find an allocation of the supply (=target) bundle among the various bidders
so that each bidder receives a bundle they demand at the market-clearing price.

Examples
-------
See readme.md for example usage.

Notes
-----
We take advantage of efficient matrix operations implemented by the Numpy package. In order to do this, the dot bids for each bidder are stored in a single 2d numpy array containing the bid vectors of bidder j as row vectors.

(c) 2019 by Edwin Lock.

See LICENCE file for software licence.

"""

import numpy as np
from sfm.fujishige_wolfe import fw_sfm, naive_sfm
from disjointset.disjointset import DisjointSet
import itertools
import json
import datetime
from typing import Dict, List, Optional

class AllocationProblem:
    """Implements a custom data structure for allocation problems.
    
    Parameters
    ----------
    good_no : int
        number of goods (without nominal reject good)
    bidder_no : int
        number of bidders
    eps : float, optional
        amount by which bids are shifted in the allocation algorithm
    
    Attributes
    ----------
    n : int
        Number of goods, including nominal reject good 0.
    m : int
        Number of bidders.
    goods : list of int
        List of goods [0,..,n-1] (including nominal reject good).
    bidders : list of int
        List of bidders [0,...,m-1]
    eps : float
        Amount by which bids are shifted in the allocation algorithm.
    bidlists : list of 2d numpy arrays of floats
        Each entry bidlists[j] is a 2d numpy array containing the bid vectors
        of bidder j as row vectors, i.e. the k-th bid of bidder j is accessed
        by bidlists[j][k].
    weights : list of 1d numpy arrays of ints
        Each entry weights[j] contains a one-dimensional numpy array
        with the weights of the bids of bidder j, i.e. weights[j][k] stores
        the weight of bidder j's k-th bid.
    prices : numpy array of floats of dimension n
        Stores a price vector with n entries, where the 0th entry is the
        price for the reject good should always be 0.
    residual : numpy array of floats of dimension n
        Denotes the items of the supply bundle (including notional reject
        items) that have not yet been allocated.
    partial : list of 1d numpy arrays of floats of dimension n
        For each bidder, we store which items of goods (including nominal
        reject items) they have already been allocated.
    """
    
    def __init__(self, good_no, bidder_no, eps = 0.0625):
        self.n = good_no+1  # includes the 'reject' good
        self.m = bidder_no
        self.goods = range(self.n)  # including reject good
        self.bidders = range(self.m)
        self.eps = eps
        self.bidlists = []
        self.weights = []
        self.prices = np.zeros(self.n, dtype=float)
        self.residual = np.zeros(self.n)
        self.partial = [np.zeros(self.n) for _ in self.bidders]
    
    def __str__(self):
        """Provides a string representing instance, mainly for debugging."""
        
        output = "prices: {}\n".format(self.prices)
        output += "residual: {}\n".format(self.residual)
        for j in self.bidders:
            output += "bidder {}\'s data:\n".format(j)
            output += "bidlist: {}\n".format(self.bidlists[j])
            output += "weights: {}\n".format(self.weights[j])
            output += "partial: {}".format(self.partial)
        return output


# TODO: add the following function
# def add_bids():
#     """TO DO"""
#     pass

def load_from_json(filename: str) -> AllocationProblem:
    """Loads an allocation problem from a JSON file.
    
    Parameters
    ----------
    filename : str
        Filename (and path to file if required) of data file.
        Example: 'examples/example1.json'.
    
    Returns
    -------
    AllocationProblem
        An AllocationProblem object encoding the allocation problem specified
        in the JSON file.
    
    """
    
    # Read data from file
    with open(filename) as input_file:
        data = json.load(input_file)
    # Unpack data
    goods = data['goods']
    bidders = data['bidders']
    bid_data = data['bidlists']
    supply = data['supply']
    # Create AllocationProblem instance
    alloc = AllocationProblem(goods, bidders)
    # Populate bidlists and weights
    item_counter = 0  # counts total number of items allocated by all bids
    for j in alloc.bidders:
        B = len(bid_data[j])  # number of bids of bidder j
        alloc.bidlists.append(np.empty(shape=(B, goods+1), dtype=float))
        alloc.weights.append(np.empty(B, dtype=float))
        for i in range(B):
            bid_entry = bid_data[j][i]
            vector = np.array([0] + bid_entry['vector'], dtype=float)
            weight = bid_entry['weight']
            alloc.bidlists[j][i] = vector
            alloc.weights[j][i] = weight
            item_counter += weight
    # Compute and set residual vector
    reject_items = item_counter-sum(supply)  # number of reject items
    alloc.residual = np.array([reject_items] + data['supply'], dtype=float)
    return alloc
    
def save_to_json(alloc: AllocationProblem,
                 filename: Optional[str] = None) -> None:
    """Store the allocation problem encoded in the bid lists as a JSON file.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance we want to save to a JSON file.
    filename: str
        Location and name of output file.
    
    Returns
    -------
        None
    """

    data = {}
    data['title'] = "Auto-generated"
    data['date'] = str(datetime.date.today())
    data['goods'] = alloc.n-1
    data['bidders'] = alloc.m
    data['supply'] = list(alloc.residual[1:])
    data['bidlists'] = []

    for j in alloc.bidders:
        B = len(alloc.bidlists[j])
        data['bidlists'].append([{'weight': alloc.weights[j][i],
                                  'vector': list(alloc.bidlists[j][i])[1:]}
                                  for i in range(B)])
    # Save to file
    if filename is None:
        filename = "auto-{}-{}-{}".format(alloc.n, alloc.m, alloc.q)
    with open(filename, 'w') as output_file:
        json.dump(data, output_file, indent=4)

def demanded_goods(bid_vector: np.ndarray, p: np.ndarray) -> set:
    """Determine and return the set of goods demanded by bid_vector at p.
    
    Parameters
    ----------
    bid_vector: 1d np.ndarray
        A single bid vector.
    p: 1d np.ndarray
        A price vector with the same expected length as bid_vector.
    
    Returns
    -------
    set
        Set of goods demanded by bid_vector at prices p.
    
    """
    
    diff = bid_vector - p
    max_utility = diff.max()
    return np.nonzero(diff == max_utility)[0]

def get_demand_vectors(bidlist: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute a matrix of vectors indicating the demanded goods for each bid.
    
    Recall that bidlist is a 2d numpy array consisting of dot bid row vectors.
    We return a boolean matrix in which the k-th row is a vector v that has
    entry 1 at the i-th position if good i is demanded by the k-th bid at
    prices p, and has entry 0 otherwise.
    
    Parameters
    ----------
    bidlist: 2d np.ndarray
        A 2d numpy matrix containing rows of bid vectors.
    p: 1d np.ndarray
        A price vector.
    
    Returns
    -------
    np.ndarray
        A boolean matrix M of the same dimensions as bidlist. M[k][i] = 1 iff
        good i is demanded by the k-th bid in bidlist at prices p.

    """
    
    diff = bidlist - p
    utility = diff.max(axis=1, keepdims=1)
    # initialise demand_vectors
    demand_vectors = np.empty(shape=diff.shape, dtype = np.int8)
    # populate
    np.equal(diff, utility, out = demand_vectors)
    return demand_vectors

def shift(bidlist: np.ndarray, i: int, eps: float) -> None:
    """Shifts all vectors in bidlist by eps in direction i.
    
    Effectively, the vector eps*e^i is added to each row in bidlist.

    Parameters
    ----------
    bidlist: 2d np.ndarray
        A 2d numpy matrix containing rows of bid vectors.
    i: int
        Some good.
    eps: float
        The amount by which bids get shifted in direction i.

    Returns
    -------
    None
    
    """
    n = bidlist.shape[1]
    char_vector = np.zeros(n)
    char_vector[i] += eps
    bidlist += char_vector

def project(bidlists: list, prices: np.ndarray) -> None:
    """Projects bid vectors of all bidders according to prices.
    
    Parameters
    ----------
    bidlists: list of 2d np.arrays
        A Python list of the bidlists of all bidders.
    
    prices: np.ndarray
        Some price vector.
    
    Returns
    -------
    None
    
    """
    
    n = len(prices)
    for bidlist in bidlists:
        diff = bidlist - prices
        max_utility = diff.max(axis=1, keepdims=1)
        demand_vectors = (diff == max_utility)
        demands_reject = demand_vectors[:,[0]]
        # Add demand_vectors to bidlist
        bidlist += demand_vectors
        # If 0 is demanded, subtract the all ones vector
        subtract = np.repeat(demands_reject, n, axis=1)
        bidlist -= subtract

def lyapunov(alloc: AllocationProblem, prices: np.ndarray) -> float:
    """Implements the Lyapunov function.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    
    prices: np.ndarray
        Price vector at which we wish to evaluate the Lyapunov function.
    
    Returns
    -------
    float
        Result of evaluating the Lyapunov function.

    """

    output = np.dot(alloc.residual, prices)
    for j in alloc.bidders:        
        # get the utility for each bid
        utilities = (alloc.bidlists[j]-prices).max(axis=1)
        output += np.dot(utilities, alloc.weights[j])
    return output

def demanded_bundle(alloc: AllocationProblem, prices: np.ndarray) -> np.ndarray:
    """Returns a bundle x that is demanded at p by accepting, for each bid, the
    smallest good demanded at p.
    
    Note that x may not be uniquely demanded at p!
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    
    prices: np.ndarray
        Price vector at which we wish to find a demanded bundle.
    
    Returns
    -------
    np.ndarray
        Some bundle that is demanded at p.
    
    """
    
    bundle = np.zeros(alloc.n, dtype=float)
    for j in alloc.bidders:
        for i in range(len(alloc.bidlists[j])):
            bid_vector = alloc.bidlists[j][i]
            smallest_good = np.argmax(bid_vector - prices)
            bundle[smallest_good] += alloc.weights[j][i]
    return bundle

def min_up(alloc: AllocationProblem,
           long_step_routine: str = "binarysearch",
           prices: np.ndarray = None) -> np.ndarray:
    """Implements the MinUp algorithm from BGKL.
           
    This is a steepest descent algorithm starting from the origin;
    it uses submodular minimisation to find the direction.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    
    long_step_routine: str
        expected to be the empty string (for unit steps), "binarysearch" or
        "demandchange" (cf. BGKL, Section 3.1 on longer step sizes)
    
    prices: np.ndarray
        initial price vector (default is 0). Must be a price vector that is
        element-wise dominated by some market-clearing price vector.

    Returns
    -------
    np.ndarray
        Market-clearing price vector, i.e. the prices at which competitive
        equilibrium is reached.

    """
    steps = 0
    # Initialise starting point
    if prices is None:
        p = np.zeros(alloc.n, dtype=float)
    else:
        p = prices.copy()
    # Perform steepest descent
    while True:
        # Determine inclusion-wise minimal minimiser S of submodular fn g'
        g_dash = get_g_dash(alloc, p)
        S = fw_sfm(alloc.n-1, g_dash)
        S = {i+1 for i in S}
        if not S:  # S is empty, minimiser found
            return p
        else:  # determine the steepest descent direction from S
            d = np.zeros(alloc.n)
            for s in S:
                d[s] += 1 # translate s, as sfm returns subset of {0,...,n-1}
            # Compute step length
            if long_step_routine == "":
                length = 1
            elif long_step_routine == "binarysearch":
                length = _binary_search(alloc, p, d)
            else:
                length = _demand_change(alloc.bidlists, p, S)
            # Perform step in direction d by updating p
            p += length * d
            steps += 1

def _demand_change(bidlists, prices, S):
    """Implements the demand change technique for long steps in MinUp.
    
    Parameters
    ----------
    bidlists: np.ndarray
        2d Numpy array consisting of rows of bid vectors.
    
    prices: np.ndarray
        Some price vector.
    S: set
        set of goods defining the direction e^S.
        
    Returns
    -------
    int
        Step length
    
    """
    
    n = len(prices)
    length = np.inf
    for bidlist in bidlists:
        demand_vec = get_demand_vectors(bidlist, prices)
        goods_vec = np.array([i in S for i in range(0, n)])
        is_subset = (np.any(demand_vec & goods_vec, axis=1)
                    & ~np.any(demand_vec & ~goods_vec, axis=1))
        utilities = (bidlist[is_subset] - prices).max(axis=1)
        other = (bidlist[is_subset] * ~goods_vec - prices).max(axis=1)
        surplus_gap = utilities - other
        length = min(length, np.min(surplus_gap))
    return length

def _binary_search(alloc, p, d):
    """Compute step length using the binary search long step from BGKL.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    p: np.ndarray
        Price vector.
    d: np.ndarray
        direction vector s.t. d in {0,1}^n.
    
    Returns
    -------
    int:
        Positive step length

    """

    M = max([bidlist.max() for bidlist in alloc.bidlists])
    # initialise binary search interval
    lower = 1
    upper = M
    def lyap(p):  # shorthand for convenience
        return lyapunov(alloc, p)

    # Compute the value of g' at p
    val_at_p = lyap(p+d) - lyap(p)
    while upper-lower > 1:
        avg = (lower + upper) // 2
        if val_at_p < lyap(p+avg*d) - lyap(p+(avg-1)*d):
            upper = avg-1
        else:
            lower = avg

    # At this point we have either upper == lower or upper-lower == 1.
    # If upper == lower, returning upper and lower is obviously the same.
    # Otherwise, if upper-lower == 1, we check which to return.
    if val_at_p == lyap(p+upper*d) - lyap(p+(upper-1)*d):
        return upper
    else:
        return lower

def procedure1(alloc):
    """Allocate all unambiguous non-marginal bids to the relevant bidder.
    
    This corresponds to Procedure 1 in BGKL. The function deletes all
    non-marginal bids from all bidlists and updates alloc.partial
    as well as alloc.residual.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    
    Returns
    -------
    None

    """

    for j in alloc.bidders:
        demand_vectors = get_demand_vectors(alloc.bidlists[j], alloc.prices)
        non_marginals = np.sum(demand_vectors, axis=1, keepdims=1) == 1
        change = demand_vectors*non_marginals*alloc.weights[j].reshape(-1,1)
        summed = np.sum(change, axis=0)
        alloc.partial[j] += summed
        alloc.residual -= summed
        # Delete bid vectors and weights from bidlists and weights
        mask = ~non_marginals.reshape(-1)
        alloc.bidlists[j] = alloc.bidlists[j][mask]
        alloc.weights[j] = alloc.weights[j][mask]

def procedure2(alloc, I, j, i):
    """Allocate unambiguous marginal bids to the relevant bidder. 
    
    This corresponds to Procedure 2 in BGKL.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    I: set
        Set of goods, the first element of the key list (I,j)
    j: int
        Some bidder, the second element of the key list (I,j)
    i: int or None
        Link good, if the key list (I,j) has a link good, otherwise None.

    Returns
    -------
    None
    
    """

    # Compute the bids BIj of bidder j that are marginal on goods in I.
    demand_vectors = get_demand_vectors(alloc.bidlists[j], alloc.prices)
    goods_vector = np.array([k in I for k in range(0, alloc.n)])
    BIj = np.any(demand_vectors & goods_vector, axis=1)
    
    if i is not None:  # Keylist has link good i
        # Compute difference of sum of weights of bids in BIj and residual
        # supply of all goods in I apart from i.
        weight_sum = np.dot(alloc.weights[j], BIj)
        supply = np.dot(alloc.residual, goods_vector) - alloc.residual[i]
        diff = weight_sum - supply
        # Allocate items of good i to bidder j
        alloc.partial[j][i] += diff
        alloc.residual[i] -= diff
        # Remove i from goods in I as these items have already been allocated
        goods_vector[i] = False

    # Allocate items of goods in I to bidder j
    alloc.partial[j] += np.multiply(alloc.residual, goods_vector)
    alloc.residual -= np.multiply(alloc.residual, goods_vector)
    # Remove bid vectors and weights of BIj from bidlists[j] and weights[j]
    alloc.bidlists[j] = alloc.bidlists[j][~BIj]
    alloc.weights[j] = alloc.weights[j][~BIj]

def get_g_dash(alloc, p, eps=1):
    """Returns the function g_dash as defined in BGKL.
    
    The function is defined as g_dash(S) := g(p+eps*e^S) - g(p), where g
    is the Lyapunov function.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    p: np.ndarray
        price vector
    eps:
        parameter
    
    Returns
    -------
    function:
        A Python function encoding the function g_dash from BGKL.

    """
    
    normaliser = lyapunov(alloc, p)

    def g_dash(S):
        """Expects a subset S of {0,...,alloc.n-1} and returns
        lyapunov(alloc, p + eps*e^S) - lyapunov(alloc, p).
        """
        q = p.copy()
        for s in S:
            q[s+1] += eps
        return lyapunov(alloc, q) - normaliser
    return g_dash

def get_h_dash(alloc, p, eps=1):
    return get_g_dash(alloc, p, -eps)

def procedure3(alloc, i, j):
    """ Shift, project and unshift to break ambiguities and simplify the
    allocation problem.
    
    Takes as input a link good i and bidder j such that
    there exists an edge from i to a key list (I,j) with i in I that lies
    in a circle in the marginal bids graph. This corresponds to Procedure 3
    in BGKL.
    """
    # Shift the bids of bidder j in direction e^i by alloc.eps
    shift(alloc.bidlists[j], i, alloc.eps)
    # Find new market-clearing price and update alloc.prices:
    # Define submodular functions
    g_dash = get_g_dash(alloc, alloc.prices, alloc.eps)
    h_dash = get_h_dash(alloc, alloc.prices, alloc.eps)
    # Find minimal submodular minimisers of g_dash and h_dash
    S = fw_sfm(alloc.n-1, g_dash)  # find the minimal minimiser
    T = fw_sfm(alloc.n-1, h_dash)  # find the minimal minimiser
    # Determine new market-clearing prices according to result (S or T)
    p = alloc.prices.copy()
    if g_dash(S) < h_dash(T):
        for good in S:
            p[good+1] += alloc.eps
    else:
        for good in T:
            p[good+1] -= alloc.eps
    # Project bids of all bidders according to new prices p
    project(alloc.bidlists, p)
    # Unshift
    shift(alloc.bidlists[j], i, -alloc.eps)

def get_keylists(alloc):
    """Computes and returns all key lists of the marginal bids graph associated
     with the allocation problem given in the input.
    """
    keylists = []
    for j in alloc.bidders:
        """Generate vertex sets corresponding to the connected components of
        the graph induced by all j-labelled edges in the marginal bids graph.
        Uses disjoint set data structure implemented by the DisjointSet class
        (see disjointset.pyx)."""
        DS = DisjointSet(alloc.n) 
        demand_vectors = get_demand_vectors(alloc.bidlists[j], alloc.prices)
        DS.bulk_union(demand_vectors)
        # Add vertex set and bidder j label as a keylist.
        for vx_set in DS.vertex_sets():
            keylists.append((vx_set, j))
    return keylists
    
def derived_graph(alloc):
    """Computes a derived graph and returns adjacency lists for the link
    goods and key lists.
    """
    # Compute all key lists
    keylists = get_keylists(alloc)
    # Compute number of keylists.
    keylist_len = len(keylists)
    # Count number of keylists in which each good appears.
    good_counter = {i: 0 for i in alloc.goods}
    for k in range(keylist_len):
        for good in keylists[k][0]:
            good_counter[good] += 1    
    # Initialise adjacency dict (neighbours) for the link goods and key lists
    # In the neighbours dict, link goods are represented by their good number
    # and the k-th keylist is represented by the number alloc.n+k.
    neighbours = {i: [] for i in alloc.goods if good_counter[i] > 1}
    neighbours.update({k: [] for k in range(alloc.n, alloc.n + keylist_len)})
    # Determine the neighbour lists for link goods and key lists.
    for k in range(keylist_len):
        for good in keylists[k][0]:
            if good_counter[good] > 1:  # good is a link good
                # Add edge between good and keylist (=alloc.n+k)
                neighbours[good].append(alloc.n+k)
                neighbours[alloc.n+k].append(good)
    return neighbours, keylists

def find_params(alloc):
    # Get derived graph
    neighbours, keylists = derived_graph(alloc)
    # Step 2a: If a key list is isolated or has only one link good, return it
    for k in range(alloc.n, alloc.n+len(keylists)):
        if len(neighbours[k]) == 0:
            return (keylists[k-alloc.n], None)
        elif len(neighbours[k]) == 1:
            i = neighbours[k][0]
            return (keylists[k-alloc.n], i)
    # Step 2b: Take a walk through the derived graph to find a cycle
    prev_vx, curr_vx, next_vx = None, None, next(iter(neighbours))
    visited_vxs = set()
    curr_vx_is_linkgood = False
    while next_vx not in visited_vxs:  # walk until we find a cycle
        # Take a step
        prev_vx = curr_vx
        curr_vx = next_vx
        visited_vxs.add(curr_vx)
        curr_vx_is_linkgood = not curr_vx_is_linkgood
        # Pick next vertex to visit
        next_vx = neighbours[curr_vx][0]
        if prev_vx == next_vx:  # we don't want to walk backwards
            next_vx = neighbours[curr_vx][1]
    if  curr_vx_is_linkgood:
        i, j = curr_vx, keylists[next_vx-alloc.n][1]
    else:
        i, j = next_vx, keylists[curr_vx-alloc.n][1]
    return ((None, j), i)

def allocate(alloc, test=False):
    """Implements the main routine ALLOCATE from BKGL.
    
    IMPORTANT:
    Assumes that the supply alloc.residual is demanded at prices alloc.p.
    
    Parameters
    ----------
    alloc: AllocationProblem
        Allocation problem instance
    test: bool
        Set to False by default. If True, the function returns how many times
        procedures 1 to 3 were called.

    Returns
    -------
    list or (list, int, int, int):
        If test=False, return value is a list of bundles that are
        allocated to the bidders. If test=True, the number of times that
        procedures 1 to 3 were called is also returned.
    
    """
    # Initialise counters for procedures 1,2 and 3
    proc1 = 0
    proc2 = 0
    proc3 = 0
    
    # Main loop
    while True:
        # call Procedure 1
        proc1 += 1
        procedure1(alloc)
        # if bid lists are empty:
        if all(not alloc.bidlists[j].size for j in alloc.bidders):
            if test:
                return alloc.partial, proc1, proc2, proc3
            else:
                return alloc.partial
        else:  # there are more bids
            (I, j), i = find_params(alloc)
            if I:  # found a key list (I,j) with at most one link good i
                proc2 += 1
                procedure2(alloc, I, j, i)
            else:  # there is no key list with at most one link good
                proc3 += 1
                procedure3(alloc, i, j)

def _is_negative_H(alloc, p, i, bidder):
    """Checks whether H^p_i is negative or not (BGKL terminology)."""
    # Run through all bids of bidder and increment total_weight
    total_weight = 0
    bidlist = alloc.bidlists[bidder]
    weights = alloc.weights[bidder]
    for index in range(len(bidlist)):
        bid_vector = bidlist[index]
        bid_weight = weights[index]
        if bid_vector[i] == p[i] and np.all(p >= bid_vector):
            total_weight += bid_weight
    return total_weight < 0

def _is_negative_F(alloc, p, i, j, bidder):
    """Checks whether F^p_ij is negative or not (BGKL terminology)."""
    # Run through all bids of bidder and increment total_weight
    total_weight = 0
    bidlist = alloc.bidlists[bidder]
    weights = alloc.weights[bidder]
    for index in range(len(bidlist)):
        bid_vector = bidlist[index]
        bid_weight = weights[index]
        if (bid_vector[i] - p[i] == bid_vector[j] - p[j]
                and np.all((bid_vector - (bid_vector[i] - p[i])) <= p)):
            total_weight += bid_weight
    return total_weight < 0

def _md(alloc, U):
    """Returns the component-wise max vector over all the vectors in U.
    Input: set of numpy vectors.
    """
    result = next(iter(U))
    for vector in U:
        result = np.maximum(result, vector)
    return result

def _mdF(alloc, i, U):
    """Returns the 'diagonal' component-wise max."""
    return min((v[i] for v in U)) + _md(alloc, (v - v[i] for v in U))

def is_valid(alloc):
    """Checks whether the bids of bidder are valid (cf. Appendix of BGKL.)
    
    This code has not been optimised and is probably inefficient.
    """
    
    invalid_lists = set()
    for bidder in alloc.bidders:
        # Compute all negative bids
        
        neg_bids = (alloc.bidlists[bidder][i]
                    for i in range(len(alloc.bidlists[bidder]))
                    if alloc.weights[bidder][i] < 0)

        # Iterate through all subsets of neg_bids of cardinality up to n
        for s in range(1, alloc.n+1):
            for U in itertools.combinations(neg_bids, s):
                # Hod checks
                mdU = _md(alloc, U)
                for i in range(1,alloc.n):
                    # Check whether the vectors in U agree on i-th coord
                    val = next(iter(U))[i]
                    agree = all(map(lambda v: v[i] == val, U))
                    # Ensure that H^p_i is non-negative, with p = md(U)
                    if agree and _is_negative_H(alloc, mdU, i, bidder):
                        invalid_lists.add(bidder)
                        print("Hod H^{}_{} is negative".format(mdU, i))
                        continue

                # Flange checks
                for i, j in itertools.combinations(range(1, alloc.n), 2):
                    # Check if the vectors in U agree on diff b[i]-b[j]
                    vector = next(iter(U))
                    val = vector[i]-vector[j]
                    agree = all(map(lambda v: v[i]-v[j] == val, U))
                    if agree:
                        # Ensure that F^p_ij is non-negative
                        mdFi = _mdF(alloc, i, U)
                        if _is_negative_F(alloc, mdFi, i, j, bidder):
                            invalid_lists.add(bidder)
                            print("Flange F^{}_{}{} is negative".format(mdU,i,j))
                            continue
    return not invalid_lists

### AD HOC TEST ###
if __name__ == "__main__":
    filename = 'examples/example2.json'
    alloc = load_from_json(filename)
    print("supply: {}".format(alloc.residual))
    alloc.prices = min_up(alloc, long_step_routine="binarysearch")
    print("prices:", alloc.prices)
    print("allocation:", allocate(alloc))