# -*- coding: utf-8 -*-
#
# Copyright 2019 by Edwin Lock
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

"""We denote by BGKL the working paper by Elizabeth Baldwin, Paul Goldberg,
Paul Klemperer and Edwin Lock.

This is an implementation of the Product-Mix Auction of Paul Klemperer and the
algorithms presented in BGKL to solve the auction, consisting of two parts:

1) Find the component-wise minimal market-clearing price using a steepest
descent approach. Both long-step methods described in the paper are
implemented.

2) Find an allocation of the supply (=target) bundle among the various bidders
so that each bidder receives a bundle they demand at the market-clearing price.

### Usage ###

Load an allocation problem from a file
> alloc = load_from_json('examples/example2.json')

Find a market-clearing price using unit step steepest descent
> prices = min_up(alloc, long_step_routine="")

Find market-clearing prices using long step steepest descent
> prices = min_up(alloc, long_step_routine="demandchange")
or
> prices = min_up(alloc, long_step_routine="binarysearch")

Find an allocation
> allocation = allocate(alloc)

Check validity of bid lists
> is_valid(alloc)

Compute Lyapunov function at prices p
> lyapunov(alloc, p)

Get a demanded bundle at prices p (not necessarily unique!)
> demanded_bundle(alloc, p)
"""

import numpy as np
from fujishige_wolfe import fw_sfm, naive_sfm
from disjointset import DisjointSet
import itertools
import json
import datetime

# Data structures
# bidlists - a Python array of 2D numpy arrays with the following axes
# First axis: j-th bid,
# Second axis: k-th entry in bid
# weights - 2D numpy array.  First axis: weights of i-th bidlist,
#                                Second axis: weight of the j-th bid
# prices - 1D numpy array
# residual - 2D numpy array


class AllocationProblem:
    def __init__(self, goods, bidders, eps = 0.0625):
        self.n = goods+1  # add the reject good
        self.m = bidders
        self.goods = range(self.n)  # including reject good
        self.bidders = range(self.m)
        self.eps = eps
        self.bidlists = []
        self.weights = []
        self.prices = np.zeros(self.n, dtype=float)
        self.residual = np.zeros(self.n)
        self.partial = [np.zeros(self.n) for _ in self.bidders]

def add_bids():
    """TO DO"""
    pass

def load_from_json(filename):
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
    
def save_to_json(alloc, filename = None):
    """Store the allocation problem encoded in the bid lists as a JSON file."""
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

def demanded_goods(bid_vector, p):
    """Determine and return the set of goods demanded by bid_vector at p.
    Returns a Python set.
    """
    diff = bid_vector - p
    max_utility = diff.max()
    return np.nonzero(diff == max_utility)[0]

def get_demand_vectors(bidlist, p):
    diff = bidlist - p
    utility = diff.max(axis=1, keepdims=1)
    # initialise demand_vectors
    demand_vectors = np.empty(shape=diff.shape, dtype = np.int8)
    # populate
    np.equal(diff, utility, out = demand_vectors)
    return demand_vectors

def shift(bidlist, i, eps):
    """Shifts all vectors in bidlist by eps in direction i."""
    n = bidlist.shape[1]
    char_vector = np.zeros(n)
    char_vector[i] += eps
    bidlist += char_vector

def project(bidlists, prices):
    """Projects bid vectors of all bidders according to prices."""
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

def lyapunov(alloc, prices):
    """Implements the Lyapunov function. Efficient implementation.
    """
    output = np.dot(alloc.residual, prices)
    for j in alloc.bidders:        
        # get the utility for each bid
        utilities = (alloc.bidlists[j]-prices).max(axis=1)
        output += np.dot(utilities, alloc.weights[j])
    return output

def demanded_bundle(alloc, prices):
    """Returns a bundle that is demanded at p by accepting, for each bid, the
    smallest good demanded at p.
    """
    bundle = np.zeros(alloc.n, dtype=float)
    for j in alloc.bidders:
        for i in range(len(alloc.bidlists[j])):
            bid_vector = alloc.bidlists[j][i]
            smallest_good = np.argmax(bid_vector - prices)
            bundle[smallest_good] += alloc.weights[j][i]
    return bundle

def min_up(alloc, long_step_routine="binarysearch", prices = None, test=False):
    """Implements the MinUp algorithm from BGKL. This is a steepest
    descent algorithm starting from the origin; uses SFM to find the
    direction.
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
            if test:
                return p, steps
            else:
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
    Input: price vector p, direction vector d (param S is not used).
    Output: positive integer.
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
    This corresponds to Procedure 1 in BGKL.
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
    """Allocate unambiguous marginal bids to the relevant bidder. This
    corresponds to Procedure 2 in BGKL.
    Input: key list (I,j), possible link good i (may be None).
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
    """Returns the function g_dash."""
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
    allocation problem. Takes as input a link good i and bidder j such that
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

def find_params(alloc):
    """Implements FindParams from BGKL."""
    # Step 1: get derived graph
    link_neighbours, key_neighbours, keylists = derived_graph(alloc)
    keylist_len = len(keylists)

    # Step 2a: If some key list is isolated, return it
    for k in key_neighbours:
        if len(key_neighbours[k]) == 0:
            return (keylists[k], None)
        elif len(key_neighbours[k]) == 1:
            i = key_neighbours[k][0]
            return (keylists[k], i)

    # Step 2b: take a walk between the two vertex sets
    current_vx = iter(link_neighbours).next()  # pick starting link good
    prev_vx = None  # we store the previous vx so we don't walk backwards
    next_vx = None
    current_vx_is_linkgood = True

    # keep track of link goods and key lists visited
    keylists_visited = set()
    linkgoods_visited = set()
    linkgoods_visited.add(current_vx)

    # perform walk
    while True:
        if current_vx_is_linkgood:
            # Determine next vx in walk, which is a neighbouring keylist
            next_vx = link_neighbours[current_vx][0]
            if prev_vx == next_vx:  # we don't want to walk backwards
                next_vx = link_neighbours[current_vx][1]

            # Check whether we have found a cycle
            if next_vx in keylists_visited:  # cycle found
                # Return the last key list and link good visited
                j = keylists[next_vx][1]
                i = current_vx
                return ((None, j), i)
            else:  # no cycle found
                # Make step by updating variables
                prev_vx = current_vx
                current_vx = next_vx
                keylists_visited.add(current_vx)

        else:  # current_vx is a keylist
            next_vx = key_neighbours[current_vx][0]
            if prev_vx == next_vx:  # we don't want to walk backwards
                next_vx = key_neighbours[current_vx][1]

            # Check whether we have found a cycle
            if next_vx in linkgoods_visited:  # we have found a cycle
                # Return the last key list and link good visited
                i = next_vx
                j = keylists[current_vx][1]
                return ((None, j), i)
            else:  # no cycle found
                prev_vx = current_vx
                current_vx = next_vx
                linkgoods_visited.add(current_vx)

        # Update current vx type
        current_vx_is_linkgood = not current_vx_is_linkgood

def print_debug(alloc):
    for j in alloc.bidders:
        print 'Bidder {}'.format(j)
        print "Bid vectors\n", alloc.bidlists[j]
        print "Bid weights\n", alloc.weights[j]
        print "Partial allocation\n", alloc.partial[j]
    print "Residual\n", alloc.residual

def allocate(alloc, test=False):
    """Implements the main routine ALLOCATE from BKGL. IMPORTANT:
    Assumes that the supply alloc.residual is demanded at prices alloc.p.
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

def derived_graph(alloc):
    """Computes a derived graph and returns adjacency lists for the link
    goods and key lists.
    """
    # Compute all key lists
    keylists = []
    for j in alloc.bidders:
        # Generate the vertex sets corresponding to the connected
        # components of the graph induced by all j-labelled edges in the
        # marginal bids graph. Uses disjoint set data structure implemented by 
        # the DisjointSet class (see disjointset.pyx).
        DS = DisjointSet(alloc.n)    
        demand_vectors = get_demand_vectors(alloc.bidlists[j], alloc.prices)
        DS.bulk_union(demand_vectors)
        # Add vertex set and bidder j label as a keylist.
        for vx_set in DS.vertex_sets():
            keylists.append((vx_set, j))

    # Compute number of keylists.
    keylist_len = len(keylists)
    # Count number of keylists in which each good appears.
    good_counter = {i: 0 for i in alloc.goods}
    for k in range(keylist_len):
        for good in keylists[k][0]:
            good_counter[good] += 1

    # Initialise the adjacency dicts for the link goods.
    key_neighbours = {k: [] for k in range(keylist_len)}
    link_neighbours = {i: [] for i in alloc.goods if good_counter[i] > 1}

    # Determine the neighbour lists for link goods and key lists.
    for k in range(keylist_len):
        for good in keylists[k][0]:
            if good_counter[good] > 1:  # good is a link good
                # Add edge between good and k
                key_neighbours[k].append(good)
                link_neighbours[good].append(k)

    return link_neighbours, key_neighbours, keylists

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
    result = iter(U).next()
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
                    val = iter(U).next()[i]
                    agree = all(map(lambda v: v[i] == val, U))
                    # Ensure that H^p_i is non-negative, with p = md(U)
                    if agree and _is_negative_H(alloc, mdU, i, bidder):
                        invalid_lists.add(bidder)
                        print "Hod H^{}_{} is negative".format(mdU, i)
                        continue

                # Flange checks
                for i, j in itertools.combinations(range(1, alloc.n), 2):
                    # Check if the vectors in U agree on diff b[i]-b[j]
                    vector = iter(U).next()
                    val = vector[i]-vector[j]
                    agree = all(map(lambda v: v[i]-v[j] == val, U))
                    if agree:
                        # Ensure that F^p_ij is non-negative
                        mdFi = _mdF(alloc, i, U)
                        if _is_negative_F(alloc, mdFi, i, j, bidder):
                            invalid_lists.add(bidder)
                            print "Flange F^{}_{}{} is negative".format(mdU,i,j)
                            continue
    return not invalid_lists

### AD HOC TESTS ###

if __name__ == "__main__":
    # filename = "experiments/experiment9/experiment-10-5-100-380-39.json"
    # filename = "experiments/experiment8/experiment-2-5-100-500-48.json"
    filename = 'test/new_test.json'
    alloc = load_from_json(filename)
    print "supply: {}".format(alloc.residual)
    alloc.prices = min_up(alloc, long_step_routine="binarysearch")
    print allocate(alloc)