import random
import productmix as pm
import numpy as np
from fujishige_wolfe import fw_sfm

"""
This script generates lists of dot bids with negative and positive weights
and a bundle x that is demanded at a price vector p. By default we fix p to be
M/2 * e^[n], where M is the size of the bounding cube and choose x so that p is
the minimum market-clearing price vector. The dot bids are generated in such a
way that all bids are marginal between goods at p so that it is hard to
allocate x to the various bidders.

Usage: call generate_bidlist(n, m, M, q, p = None)
Parameters:
n: number of goods
m: number of bidders
M: length of the sides of the cube that all bids must lie in.
q: quantity parameter
p: price vector at which all bids are marginal.

The procedure for generating a bid list:

Repeat the following q times.

First pick a subset S of goods from {1, ..., n-1} on which the new bid will be
marginal at prices p. Then flip a coin to decide whether the bid will be
positive or negative.

If it's heads, generate a positive bid of weight 1 that is marginal on S at p
and we are done. If it's tails, generate a negative bid of weight -1 marginal
on S at p. Additionally, generate 3 positive bids that 'cover' the negative
bid. Now we have a valid bid list. Note that some coordinates are chosen
deterministically and some probabilistically; for details, see the code.

While the whole procedure is probabilistic due to the coin toss, we generate
2q positive bids and q/2 negative bids in expectation.
"""

def marginal_vector_on(demanded_goods, p, M):
    """Generate a bid that is marginal on goods in the demanded set at p.
    All the bid entries are integers between 0 and M and may be chosen
    probabilistically.
    """
    n = len(p)
    vector = np.zeros(n, dtype=float)
    # Fix the utility that bid derives at p
    if 0 in demanded_goods:  # the reject good is demanded
        utility = 0
    else:
        utility = random.randint(0, M - max([p[i] for i in range(n)]))
    # Determine values for bid vector entries so that demanded goods achieve
    # the utility fixed above and the other goods achieve less utility.
    for i in range(1,n):
        if i in demanded_goods:
            vector[i] = p[i] + utility
        else:
            vector[i] = random.randint(0, p[i] + utility - 1)
    return vector

def random_subset(n):
    """Returns a random subset with cardinality at least two of the elements
    {0, ..., n}."""
    I = set()
    while len(I) <= 1:
        I = set(i for i in range(n+1) if random.randint(0,1))
    return I

def prefix_elem(vector, val = 0):
    prefix = np.array([0])
    return np.concatenate((prefix, vector), axis=0)

def random_demanded_bundle(vectors, weights, prices):
    bundle = np.zeros(len(prices))
    permutation = np.random.permutation(len(prices))
    for i in range(len(vectors)):
        diff = vectors[i] - prices
        good = np.argmax(diff[permutation])
        bundle[good] += weights[i]
    return bundle

def get_bidlist(n, M, q, p):
    """Generate a bid list for a single bidder using the parameters given."""
    bidlist = []
    weights = []
    assert len(p) == n+1
    dem_bundle = np.zeros(n+1)
    for _ in range(q):  # repeat q times
        temp_vectors = []
        temp_weights = []
        # Pick a set of marginal goods of cardinality >=2 uniformly at random.
        I = random_subset(n+1)
        # Decide whether to generate a positive or a negative bid.
        flip = random.randint(0,1)
        if flip:  # generate a positive bid marginal on I at p
            temp_vectors.append(marginal_vector_on(I, p, M))
            temp_weights.append(1)
        else:
            # generate one negative bid marginal on I at p and three covering
            # positive bids
            neg_vector = marginal_vector_on(I, p, M)
            temp_vectors.append(neg_vector)
            temp_weights.append(-1)
            # Create two 'covering' positive bids that make all 'hods' positive
            for i in random.sample(range(n+1),2):
                pos_vector = neg_vector.copy()
                pos_vector[i] = random.randint(0, neg_vector[i])
                temp_vectors.append(pos_vector)
                temp_weights.append(1)
            # Create a single positive bid that makes all flanges positive
            offset = np.random.randint(0, M-neg_vector.max()+1)
            temp_vectors.append(neg_vector+offset)
            temp_weights.append(1)
        # Add the four bids to the bidlist and weights
        bidlist += temp_vectors
        weights += temp_weights
        # Compute demanded bundle
        dem_bundle += random_demanded_bundle(temp_vectors, temp_weights, p)
    return np.row_stack(bidlist), np.array(weights, dtype=int), dem_bundle
    
def get_allocation_problem(n, m, M, q):
    """Returns an object of type AllocationProblem with the supply set to the
    bundle demanded at p.
    """
    alloc = pm.AllocationProblem(n, m)
    prices = np.array([0] + [M/2]*n, dtype=float)
    alloc.prices = prices
    demanded_bundle = np.zeros(n+1)
    while True:
        alloc.bidlists = []
        alloc.weights = []
        # Add bid lists
        for _ in range(m):
            bidlist, weights, bidder_demand = get_bidlist(n, M, q, prices)
            alloc.bidlists.append(bidlist)
            alloc.weights.append(weights)
            demanded_bundle += bidder_demand
        # Determine demanded bundle at p and set it as supply/residual
        alloc.residual = demanded_bundle
        # Check that alloc.prices is minimum market-clearing price
        if check_min_prices(alloc, M):
            break
        print 'Price is not minimal, retrying.'
    return alloc

def set_market_clearing_prices(alloc, M):
    alloc.prices = np.array([0] + [M/2]*(alloc.n-1), dtype=float)

def check_min_prices(alloc, M):
    """ Checks for allocation problem alloc whether alloc.prices is the
    minimum market-clearing price vector.
    """
    p = alloc.prices - np.array([0] + [1]*(alloc.n-1))
    g_dash = pm.get_g_dash(alloc, p)
    # Find minimal submodular minimiser of g_dash
    S = fw_sfm(alloc.n-1, g_dash)
    # Check that S contains all goods 1, ..., alloc.n-1
    return len(S)==alloc.n-1
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
    'Create and save a hard allocation problem and save it to a file.\
    Generates n + q(n+2)/2 positive bids and q/2 negative bids in expectation.')
    # Add arguments for input parameters n, m, M and q
    parser.add_argument('-n', required=True,
                        help='number of goods', type=int)
    parser.add_argument('-m', required=True,
                        help='number of bidders', type=int)
    parser.add_argument('-M', required=True,
                        help='upper bound on the value of bid entries',
                        type=int)
    parser.add_argument('-q', required=True,
                        help='size parameter', type=int)
    parser.add_argument('-f', required=True,
                        help='output file name', type=str)
    args = parser.parse_args()

    alloc = get_allocation_problem(args.n, args.m, args.M, args.q)
    pm.save_to_json(alloc, args.f)