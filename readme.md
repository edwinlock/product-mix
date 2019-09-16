# About

This is an implementation of the algorithms developed in the working paper (BGKL) by Elizabeth Baldwin, Paul Goldberg,
Paul Klemperer and Edwin Lock available on the ArXiv at [URL]. The algorithms solve the Product-Mix Auction (originally
developed by Paul Klemperer, see https://www.nuffield.ox.ac.uk/users/klemperer/productmix.pdf) that uses the tropical
bidding language with positive *and* negative bids; that is, they find a competitive (Walrasian) equilibrium.


**Installation instructions for this implementation can be found [here](install.md).**

As described in BGKL, computing a competitive equilibrium can be separated into two parts:

1) Find the component-wise minimal market-clearing price using a steepest
descent approach. Both long-step methods described in the paper are
implemented.

2) Find an allocation of the supply (=target) bundle among the various bidders
so that each bidder receives a bundle they demand at the market-clearing price.
This is implemented according to algorithm described in the BGKL paper.

### A note on the encoding of prices and bundles
For an auction with n goods, prices and bundles of goods are represented as (n+1)-dimensional vectors, where the i-th
entry corresponds to the i-th good and the 0-th entry corresponds to a notional 'reject' good that is useful for
technical reasons (see BGKL). In particular, every price vector has an 0-th entry of value 0. Moreover, an allocation
of the target bundle among the bidders consists of a list containing a bundle vector for each bidder, and each vector's
0-th entry denotes how many notional 'reject' goods the bidder receives.

## Example Usage

### Initial
Activate the virtual environment and launch an interactive Python shell:

```console
$ cd ..path..to../product-mix
$ source venv/bin/activate
$ python
```

Import the product-mix package
```python
>>> import productmix as pm
```

### Solving the Product-Mix auction

Load an allocation problem from a file
```python
>>> alloc = pm.load_from_json('examples/example2.json')
```

Find a market-clearing price using unit step steepest descent
```python
>>> prices = pm.min_up(alloc, long_step_routine="")
```

Find market-clearing prices using long step steepest descent
```python
>>> prices = pm.min_up(alloc, long_step_routine="demandchange")
```
or
```python
>>> prices = pm.min_up(alloc, long_step_routine="binarysearch")
```

Print and set market-clearing prices in allocation problem object
```python
>>> print(prices)
[0. 2. 4.]
>>> alloc.prices = prices
```

Compute a valid allocation. Outputs a list of bundles.

**Note that running the `pm.allocate(alloc)` method has
the side effect that all bids in the allocation problem instance `alloc`
are deleted!**
```python
>>> allocation = pm.allocate(alloc)
>>> print(allocation)
[array([3., 3., 0.]), array([2., 4., 1.])]
```
Hence the first bidder is allocated 3 items of good 1 and no items of good 2,
while the second bidder is allocated 4 items of good 1 and one 1 item of good 2.


### Miscellaneous methods

Reload the allocation problem from a file
```python
>>> alloc = pm.load_from_json('example data/example2.json')
```

Check validity of bid lists
```python
>>> pm.is_valid(alloc)
True
```

Define some price vector `p` and compute market-clearing vector `prices`
```python
>>> import numpy as np
>>> p = np.array([0,1,1])
>>> prices = pm.min_up(alloc)  # Uses 'binary search' long step technique by default
```

Compute the *Lyapunov* function at price vectors `p` and `prices`
```python
>>> pm.lyapunov(alloc, p)
49.0
>>> pm.lyapunov(alloc, prices)
39.0
```

Get a demanded bundle at `p` and `prices` (not necessarily unique!)
```python
>>> pm.demanded_bundle(alloc, p)
array([0., 7., 6.])
>>> pm.demanded_bundle(alloc, prices)
array([5., 7., 1.])
```
