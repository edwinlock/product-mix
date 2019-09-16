# About

This is an implementation of the algorithms developed in the working paper (BGKL) by Elizabeth Baldwin, Paul Goldberg,
Paul Klemperer and Edwin Lock available on the ArXiv at [URL]. The algorithms solve the Product-Mix Auction originally
developed by Paul Klemperer (cf. https://www.nuffield.ox.ac.uk/users/klemperer/productmix.pdf); that is, they find a
competitive (Walrasian) equilibrium. Installation instructions for this implementation can be found [here](install.md).


As described in the above paper, computing the equilibrium can be separated into two parts:

1) Find the component-wise minimal market-clearing price using a steepest
descent approach. Both long-step methods described in the paper are
implemented.

2) Find an allocation of the supply (=target) bundle among the various bidders
so that each bidder receives a bundle they demand at the market-clearing price.
This is implemented according to algorithm described in the BGKL paper.


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
>>> alloc = pm.load_from_json('example data/example1.json')
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
>>> alloc.prices = prices
```

Compute a valid allocation. Note that running the `pm.allocate(alloc)` method has
the side effect that all bids in the allocation problem instance `alloc`
are deleted.
```python
>>> allocation = pm.allocate(alloc)
>>> print(allocation)
```

### Miscellaneous methods

Check validity of bid lists
```python
>>> pm.is_valid(alloc)
```

Compute the Lyapunov function at prices p
```python
>>> p = np.array
>>> pm.lyapunov(alloc, p)
```

Get a demanded bundle at prices p (not necessarily unique!)
```python
>>> pm.demanded_bundle(alloc, p)
```
