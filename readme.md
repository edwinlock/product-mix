# About

This is an implementation of the algorithms developed in the working paper by Elizabeth Baldwin, Paul Goldberg,
Paul Klemperer and Edwin Lock available on the ArXiv at [URL]. The algorithms solve the Product-Mix Auction originally developed by Paul Klemperer (cf. https://www.nuffield.ox.ac.uk/users/klemperer/productmix.pdf); that is, they find a competitive (Walrasian) equilibrium. Computing the equilibrium can be separated into two parts:

1) Find the component-wise minimal market-clearing price using a steepest
descent approach. Both long-step methods described in the paper are
implemented.

2) Find an allocation of the supply (=target) bundle among the various bidders
so that each bidder receives a bundle they demand at the market-clearing price.

Installation instructions can be found in install.md.

## Example Usage

Activate the virtual environment and launch an interactive Python shell:
```console
$ cd ..path../product-mix
'''
```console
$ source venv/bin/activate
$ python
'''

Import the product-mix package
>>> import productmix as pm

Load an allocation problem from a file
>>> alloc = pm.load_from_json('example data/example1.json')

Find a market-clearing price using unit step steepest descent
>>> prices = pm.min_up(alloc, long_step_routine="")

Find market-clearing prices using long step steepest descent
>>> prices = pm.min_up(alloc, long_step_routine="demandchange")
or
>>> prices = pm.min_up(alloc, long_step_routine="binarysearch")

Set prices
>>> alloc.prices = prices

Find an allocation
>>> allocation = pm.allocate(alloc)

Check validity of bid lists
>>> pm.is_valid(alloc)

Compute Lyapunov function at prices p
>>> pm.lyapunov(alloc, p)

Get a demanded bundle at prices p (not necessarily unique!)
>>> pm.demanded_bundle(alloc, p)