import numpy as np
cimport numpy as np
cimport cython

cdef class DisjointSet:
	"""Implements the disjoint-set data structure, see
	https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
	We assume that the n elements of the set are the integers 0 to n-1.
	"""
	cdef int n
	cdef int [:] parent
	cdef int [:] rank
	cdef int [:] size
	
	def __init__(self, n):
		"""Initialise the data structure: add n elements to the tree, each
		with rank 0 and the parent pointer pointing at itself.
		"""
		self.n = n
		self.parent = np.arange(n, dtype=np.int32)
		self.rank = np.zeros(n, dtype=np.int32)
		self.size = np.ones(n, dtype=np.int32)
	
	@cython.boundscheck(False)
	cdef int find(self, int x):
		"""Find the representative element of the set containing x. Compress
		the active path using the 'halving' technique.
		"""
		while x != self.parent[x]:
			# path halving
			self.parent[x] = self.parent[self.parent[x]]
			x = self.parent[x]
		return x

	@cython.boundscheck(False)
	cdef void union(self, int x, int y):
		"""
		Merges the two sets containing x and y, updating size and rank.
		"""
		xroot, yroot = self.find(x), self.find(y)

		# check that the roots aren't identical.
		if xroot == yroot:
			return

		# ensure that xroot.rank <= yroot.rank
		if self.rank[xroot] > self.rank[yroot]:
			xroot, yroot = yroot, xroot

		# set yroot as the parent of xroot
		self.parent[xroot] = yroot
		if self.rank[xroot] == self.rank[yroot]:
			# update rank if necessary
			self.rank[yroot] += 1

		# update size of subtree rooted at yroot
		self.size[yroot] += self.size[xroot]

	def vertex_sets(self):
		"""Returns a list of all vertex sets (Python set) of size at least 2.
		"""
		# maps representative vxs to the index of their part in partition list
		repsToIndex = [None for i in range(self.n)]
		partition = []

		for elem in range(self.n):
			root = self.find(elem)
			partitionIndex = repsToIndex[root]
			if partitionIndex is None and self.size[root] > 1:
				# partition does not contain a vx of the component of elem
				partition.append(set([elem]))  # append a set containing elem
				repsToIndex[root] = len(partition)-1  # update repsToIndex
			if partitionIndex is not None:
				# partition already contains a vx of the component of elem
				partition[partitionIndex].add(elem)
		return partition

	@cython.boundscheck(False)
	def bulk_union(self, np.int8_t [:,:] demand_vectors):
		cdef long [:] marginals
		cdef long [:] demand
		cdef long i, j
		cdef long first_dem, other_good

		for i in range(len(demand_vectors)):
			# get first demanded good in demand_vectors[i]
			first_dem = 0
			while not demand_vectors[i, first_dem]:
				first_dem += 1
			for other_dem in range(first_dem+1, len(demand_vectors[0])):
				if demand_vectors[i, other_dem]:
					self.union(first_dem, other_dem)