from distutils.core import setup
from Cython.Build import cythonize

setup(name="disjointset", ext_modules=cythonize('disjointset.pyx'),)