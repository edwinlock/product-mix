from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="disjointset",
    ext_modules=cythonize('disjointset.pyx'),
    include_dirs=[numpy.get_include()]
)