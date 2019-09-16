# Installation
We give brief instructions to set up a Python virtual environment, install the required dependencies and compile the Cython code.
Some steps are OS-dependent and given below.

## MacOS (and Linux)
These are brief instructions for macOS. They should also work for Linux systems.
Please contact edwinlock@gmail.com if you have any issues.

1. Install Python 3.5 or newer.
2. Get the files from the GitHub server:
$ git clone https://github.com/edwinlock/product-mix.git
3. Set up a virtual environment (venv):
$ cd product-mix
$ python -m venv venv
4. Activate virtual environment:
$ source venv/bin/activate
5. Install the dependencies:
$ pip install -r requirements.txt
6. Compile the derived graph Cython code:
$ cd disjointset
$ python setup.py build_ext --inplace

## Windows 10
Please contact simon.finster@econ.ox.ac.uk if you have any issues. (SIMON, DO YOU MIND?)

1. Install Python 3.5 or newer.
2. Install git, install pip, install virtualevn, install mock
3. Get the files from the GitHub server:
$ git clone https://github.com/edwinlock/product-mix.git
4. Set up a virtual environment (venv):
$ cd product-mix
% python -m virtualenv venv
5. Activate virtual environment:
$ venv\Scripts\activate.bat
6. Install the dependencies:
$ python -m pip install -r requirements.txt
7. Compile the derived graph Cython code:
$ cd disjointset
$ python setup.py build_ext --inplace