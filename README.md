# Robust numerical utils

This package is designed to help with some numerics in Python.
It consists of simple routines which are generically more robust (and also slower) than their SciPy analogs.


#### Contents of the package

- `bisect_ignore_nan.py` contains a standard bisect routine upgraded for working 
with functions that sometimes return NaN.

- `fmin_robust.py` is designed for finding a local minimum of a scalar function of
two variables. The scalar function is allowed to return NaN and/or go to minus infinity.
Although this routine is not all-powerful, I found that it is significantly more reliable
(and slower) that the analogous SciPy routines.

Apart from these, there are additional functions.

- `utils.py` contains simple utils that allow tracking e.g. progress of a `for`-loop,
find zero crossings of a 1d array, find a pair of close elements in an array.  

- `tests.py` contains tests and usage examples for (most of) 
the functions listed above.


#### Usage

Syntax and usage example are written in the description of each individual function,
as well as in the file `tests.py`.



#### History and acknowledgements

- 2020.01 Started development in the course of the projects 
supported by 
the Russian Foundation for Basic Research according to the research projects
No. 18-32-00015 and No. 18-02-00048, 
and by the Foundation for the Advancement of Theoretical Physics and Mathematics "BASIS" according to
the research project No. 19-1-5-106-1.
This code was used in the numerics for arXiv:1911.08328 [hep-th].
- 2020.09 A few minor improvements