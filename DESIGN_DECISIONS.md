## Operations on Matrices and Vectors

* We follow the numpy convention (e.g. matmul) and leverage the dshape information to determine whether we have to issue a BLAS level 2 oder level 3 function.
* In addition we add a 'hint' that enables the user to indicate a special shape of the matrix (e.g. Symmetric or Triangular)
