## Operations on Matrices and Vectors

* We follow the numpy convention (e.g. matmul) and leverage the dtshape information to determine whether we have to issue a BLAS level 2 oder level 3 function.
* In addition we add a 'hint' that enables the user to indicate a special shape of the matrix (e.g. Symmetric or Triangular)

## Blosc usage / settings

* We observed problems in the Windows realloc implementation which is used heavily for the blosc-frame implementation when the frame is not on disk. Therefore we decided to use the blosc-frame ONLY for persistent containers. In any case its much better not the rely on realloc optimization (finding enough contiguous space), but rather use an implemenation (super-chunks) which does not require contiguous memory.

## Benchmarks

### General

* Whenever we create a benchmark to bench against a competitor we should do it in a seprate repository
* For each high-level benchmark (Java / Python) we want a low-level benchmark in C. 

### Directories

* bench: The bench folder is meant for benchmarks running in the inac bench framework
* tools: The tools folder is meant for arbitrary tools to develop test or operate iron-array. Thus the manual benchmarks are prefixed with 'perf_'
