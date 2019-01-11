# IronArray

This is a working document to brainstorm and define: scope, features and priorities of the project.

**Executive goal: Save money on infrastructure by increasing efficiency (less Hardware, more performance)** 

Goal of the project is to develop a math library written in C that operates on **compressed** data-structures.

## Features

### High-level API's

#### 1 Priority

* Java
* Python

#### 2 Priority

* R
* ...

### Supported Data-Types

Long term we want to support all native C data-types, but we implement in stages.

#### 1 Priority

* `double`
* `float`

#### 2 Priority

* `int32`
* `char` -> `bool` -> `int8`
* `int64`
* ...


### Supported Math Operations

* Try to support whatever Intel MKL supports, maybe look to IPP and DAAL
* We could also look at competitors
	* ND4J
	* JBlas
	* Numpy
	  * Broadcasting
	  * Tabluar data
	* SciPy
	* Armadillo
	* Eigen
* Linear algebra 
	* BLAS
	* LAPACK
	* ScaLAPACK
	* SparseBLAS
	* Solvers
* FFT
* Vector Statistics and Data Fitting
	* Random Number Generators
	* Probability Distributions
* Vector Math
	* VML
	* SVML

### Data-Structures

* Vector
* Matrix
* N-dimensional array (Tensor)

### Data-Compression

* Uncompressed (reference/baseline)
* Loss-less compression (leverage c-blosc2 for CPU)
* Lossy compression (for random-data); operate on compressed data (e.g. based on vector quantization)
* Custom operators on data-structures
	* Possibility to define custom operators (C based)
	* Offer auto-tuning tools (blocking, tiling etc.)

### Technical features

* Auto-tuning either offline or at run-time, need to investigate
	* Select the right compression algorithm (ZSTD dict?) 
	* Select compression level for the problem
	* Custom operators
		* Blocking/Tiling
		* Unrolling
		* ...

### Heterogeneous Hardware-Support

* CPU
* FGPA
	* Focus on Intel FPGA
* GPU
	* If at all focus on NVIDIA


### Persistence

* Optionally persistent (e.g. via mmap / lmdb.. nvram.. persistent memory)

![](irontensor_overview.jpg)

## Use-Cases

Due to the fact that everybody worries about machine-learning and artificial intelligence that space is very crowded. Therefore I'd like to start with more traditional use-cases.

1. **Linear algebra**: My main customer has lots of pricing and simulation engines. Older ones written in Java and Matlab, newer ones written in Python and R. Most of those applications make use of linear algebra one or the other way and most of the time the hot-spot's are related to linear algebra computations or other numeric code.
- Subset BLAS 1,2,3
- Subset LAPACK (e.g. Transponse)
- Only dense for first release

2. **Monte-Carlo simulations**: For Monte-Carlo simulations we could leverage vector quantization, where we operate on lossy data. I have a customer which has at least two applications which could benefit from this, given the precision loss is not too high.
- Probablility Distributions
- Vector quantization

3. Sliding window statistics 
- Pandas

4. Example from Spark
- Numerical 

However, linear algebra is as well at the core of machine-learning etc.. 
so in the future that could be an option too.

My client-base is the financial industry (banking, insurance) so I'll try to come-up with full use-cases and example data from that domain.

## Approach

Following the iterative approach how I would develop the project.

### Benchmark / Performance test-suite

First, create a benchmark / performance test-suite for the use-cases we want to support.
Benchmark against competitors. Following properties:

* Define and manage competitors
* Benchmark on private machines as well as on public cloud VM's (Azure and AWS) 
* Benchmark should be separate project so that we can open-source it independently and people can plugin or maintain their favorite competitor.
* Benchmark should be plug-able and language independent since we compare different language run-times
* For native code compile with latest GCC, Clang and Intel Compiler
* For Java use JDK8 and lasted GA .. e.g. JDK10
* For Python use latest default? version, Intel distribution for Python

### Dependencies and tools

Goal should be to have a minimal set of tools and dependencies that are required to start working with iron-tensor. Also in terms of development.

* INAC: INAOS C library providing OS abstraction, test, benchmark and build framework
* blosc: C library version that supports chunked compression
* CMake
* Intel MKL

### Develop API

Todo

### Develop CPU backend

Todo

## Commercial aspects

### Editions

* Open-Source (Community Edition); reduced features (CPU version, OpenBlas/Atlas and MKL)
* CPU, commercial - full features, only Intel MKL
* GPU, commercial
* FPGA, commercial

### Disclosures

* Include blosc boundaries
* Persistence
* ...

## Priorities and Outlook

1. Create a minimum viable product (MVP), that supports: 
	- Compression and vector quantization
	- Vector, Matrix, Cube data-types
	- BLAS, some basic LAPACK (e.g. transpose), some vector math and statistics


Rest to be defined.

## Implementation decisions for MVP

This section is to document design decisions.

### Python API

* Support expressions as strings
* No explicit syntactic sugar, will come later
* 

### Java API

* Fluent API with syntactic sugar -> avoid temporaries
* Support expression as string 
* The Java wrapper should create a string from the API calls in the first release

### C API

* Expression (string) based
* Byte-code will be considered in the second release



### Memory management

* We decided to use mmap allocated memory (linked to a file, for future persistence) to store the blosc chunks
* We plan to leverage inac mempool to manage the blosc chunks, however we need to make sure that the mempool does not allocate more memory than required.
* We decided to create a benchmark for this

### Array layout in memory

* Use row-major (C convention) - same as numpy
* Index order also same as numpy 


## Input and Links:

* https://github.com/Blosc
* https://github.com/dblalock/bolt
* https://github.com/astojanov/Clover
* https://arxiv.org/pdf/1706.10283
* http://bitmagic.io/sparse-vector-search.html?cn=ZmxleGlibGVfcmVjcw%3D%3D&refsrc=email
* https://www.anandtech.com/show/13047/ngd-launches-catalina-2-programmable-ssds
* https://software.intel.com/en-us/mkl-developer-reference-c-mkl-jit-create-gemm
* https://github.com/pytorch/glow/blob/405e632ef138f1d49db9c3181182f7efd837bccc/lib/Backends/CPU/libjit/libjit_matmul.cpp
* https://github.com/codeplea/tinyexpr
* http://www.singularsys.com/jep/
* https://github.com/topics/math-expressions
* https://github.com/topics/expression-evaluator?l=c
* https://github.com/killme2008/aviator
* https://github.com/cdman/fast-java-expr-eval
* https://github.com/AlaskanEmily/dcjit
* https://pypi.org/project/pytest-benchmark/
* https://www.gnu.org/software/libjit/
* https://www.gnu.org/software/lightning/
* https://corsix.github.io/dynasm-doc/index.html
* https://www.eclipse.org/omr/




