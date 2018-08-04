# IronTensor

This is a working document to brainstorm and define: scope, features and priorities of the project.

**Executive goal: Save money on infrastructure by increasing efficiency (less Hardware, more performance)** 

Goal of the project is to develop a math library written in C that operates on **compressed** data-structures.

## Features

### High-level API's

* Java
	* Vectorized Streams?
* Python
* R (later)
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

## Priorities and Outlook

1. Create a minimum viable product (MVP), that supports: 
	- Compression and vector quantization
	- Vector, Matrix, Cube data-types
	- BLAS, some basic LAPACK (e.g. transpose), some vector math and statistics


Rest to be defined.

## Input and Links:

* https://github.com/Blosc
* https://github.com/dblalock/bolt
* https://github.com/astojanov/Clover
* https://arxiv.org/pdf/1706.10283
* http://bitmagic.io/sparse-vector-search.html?cn=ZmxleGlibGVfcmVjcw%3D%3D&refsrc=email
* https://www.anandtech.com/show/13047/ngd-launches-catalina-2-programmable-ssds

