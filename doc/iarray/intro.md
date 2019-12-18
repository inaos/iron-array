ironArray Manual
================

## What it is

In short, ironArray is a multi-dimensional, compressed data container and computational engine optimized for big-data applications. Computations are carried out transparently on the compressed data either in memory or on fast SSD disks.

We do believe that Big Data handling can play better with our environment by making a more intelligent use of the energy, so we are making ironArray in such a way that it allows computations with large datasets to make a more effective use of modern, cost-effective multi-core CPUs and fast, local storage. This effectively enables reducing the need for large, expensive multi-node clusters, making system administration simpler while saving resources and power consumption.

For achieving this we are making use of some of the most advanced techniques and libraries.  Among these, we can find:

* Data partitioning: When dealing with large multidimensional arrays, sometimes you just want to access a tiny part of the original array.  ironArray allows you to retrieve the data in all those small chunks and present it back to you wrapped in a convenient container for later consumption.  As you can choose the different dimensions of every chunk that the dataset is made of, you can fine-tune these dimensions so as to optimize for different data access patterns.

* Compression: Many times you are dealing with data that can be compressed so when you enable this you can effectively deal with more data using the same resources.  For that, ironArray uses the high performance [C-Blosc2](https://github.com/Blosc/c-blosc2) compressor that works by splitting the datasets in small chunks and data blocks that are CPU-cache friendly in the sense that they can be compressed and decompressed very efficiently.

* LLVM support: The Python API (Java support is coming too) has support for converting Python code into optimized machine code via the [LLVM libraries](https://llvm.org) that runs very efficiently on modern processors.  This closes the gap between very productive languages, albeit slow, like Python and low-level languages like C or C++ that are well known for its speed, alowing to perform computations in ironArray containers at full speed.

* Intel MKL support: ironArray is all about speed and it shows because comes with integrated support for [Intel MKL](https://software.intel.com/en-us/mkl), one of the fastest and most used library for doing mathematical computations on Intel architectures.  MKL comes with highly optimized, threaded, and vectorized math functions that maximize performance on each processor family, even the more Intel/AMD modern architectures.

### Computation on compressed datasets

### Operations supported

* General arithmetic expressions
* Math functions in NumPy
* Linear algebra (matrix-matrix multiplication)
* Slicing

### Different memory layouts

* Compressed
* Backed by plain buffers

## Examples of use

* Create arrays
* Evaluation of general expressions on arrays
* Linear algebra
* Playing with compression

## Dealing with errors

* Checking errors
* Tracing errors

## C API

## Python API

## Java API

## Optimization tips

## Utilities