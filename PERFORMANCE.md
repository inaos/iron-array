# Goals

Here we document the goals.

## Version 1

### Plain Buffers

* The performance should be same as when using MKL with C directly.
* The scaling on the same node should also be in-line with what is achievable with MKL and pure native code.

### Chunked containers (without compression)

This only makes sense for containers stored on disk, larger than memory. Otherwise it is advisable to use plain-buffers instead.

TODO: Need to profile and define it

### Compressed containers

TODO: Need to profile and define it

# Performance Thoughts

This section lists different thoughts or tools that we may want to adopt for enhancing and monitoring the performance of IronArray.

## AirSpeed Velocity (ASV)

This tool (https://github.com/airspeed-velocity/asv/) allows for monitor performance of different functionality of a software in order to detect regressions as soon as possible.  However, this a Python tool, so it requires the Python wrapper for IronArray.

You can find an example for NumPy here: https://pv.github.io/numpy-bench/. We can follow a similar setup for the parameters that NumPy is using for the different functions.  For example, for matmul, we should be using the same shape and type for operands so that we can compare the performance of NumPy with respect to IronArray.
