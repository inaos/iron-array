# Goals

Here we document the goals.

## Version 1

### Plain Buffers

* The performance should be same as when using MKL with C directly.
* The multi-core scaling should also be in-line with what is achievable with MKL and pure native code.

### Chunked containers (without compression)

This only makes sense for containers stored on disk, larger than memory. Otherwise it is advisable to use plain-buffers instead.

TODO: Need to profile and define it

### Compressed containers

* We should reach the performance of pure native code - on a reduction (e.g. sum) - when running on 16 CPU Cores
* All other performance goals will be defined later

### Parallelism

* For V1 we only support parallelism through the expression/computation-engine.
* For constructing containers we'll automatically benefit from the blosc paralellism
* No other functions will be parallelized for V1

# Performance Thoughts

This section lists different thoughts or tools that we may want to adopt for enhancing and monitoring the performance of IronArray.

## AirSpeed Velocity (ASV)

This tool (https://github.com/airspeed-velocity/asv/) allows for monitor performance of different functionality of a software in order to detect regressions as soon as possible.  However, this a Python tool, so it requires the Python wrapper for IronArray.

You can find an example for NumPy here: https://pv.github.io/numpy-bench/. We can follow a similar setup for the parameters that NumPy is using for the different functions.  For example, for matmul, we should be using the same shape and type for operands so that we can compare the performance of NumPy with respect to IronArray.
