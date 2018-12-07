# Performance Thoughts

This document lists different thoughts or tools that we may want to adopt for enhancing and monitoring the performance of IronArray.

## AirSpeed Velocity (ASV)

This tool (https://github.com/airspeed-velocity/asv/) allows for monitor performance of different functionality of a software in order to detect regressions as soon as possible.  However, this a Python tool, so it requires the Python wrapper for IronArray.

You can find an example for NumPy here: https://pv.github.io/numpy-bench/. We can follow a similar setup for the parameters that NumPy is using for the different functions.  For example, for matmul, we should be using the same shape and type for operands so that we can compare the performance of NumPy with respect to IronArray.
