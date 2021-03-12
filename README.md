[![Azure CI](https://inaos.visualstudio.com/iron-array/_apis/build/status/inaos.iron-array?branchName=develop)](https://inaos.visualstudio.com/iron-array/_build/latest?definitionId=6&branchName=develop)

# iron-array

## Setup

### Clone

    git clone --recurse-submodules git@github.com:inaos/iron-array.git

### Git commit-hooks

Execute the following commands:

      cp conf/pre-commit .git/hooks/


## Build

We use inac cmake build-system in combination with different libraries which can be installed using
miniconda3.  In particular, one can install LLVM from the numba channel, and MKL and SVML from Intel
channel in a cross-platform portable way with:

    $ conda install -c numba 'llvmdev<11'
    $ conda install -c intel mkl-include
    $ conda install -c intel mkl-static
    $ conda install -c intel icc_rt    # SVML

Beware: currently ironArray only supports LLVM 10.  Also, we strongly suggest to use the numba
channel with conda/mamba for installing the LLVM package.   In particular, we discourage the use
of native LLVM libraries in the system (either using `apt`, `brew` or any other packager), even
if they are installing LLVM 10 (the numba team seems to be doing a great job in packaging).

### Windows

* INAC build setup
    * Make sure that you have a configured repository.txt file in ~\.inaos\cmake
    * Also you'll need a directory ~\INAOS (can be empty)

* Create a build folder

         mkdir build
         cd build

* Invoke CMAKE, we have to define the generator as well as the build-type

         cmake -G"Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Debug ..
         cmake -G"Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

### Mac

* INAC build setup:
    * Make sure that you have a configured repository.txt file in ~/.inaos/cmake
    * Also you'll need a directory ~/INAOS (can be empty)

* Create a build folder:

         mkdir build
         cd build

* Invoke CMAKE, we have to define the build-type:

         cmake -DCMAKE_BUILD_TYPE=Debug ..
         cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..


### Linux

* INAC build setup
    * Make sure that you have a configured repository.txt file in ~/.inaos/cmake
    * Also you'll need a directory ~/INAOS (can be empty)
    
* MKL setup.  For Ubuntu machines, it is best to use Intel's Ubuntu repo (but you can
 use conda packages described above too):

         wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
         apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
         sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
         sudo apt-get update && sudo apt-get install intel-mkl-64bit-2019.X

* Create a build folder

         mkdir build
         cd build

* Invoke CMAKE, we have to define the build-type, but only two types are supported

         cmake -DCMAKE_BUILD_TYPE=Debug ..
         cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

* Some Linux machines (ClearLinux, Gentoo?) require the use of `llvm-config` utility.  You can enforce its use with `-DDISABLE_LLVM_CONFIG=False`::

         cmake -DCMAKE_BUILD_TYPE=Debug -DDISABLE_LLVM_CONFIG=False ..
         cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDISABLE_LLVM_CONFIG=False ..

### Tracing

Sometimes it is useful to activate the tracing mechanism for debugging purposes.  Example:

```
$ env INAC_TRACE='*' ./perf_vector_expression  -e 1 -E 2 -M 3 -t 10 -l 0 -c 9 -f 2
Time for computing and filling X values: 0.0523 s, 2918.5 MB/s
Time for compressing and *storing* X values: 0.106 s, 1434.1 MB/s
Compression for X values: 152.6 MB -> 11.2 MB (13.6x)
Time for computing and filling Y values: 0.0665 s, 2296.2 MB/s
Time for compressing and *storing* Y values: 0.135 s, 1130.3 MB/s
Compression for Y values: 152.6 MB -> 152.6 MB (1.0x)
[iarray.error] - Error compressing a blosc chunk /Users/faltet/inaos/iron-array/src/iarray_expression.c:853
Error during evaluation.  Giving up...
```

### Expressions

* For now only element-wise operations are supported in expressions.

* The iron-array library supports disabling the SVML optimization by setting a `DISABLE_SVML` environment variable to *any* value.  This can be useful for debugging purposes.
