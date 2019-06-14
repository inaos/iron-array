# Build, Release and Continuous Integration

## Continuous Integration

### Decisions

* Solution future: Azure DevOps inaos.visualstudio.com
* Legacy AppVeyor: Migrate once everything else works on Azure DevOps

### Technologies

* C - existing
* Java - todo
* Python - todo

## Build

### 64bit vs. 32bit

**We only supporting 64bit builds and packages**

### Intel MKL handling

#### Decisions

**Version <= 1.0**

* Use static libraries except for OpenMP
* Use the multi-threaded library versions of MKL

**Version >= 1.0**

* Use static libraries
* Use sequential version
* Note: This depends on parallesim that is coarse-grained

### Tools

#### C

* CMake 
* INAC CMake utilities

#### Python

* Wheel
* CMake
* INAC CMake utilities

#### Java

* Maven3
* CMake
* INAC CMake utilities

## Packages / Artifacts

### C Library

#### General

* Debug package -> only for development
* Release package

#### Windows

Directory structure:
- bin (tools, utilities)
- include (iarray.h)
- lib (iarray.dll, iarray.lib, libomp5.lib, omp.dll, debug symbols)

Supported Compilers:
- Visual Studio 2015
- Visual Studio 2017
- Intel Compiler

#### Linux

Directory structure:
- bin (tools, utilities)
- include (iarray.h)
- lib (libiarray.so, libomp5.so, debug symbols)

Supported Compilers:
- GCC
- Clang / LLVM
- Intel Compiler

Supported C library:
- We'll use Docker on Azure-DevOps to emulate an "old" version of libc

#### OS X

**Note: We only support OS X for development**

Directory structure:
- bin (tools, utilities)
- include (iarray.h)
- lib (libiarray.dylib, libomp5.dylib, debug symbols)

Supported Compilers:
- Clang / LLVM

OS Version:
- We use whatever is provided by Azure DevOps

### Python Library

### Java Library

## Release

### Repository

#### Development

* JFrog Artefactory: https://inaos.jfrog.io

#### Commercial packages

* To be decided
