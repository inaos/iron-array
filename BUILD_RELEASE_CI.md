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

Supported Compilers (this requires a different package per compiler):
- Visual Studio 2015
- Visual Studio 2017
- Intel Compiler

Archive format:
- ZIP file

#### Linux

Directory structure:
- bin (tools, utilities)
- include (iarray.h)
- lib (libiarray.so, libomp5.so, debug symbols)

Supported C library:
- We'll use Docker on Azure-DevOps to emulate an "old" version of libc

Archive format:
- TAR.GZ / (maybe TAR.XZ)

#### OS X

**Note: We only support OS X for development**

Directory structure:
- bin (tools, utilities)
- include (iarray.h)
- lib (libiarray.dylib, libomp5.dylib, debug symbols)

OS Version:
- We use whatever is provided by Azure DevOps

Archive format:
- TAR.XZ

### Python Library

* We'll use the standard 'Wheels' format to package for:
  * Windows
  * Linux
  * OS X
* We'll produce 1 single 'wheel' per platform that contains:
  * Iron-Array C library (so, dll, dylib)
  * OpenMP (so, dll, dylib)
  * Python Extensions
  * Compiled Python Code

### Java Library

* We'll support Java > 8
* We'll use 1 single JAR file that contains all the binaries for the supported platforms
* We'll bundle:
  * Iron-Array C library
  * OpenMP C library
  * Iron-Array JNI 
  * Iron-Array compiled Java-Code

## Release Procedure

* Make sure that the new release is correctly written in main CMakeLists.txt.
* Merge `develop` branch into `master` via a PR that should be approved at least by 2 people.
* Change into the `master` branch and tag the release using the next convention:

  $ git tag vX.Y.Z -m"Tagging vX.Y.Z release"

* Manually trigger the release pipeline in Azure web interface.
* Check that the new release artifacts appear in the repository (see below).

### Repository

#### Development

* JFrog Artefactory: https://inaos.jfrog.io

#### Commercial packages

* To be decided
