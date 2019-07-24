# INAOS Common C Library

The INAOS Common C Library is a collection of header files and library routines 
used to implement common operations, such as input/output, character string 
handling, memory, error and event handling. This library is designed and 
optimized for singled threaded applications and is used as common base for 
all INAOS programs/libraries written in C. The library includes a built-in
LuaJIT engine (http://luajit.org)

High level objectives:

* Be minimal but complete (keep it simple)
* High Performance
* Low complexity
* Low resource consumption
* Ease of maintenance, testing and debugging
* Fully documented

# Building

## Building on Windows

Building on Windows requires some programs to be present on your system.

### Prerequisites

* [Visual Studio 2017 Community Edition][1]
* [CMake][> 3.7]
  * Use the binary installer you don't need to build from source
  * Make sure you add cmake to your PATH

### Build

* Open a Visual Studio command prompt
* Navigate to the INAC root folder
* Create a build directory e.g. 'build'
* `cd build`
* `cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug ..`
* `nmake`

## Building on Linux or OS X

### Prerequisites

* [GCC]
* [CMake][> 3.7]

### Build

* Create a build directory e.g. 'build'
* `cd build`
* `cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..`
* `make`

## Versioning and compatibility

Describes the approach for API and ABI compatibility when INAC is used as binary dependency.

### Decisions

* INAC is only distributed as static library
* Major releases break the compatibility
* Minor releases are backward compatible ABI and API wise
* Introduce compiler warnings to deprecate API's

## Dependency handling

* CMake based
* User must provide:
	* Repository: Local or URL (https://inaos.jfrog.io/inaos/webapp/...)
	* Package name
	* Version
* CMake macro will compile full path
* On Windows we use environment variable: ${ENV:VisualStudioVersion} for version info
* Unpacking should happen in user home e.g. %USERPROFILE%\.ina\cmake\...
* Transitive dependencies must be managed by the User. CMake will only throw and error if a library is not present


## Guidelines

### Resource management

### General function argument structure and behavior

* Output parameters to function are always after input parameters
* Structure/Object creation follows the pattern: Output parameter pointer-to-pointer. Functions  always return error-codes expect `free()` and `destroy()` which are void.
* Modules that require static resources provide `init()` and `destroy()` methods that have to be called by the user exactly once per process-lifetime. In debug mode the application will crash if `init()` or `destroy()` are executed more than once.
* For every `_new` function call in a module the user has to issue a `_free` function call.
* 

### Initialization and destruction

* For modules we should have only ```init()```and ```destroy()```. if we have a `init()` we must have a `destroy()` also.
* For allocation we should have only ```new()```and ```free()```. if we have a `new()` we must have a `free()`
* `init()` and `new()` must return always a valid RC for error handling
* `destroy()` and `free` must always succeed. They are therefore exceptionally defined as INA_API(void). 
* Errors during `destroy()` and `free()` must be logged
* Errors in `destroy()` and `free()` should abort execution.
* Output parameters for `init()` and `new()`  must be after input parameters.
* Output parameters must be properly initialized by `init()` and `new()`. They should be set to NULL on failure 
- `destroy()` and `free()` must set the pointer to NULL after releasing the resources.
- Every `destroy()` and `free()` must check if the pointer-pointer argument can be dereferenced and should return if the pointer is already NULL.

### Use of `const` keywords for pointer arguments in API functions

Usage should reflect reality. Use only if memory is readonly! E.g. a constant parameter. Casting a `const` argument into a non constant pointer should be strictly avoided in the implementation.


## Compile time configuration
 * `INA_TRACE_ENABLED`  : Enable/disable tracing. Default enabled.
 * `INA_TRACE_LEVEL`    : Set trace level (1-3). Default 1.
 * `INA_LOG_ENABLED`    : Enable/disable logging. Default enabled.
 * `INA_LOG_LEVEL`      : Set log level from 1 (errors) to 4(debug). 
                          Default 3 (info).


All constants are prefaced with `INA_` . Other identifiers are prefaced with
`ina_`. Type names are suffixed with `_t` and typedef so that the struct 
keyword need not be used.

## Starting to code

Start by including the INAOS library header in your code:

	#include <libinac/lib.h>;

## For library consumers
Initialize the library context as soon as possible:

	ina_init();

## For applications
For applications, initialize the application context with `ina_app_init()`. 
This must be  the first function call in your program.

	int main(int argc, char **argv,) 
	{
	    if (INA_SUCCEED(ina_app_init(argc, argv, NULL)) {
	        while (... {
	            ...
	        }
	    }
	    ina_exit(EXIT_SUCCESS);
	}

### Command line options
The library provides a builtin command line processor. For that purpose the 
`ina_app_init()` takes as fourth argument an array of `ina_opt_t` containing the 
command line options definition consisting in string, number and flag options. 
Use the designated macros to build the options array. Options are defined with 
a short, a long option name and a description. On string and number options a 
default value can de defined. 

* `INA_OPT_STRING(short,long,default,description)`: define a string option
* `INA_OPT_INT(short,long,default,description)`: define a int option
* `INA_OPT_FLAG(short,long,description)`: define a flag (default is false)


Use `INA_OPT(array-name)` to declare the option array:

	INA_OPTS(opt,
        INA_OPT_STRING("h", "host", NULL, "Hostname"),
        INA_OPT_INT("p", "port", 999, "Port"),
        INA_OPT_FLAG("k", "keep-alive", "Keep connection alive")));

Register and parse the options by passing the options array to `ina_app_init()`. 
The function fails with RC `INA_EOPT` if current command line options don't  
match with the registered definition and simple a usage screen will be printed 
out to the standard output.

	if (INA_SUCCEED(ina_app_init(argc, argv, opt)) {
	    while (... {
	            ...
	    }

To query a flag is whenever or not set use `ina_opt_isset()`:

	if (INA_SUCCEED(ina_opt_isset("keep-alive")) {

To get a int value use `ina_opt_get_int()`:
	
	int value = 0;
	ina_opt_get_int("port", &value);

To get a string value use `ina_opt_get_string()`:
	
	ina_str_t value = NULL;
	ina_opt_get_string("host", &value);

The command line options values are preserved for the until the application 
stops. 

# Portable Header
This library provides with his portable header (portable.h) macros, functions 
and types to help writing cross-platform libraries and applications.

## Compiler detection
A macro for each compiler will be defined if detected. The following compilers 
are  currently detected. 

* Borland C/C++: `INA_COMPILER_BORLAND`
* Compaq/DEC C/C++: `INA_COMPILER_DEC`
* Gnu GCC: `INA_COMPILER_GCC`
* Gnu GCC (Apple): `INA_COMPILER_APPLECC`
* HP-UX CC: `INA_COMPILER_HPCC`
* IBM C/C++: `INA_COMPILER_IBM`
* Intel C/C++: `INA_COMPILER_INTEL`
* MetroWerks CodeWarrior: `INA_COMPILER_METROWERKS`
* Microsoft Visual C++: `INA_COMPILER_MSVC`
* MIPSpro C/C++: `INA_COMPILER_MIPSPRO`
* Sun Pro: `INA_COMPILER_SUN`
* Watcom C/C++: `INA_COMPILER_WATCOM`

The name of detected compiler is defined by the `INA_COMPILER_STRING` macro. 
A warning is thrown by compile time if no compiler was detected.
 

## Target OS detection
Following target operating systems are currently supported and defined if 
detected.

* Linux: `INA_OS_LINUX`
* MacOS X: `INA_OS_OSX`
* Unix-like(generic): `INA_OS_UNIX`
* Windows: `INA_OS_WINDOWS`

The name of detected target os is defined by the `INA_OS_STRING` macro.


## Target CPU detection
Following target CPUs are currently supported and defined if detected. The name
of detected target CPU is defined by the `INA_CPU_STRING` macro.

* AMD x86-64: `INA_CPU_X86`, `INA_CPU_X86_64`
* ARM: `INA_CPU_STRONGARM`
* IA64: `INA_CPU_IA64`
* Intel 386+: `INA_CPU_X86`



## Misc macros

* INA_INLINE
* INA_CASSERT
* INA_LOW32
* INA_HIGH32
* INA_LOW
* INA_HIGH
* INA_TOWORD
* INA_SIZE_T_FMT
* INA_INT64_T_FMT
* INA_UINT64_T_FMT
* INA_ALIGNED
* INA_VSALIGNEDxxx
* INA_ALIGNEDxxx
* INA_PACKED
* INA_VS_BEGIN_PACK
* INA_VS_ENDPACK_PACK
* INA_PATH_SEPARATOR
* INA_ASM
* INA_VOLATILE
* INA_ATOMIC_INC
* INA_ATOMIC_DEC
* INA_ATOMIC_SWAP
* INA_LIKELY
* INA_UNLIKELY
* INA_RESTRICT
* INA_BSWAP_16
* INA_BSWAP_32
* INA_BSWAP_64
* INA_TLS
* INA_DISABLE_WARNING_CLANG
* INA_ENABLE_WARNING_CLANG
* INA_DISABLE_WARNING_GCC
* INA_ENABLE_WARNING_GCC
* INA_DISABLE_WARNING_MSVC
* INA_ENABLE_WARNING_MSVC

# API Reference

## Library Version
The INAOS Common C Library version is of the form A.B.C, where A is the major 
version, B is the minor version and C is the patch version. If the patch
version is zero, it's omitted from the version string, i.e. the version string 
is just A.B.
When a new release only fixes bugs and doesn't add new features or 
functionality, the patch version is incremented. When new features are added
in a backwards compatible way, the minor version is incremented and the patch
version is set to zero. When there are backwards incompatible changes, the 
major version is incremented and others are set to zero.

The following preprocessor constants specify the current version of the 
library:

`INA_MAJOR_VERSION, INA_MINOR_VERSION, INA_PATCH_VERSION`

Integers specifying the major, minor and patch versions, respectively.

`INA_VERSION`

A string representation of the current version, e.g. "1.2.1" or "1.3".

`INA_VERSION_HEX`

A 3-byte hexadecimal representation of the version, e.g. 0x010201 for version 
1.2.1 and 0x010300 for version 1.3. This is useful in numeric comparisons,
e.g.:

	#if INA_VERSION_HEX >= 0x010201
	/* Code specific to version 1.2.1 and above */
	#endif
