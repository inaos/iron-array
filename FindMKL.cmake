# Find the Intel MKL (Math Kernel Library)
#
# MKL_FOUND - System has MKL
# MKL_INCLUDE_DIRS - MKL include files directories
# MKL_LIBRARIES - The MKL libraries
#
# The environment variable MKLROOT is used to find the installation location.
# If the environment variable is not set we'll look for it in the default installation locations.
#
# Usage:
#
# find_package(MKL)
# if(MKL_FOUND)
#     target_link_libraries(TARGET ${MKL_LIBRARIES})
# endif()

# Currently we take a couple of assumptions:
#
# 1. We only use the sequential version of the MKL
# 2. We only use 64bit
#

find_path(MKL_ROOT_DIR
    include/mkl.h
    PATHS
        $ENV{MKLROOT}
        /opt/intel/compilers_and_libraries/linux/mkl
        /opt/intel/compilers_and_libraries/mac/mkl
        "C:/IntelSWTools/compilers_and_libraries/windows/mkl/"
		"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl"
)

find_path(MKL_INCLUDE_DIR
        mkl.h
        PATHS
        ${MKL_ROOT_DIR}/include
        )

if(WIN32)
    set(MKL_SEARCH_LIB mkl_core.lib)
    set(MKL_LIBS mkl_intel_lp64.lib mkl_core.lib mkl_sequential.lib)
elseif(APPLE)
    set(MKL_SEARCH_LIB libmkl_core.a)
    set(MKL_LIBS libmkl_intel_lp64.a libmkl_core.a libmkl_sequential.a)
else() # Linux
    set(MKL_SEARCH_LIB libmkl_core.a)
    set(MKL_LIBS libmkl_intel_lp64.a libmkl_core.a libmkl_sequential.a)
endif()


find_path(MKL_LIB_SEARCHPATH
    ${MKL_SEARCH_LIB}
    PATHS
        ${MKL_ROOT_DIR}/lib/intel64
        ${MKL_ROOT_DIR}/lib
)

foreach (LIB ${MKL_LIBS})
    find_library(${LIB}_PATH ${LIB} PATHS ${MKL_LIB_SEARCHPATH})
    if(${LIB}_PATH)
        set(MKL_LIBRARIES ${MKL_LIBRARIES} ${${LIB}_PATH})
    else()
        message(STATUS "Could not find ${LIB}: disabling MKL")
    endif()
endforeach()

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
include_directories(${MKL_INCLUDE_DIRS})
set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} ${MKL_LIBRARIES})
