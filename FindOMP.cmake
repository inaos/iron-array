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

if (APPLE)
    set(OMP_ROOT_LIB lib/libiomp5.a)
elseif (WIN32)
    set(OMP_ROOT_LIB compiler/lib/intel64/libiomp5md.lib)
else()
    set(OMP_ROOT_LIB lib/intel64/libiomp5.a)
endif()

find_path(OMP_ROOT_DIR
    ${OMP_ROOT_LIB}
    PATHS
        $ENV{OMPROOT}
        /opt/intel/compilers_and_libraries/linux
        /opt/intel/compilers_and_libraries/mac
        "C:/IntelSWTools/compilers_and_libraries/windows"
        "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows"
        $ENV{HOME}/miniconda3
        $ENV{USERPROFILE}/miniconda3/Library
        "C:/Miniconda37-x64/Library" # Making AppVeyor happy
        $ENV{CONDA}/envs/iArrayEnv # Azure pipelines
        /Users/vsts/.conda/envs/iArrayEnv # Azure pipelines
        C:/Miniconda/envs/iArrayEnv # Azure pipelines
)

if(APPLE)
    set(OMP_SEARCH_LIB libiomp5.dylib)
    set(OMP_LIBS libiomp5.dylib)
elseif(WIN32)
    set(OMP_SEARCH_LIB libiomp5md.lib)
    set(OMP_LIBS libiomp5md.lib)
else()
    set(OMP_SEARCH_LIB libiomp5.so)
    set(OMP_LIBS libiomp5.so)
endif()

find_path(OMP_LIB_SEARCHPATH
    ${OMP_SEARCH_LIB}
    PATHS
        ${OMP_ROOT_DIR}/lib/intel64
        ${OMP_ROOT_DIR}/lib
		${OMP_ROOT_DIR}/compiler/lib/intel64
)

foreach (LIB ${OMP_LIBS})
    find_library(${LIB}_PATH ${LIB} PATHS ${OMP_LIB_SEARCHPATH})
    if(${LIB}_PATH)
        set(OMP_LIBRARIES ${OMP_LIBRARIES} ${${LIB}_PATH})
        message(STATUS "Found OMP ${LIB} in: ${${LIB}_PATH}")
    else()
        message(STATUS "Could not find ${LIB}: disabling OMP")
    endif()
endforeach()

if(UNIX)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif()

set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} ${OMP_LIBRARIES})
