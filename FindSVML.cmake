# Find the Intel SVML (Vectorization Support)
#
# Usage:
# find_package(SVML)

if(APPLE)
    set(SVML_LIB libsvml.dylib)
elseif(WIN32)
    set(SVML_LIB svml_dispmd.dll)
else()
    set(SVML_LIB libsvml.so)
endif()

find_path(SVML_ROOT_DIR
    ${SVML_LIB}
    PATHS
        $ENV{SVMLROOT}
        $ENV{HOME}/miniconda3/lib
        $ENV{USERPROFILE}/miniconda3/Library
        $ENV{CONDA}/envs/iArrayEnv/lib/intel64 # Azure pipelines
		$ENV{CONDA}/envs/iArrayEnv/lib # Azure pipelines
        /Users/vsts/.conda/envs/iArrayEnv # Azure pipelines
        C:/Miniconda/envs/iArrayEnv # Azure pipelines
)

foreach (LIB ${SVML_LIB})
    message(STATUS "Looking ${LIB} in: ${${LIB}_PATH}")
    find_library(${LIB}_PATH ${LIB} PATHS ${SVML_ROOT_DIR})
    if(${LIB}_PATH)
        set(SVML_LIBRARY ${SVML_LIBRARY} ${${LIB}_PATH})
        message(STATUS "Found SVML ${LIB} in: ${${LIB}_PATH}")
    else()
        message(FATAL_ERROR "Could not find ${LIB}")
    endif()
endforeach()

set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} ${SVML_LIBRARY})
