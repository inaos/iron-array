# Find the Intel SVML (Vectorization Support)
#
# Usage:
# find_package(SVML)
#
# Note: this will emit a FATAL_ERROR in case the SVML library is not found (i.e. hard dependency)

# Use the fantastic ctypes shared library mechanism for finding *any* library on the system
EXEC_PROGRAM("python" ARGS "-c 'from ctypes.util import find_library; print(find_library(\"svml\"))'"
             OUTPUT_VARIABLE SVML_LIB RETURN_VALUE RET_VAL)

if(${RET_VAL} EQUAL 0 AND EXISTS ${SVML_LIB})
    set(SVML_LIBRARY ${SVML_LIB})
    message(STATUS "Found SVML lib iat: ${SVML_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find SVML lib")
endif()

set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} ${SVML_LIBRARY})
