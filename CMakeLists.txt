#
# Copyright INAOS GmbH, Thalwil, 2018.
# Copyright Francesc Alted, 2018.
#
# All rights reserved
#
# This software is the confidential and proprietary information of INAOS GmbH
# and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
# Information and shall use it only in accordance with the terms of the license agreement.
#
cmake_minimum_required (VERSION 3.12)
project(iarray VERSION 0.1.3)

option(MULTITHREADING "Use multithreaded iarray" ON)

# Disable weird MSVC warnings
if (MSVC)
add_compile_options(/wd4204)
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/inac.cmake")
    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/inac.cmake")
        message(STATUS "Downloading inac.cmake from https://github.com/inaos/inac-cmake")
        file(DOWNLOAD "https://raw.githubusercontent.com/inaos/inac-cmake/0.3/inac.cmake"
                "${CMAKE_BINARY_DIR}/inac.cmake" STATUS DS)
        if(NOT "${DS}"  MATCHES "0;")
            file(REMOVE "${CMAKE_BINARY_DIR}/inac.cmake")
            message(FATAL_ERROR "Failed to download inac.cmake")
        endif()
    else()
        message(STATUS "Use local inac.cmake")
        configure_file("${CMAKE_SOURCE_DIR}/inac.cmake" "${CMAKE_BINARY_DIR}/inac.cmake" COPYONLY)
    endif()
endif()
include("${CMAKE_BINARY_DIR}/inac.cmake")

inac_add_dependency(inac "1.0.6")

inac_add_contrib_lib(tinyexpr)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

add_subdirectory(contribs/c-blosc2)
include_directories(contribs/c-blosc2/blosc)
set(BLOSC_LIB blosc2_static) # required for caterva tests
add_subdirectory(contribs/caterva)
include_directories(contribs/caterva/caterva)
add_subdirectory(contribs/minjugg)
include_directories(contribs/minjugg/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/contribs/c-blosc2/cmake)
#configure Intel MKL
find_package(MKL)
#configure Intel IPP
find_package(IPP)
#configure OMP
if (MULTITHREADING)
    find_package(OMP)
endif()

# Find the libraries that correspond to the LLVM components
# that we wish to use
execute_process(COMMAND "llvm-config" "--libs" OUTPUT_VARIABLE llvm_libs
                OUTPUT_STRIP_TRAILING_WHITESPACE)
set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} ${IPP_LIBRARIES} ${llvm_libs})

set(SRC ${CMAKE_SOURCE_DIR}/src)
file(GLOB src ${SRC_DIR}/*.c)
add_library(iarray_c ${src})
set_target_properties(
        iarray_c
        PROPERTIES
        COMPILE_DEFINITIONS INA_LIB=1)

include_directories("${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/inac"
        "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/src")


if (DO_COVERAGE)
    target_compile_options(iarray_c PRIVATE -fprofile-arcs -ftest-coverage)
    inac_merge_static_libs(iarrays iarray_c blosc2_static caterva_static ${INAC_LIBS})
    target_link_libraries(iarrays -fprofile-arcs)
else()
    inac_merge_static_libs(iarrays iarray_c blosc2_static caterva_static ${INAC_LIBS})
endif()

if (UNIX)
set(PLATFORM_LIBS ${PLATFORM_LIBS} pthread)
set(INAC_DEPENDENCY_LIBS minjugg ${INAC_DEPENDENCY_LIBS} blosc2_static ${IPP_LIBRARIES})
else()
set(INAC_DEPENDENCY_LIBS minjugg ${INAC_DEPENDENCY_LIBS})
endif()

inac_add_tests(iarrays)

if (DO_COVERAGE)
    include(CTest)
    enable_testing()
    add_test(NAME iarray_tests COMMAND tests)
    set_tests_properties(iarray_tests PROPERTIES LABELS "iarray")
endif()
#inac_add_benchmarks(iarrays)
inac_add_tools(iarrays)
inac_add_examples(iarrays)

inac_enable_trace(${CMAKE_BUILD_TYPE} 1)

# Playing with OpenMP (available mainly on GCC)
#if (UNIX AND NOT CMAKE_C_COMPILER_ID STREQUAL Clang)
#    set_property(
#            TARGET vectors
#            APPEND PROPERTY LINK_FLAGS "-fopenmp")
#endif ()

#if (MSVC)
#    install(TARGETS iarray
#            DESTINATION libs
#            COMPONENT libraries)
#else()
#    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libiarray.a
#            DESTINATION lib
#            COMPONENT libraries)
#endif()

# dist lib
add_definitions(-DINA_DLL)
add_definitions(-DINA_LIB)
add_library(iarray SHARED ${src})
target_link_libraries(iarray ${INAC_LIBS} blosc2_static caterva_static minjugg ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})

install(TARGETS iarray
        DESTINATION lib
        COMPONENT libraries)

inac_package()

# Copy test files
file(GLOB TESTS_DATA tests/data/*.iarray)

foreach (data ${TESTS_DATA})
    file(COPY ${data}
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
endforeach(data)
