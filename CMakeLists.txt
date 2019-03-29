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
project(iarray)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/inac.cmake")
    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/inac.cmake")
        message(STATUS "Downloading inac.cmake from https://github.com/inaos/inac-cmake")
        file(DOWNLOAD "https://raw.githubusercontent.com/inaos/inac-cmake/0.1/inac.cmake"
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

inac_add_dependency(inac "1.0.4")

inac_add_contrib_lib(tinyexpr)

add_subdirectory(contribs/c-blosc2)
include_directories(contribs/c-blosc2/blosc)
set(BLOSC_LIB blosc_static) # required for caterva tests
add_subdirectory(contribs/caterva)
include_directories(contribs/caterva/caterva)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
find_package(MKL)

set(SRC ${CMAKE_SOURCE_DIR}/src)
file(GLOB src ${SRC_DIR}/*.c)
add_library(iarray_c ${src})
set_target_properties(
        iarray_c
        PROPERTIES
        COMPILE_DEFINITIONS INA_LIB=1)

include_directories("${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/inac"
        "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/src")

inac_merge_static_libs(iarrays iarray_c blosc_static caterva ${INAC_LIBS})

if (UNIX)
set(PLATFORM_LIBS ${PLATFORM_LIBS} pthread)
set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} blosc_static)
endif()

inac_add_tests(iarrays)
#inac_add_benchmarks(iarrays)
inac_add_tools(iarrays)
#inac_add_examples(iarrays)

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
add_library(iarray SHARED ${src})
target_link_libraries(iarray ${INAC_LIBS} blosc_static caterva ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})

inac_package()

# Copy test files
file(GLOB TESTS_DATA tests/data/*.iarray)

foreach (data ${TESTS_DATA})
    file(COPY ${data}
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
endforeach(data)
