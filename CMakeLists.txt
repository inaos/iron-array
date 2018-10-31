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

inac_add_dependency(inac "1.0.0")
inac_add_contrib_lib(tinyexpr)

add_subdirectory(contribs/c-blosc2)
include_directories(contribs/c-blosc2/blosc)
add_subdirectory(contribs/caterva)
include_directories(contribs/caterva/caterva)

set(SRC ${CMAKE_SOURCE_DIR}/src)
file(GLOB src ${SRC_DIR}/*.c)
add_library(iarray_c ${src})
set_target_properties(
        iarray_c
        PROPERTIES
        COMPILE_DEFINITIONS INA_LIB=1)

include_directories("${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/inac"
        "${CMAKE_SOURCE_DIR}" )

inac_merge_static_libs(iarray iarray_c blosc_static ${INAC_LIBS})

if (UNIX)
set(PLATFORM_LIBS ${PLATFORM_LIBS} pthread)
set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} blosc_static)
endif()

inac_add_tests(iarray)
#inac_add_benchmarks(iarray)
#inac_add_tools(iarray)
#inac_add_examples(iarray)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
find_package(MKL)

set(BENCH ${CMAKE_SOURCE_DIR}/bench)
# Playing with OpenMP (available mainly on GCC)
#if (UNIX AND NOT CMAKE_C_COMPILER_ID STREQUAL Clang)
#    set_property(
#            TARGET vectors
#            APPEND PROPERTY LINK_FLAGS "-fopenmp")
#endif ()
add_executable(vectors ${BENCH}/vectors.c)
add_executable(vectors-float ${BENCH}/vectors-float.c)
add_executable(vectors-frame ${BENCH}/vectors-frame.c)
add_executable(vectors-caterva ${BENCH}/vectors-caterva.c)
add_executable(gemm-caterva ${BENCH}/gemm-caterva.c)

target_link_libraries(vectors LINK_PUBLIC iarray_c blosc_static caterva tinyexpr ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})
target_link_libraries(vectors-float LINK_PUBLIC iarray_c blosc_static caterva tinyexpr ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})
target_link_libraries(vectors-frame LINK_PUBLIC iarray_c blosc_static caterva tinyexpr ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})
target_link_libraries(vectors-caterva LINK_PUBLIC iarray_c blosc_static caterva tinyexpr ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})
target_link_libraries(gemm-caterva LINK_PUBLIC iarray_c blosc_static caterva tinyexpr ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})

add_executable(test ${BENCH}/test.c)
add_executable(test2 ${BENCH}/test2.c)

#if (MSVC)
#    install(TARGETS iarray
#            DESTINATION libs
#            COMPONENT libraries)
#else()
#    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libiarray.a
#            DESTINATION lib
#            COMPONENT libraries)
#endif()

inac_package()
