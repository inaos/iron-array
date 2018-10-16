#
# Copyright INAOS GmbH, Thalwil, 2018. All rights reserved
#
# This software is the confidential and proprietary information of INAOS GmbH
# ("Confidential Information"). You shall not disclose such Confidential
# Information and shall use it only in accordance with the terms of the
# license agreement you entered into with INAOS GmbH.
#
cmake_minimum_required (VERSION 3.12)
project(iarray)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/inac.cmake")
    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/inac.cmake")
        message(STATUS "Downloading inac.cmake from https://github.com/inaos/inac-cmake")
        file(DOWNLOAD "https://raw.githubusercontent.com/inaos/inac-cmake/0.1.0/inac.cmake"
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

set(LIBINAC ${CMAKE_SOURCE_DIR}/inac/libinac)
set(SRC ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(contribs/c-blosc2)
include_directories(contribs/c-blosc2/blosc)

file(GLOB src ${SRC_DIR}/*.c)
add_library(iarray_c ${src})
set_target_properties(
        iarray_c
        PROPERTIES
        COMPILE_DEFINITIONS INA_LIB=1)

include_directories("${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/inac"
        "${CMAKE_SOURCE_DIR}" )

#inac_merge_static_libs(iarray iarray_c blosc_static ${INAC_LIBS})

#inac_add_tests(iarray)
#inac_add_benchmarks(iarray)
#inac_add_tools(iarray)
#inac_add_examples(iarray)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
find_package(MKL)

set(BENCH ${CMAKE_SOURCE_DIR}/bench)
set(TESTS ${CMAKE_SOURCE_DIR}/tests)
# Playing with OpenMP (available mainly on GCC)
if (UNIX AND NOT CMAKE_C_COMPILER_ID STREQUAL Clang)
    set_property(
            TARGET vectors
            APPEND PROPERTY LINK_FLAGS "-fopenmp")
endif ()
add_executable(vectors ${BENCH}/vectors.c)
add_executable(vectors-float ${BENCH}/vectors-float.c)

message(STATUS "DEBUG3: ${INAC_DEPENDENCY_LIBS}")

target_link_libraries(vectors LINK_PUBLIC iarray_c blosc_static tinyexpr.lib ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})
target_link_libraries(vectors-float LINK_PUBLIC iarray_c blosc_static tinyexpr.lib ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})

add_executable(test_eval_chunk ${TESTS}/test_eval_chunk.c)
target_link_libraries(test_eval_chunk LINK_PUBLIC iarray_c blosc_static tinyexpr.lib ${INAC_DEPENDENCY_LIBS} ${PLATFORM_LIBS})

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
