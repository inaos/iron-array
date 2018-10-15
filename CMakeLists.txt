cmake_minimum_required(VERSION 2.8.10)
project(iarray)

add_subdirectory(c-blosc2)

set(LIBINAC ${CMAKE_SOURCE_DIR}/inac/libinac)
set(SRC ${CMAKE_SOURCE_DIR}/src)
set(BENCH ${CMAKE_SOURCE_DIR}/bench)

include_directories("${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/inac" "${CMAKE_SOURCE_DIR}")

# We start by creating an executable (will convert into a library later on)
#add_executable(iarray ${SRC}/iarray.c ${SRC}/tinyexpr.c ${LIBINAC}/libinac.c)
#target_link_libraries(iarray blosc_shared)

# examples
#add_executable(find_roots_schunk ${SRC}/find_roots_schunk.c)
#target_link_libraries(find_roots_schunk blosc_shared)

add_library(iarray ${SRC}/iarray.c ${SRC}/tinyexpr.c ${LIBINAC}/libinac.c)
## FIXME: attempt at using a shared library for both iarray and c-blosc2
#add_library(iarray SHARED ${SRC}/iarray.c ${SRC}/tinyexpr.c ${LIBINAC}/libinac.c)
#set_target_properties(iarray PROPERTIES PUBLIC_HEADER include/iarray.h)
#target_include_directories(iarray PRIVATE include)

# Playing with OpenMP (available mainly on GCC)
if (UNIX AND NOT CMAKE_C_COMPILER_ID STREQUAL Clang)
    set_property(
            TARGET vectors
            APPEND PROPERTY LINK_FLAGS "-fopenmp")
endif ()
add_executable(vectors ${BENCH}/vectors.c)
add_executable(vectors-float ${BENCH}/vectors-float.c)

target_link_libraries(vectors LINK_PUBLIC iarray blosc_shared)
target_link_libraries(vectors-float LINK_PUBLIC iarray blosc_shared)

add_executable(test ${BENCH}/test.c)
add_executable(test2 ${BENCH}/test2.c)
