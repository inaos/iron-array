cmake_minimum_required(VERSION 2.8.10)
project(iarray)

add_subdirectory(c-blosc2)

set(LIBINAC ${CMAKE_SOURCE_DIR}/inac/libinac)
set(SRC ${CMAKE_SOURCE_DIR}/src)

include_directories("${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/inac" "${CMAKE_SOURCE_DIR}")

# We start by creating an executable (will convert into a library later on)
add_executable(iarray ${SRC}/iarray.c ${SRC}/tinyexpr.c ${LIBINAC}/libinac.c)
target_link_libraries(iarray blosc_shared)

# examples
add_executable(find_roots_schunk ${SRC}/find_roots_schunk.c)
target_link_libraries(find_roots_schunk blosc_shared)

add_executable(vectors ${SRC}/iarray.c ${SRC}/tinyexpr.c ${LIBINAC}/libinac.c ${SRC}/vectors.c)
target_link_libraries(vectors blosc_shared)

add_executable(test ${SRC}/test.c)
add_executable(test2 ${SRC}/test2.c)
