# We start by creating an executable (will convert into a library later on)
add_executable(iarray iarray.c tinyexpr.c)
target_link_libraries(iarray blosc_shared)

# examples
add_executable(find_roots_schunk find_roots_schunk.c)
target_link_libraries(find_roots_schunk blosc_shared)
