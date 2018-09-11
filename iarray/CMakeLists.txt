# We start by creating an executable (will convert into a library later on)
add_executable(iarray iarray.c)
target_link_libraries(iarray blosc_shared)
