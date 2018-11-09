file(GLOB SOURCES test_*.c)

foreach (source ${SOURCES})
    get_filename_component(target ${source} NAME_WE)
    add_executable(${target} ${CMAKE_SOURCE_DIR}/tests/main.c ${target}.c)
    target_link_libraries(${target} caterva iarray blosc ${MKL_LIBRARIES} ${INAC_DEPENDENCY_LIBS} )
    add_test(NAME ${target} COMMAND ${target})
endforeach (source)
