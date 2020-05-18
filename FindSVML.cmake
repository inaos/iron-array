# Find the Intel SVML (Vectorization Support)
#
# Usage:
# find_package(SVML)

if(APPLE)
    set(SVML_LIB libsvml.dylib)
elseif(WIN32)
    set(SVML_LIB svml_dispmd.dll)
else()
    set(SVML_LIB libsvml.so)
endif()

find_path(SVML_ROOT_DIR
    ${SVML_LIB}
    PATHS
        $ENV{SVMLROOT}
        $ENV{CONDA_PREFIX}/lib # conda environments are accessible here (including base)
        /opt/intel/compilers_and_libraries/linux/lib/intel64_lin # Intel ICC on Linux
	    /opt/intel/compilers_and_libraries/mac/lib/intel64_lin # Intel ICC on MacOS
)

foreach (LIB ${SVML_LIB})
    message(STATUS "Looking ${LIB} in: ${${LIB}_PATH}")
    find_file(${LIB}_PATH ${LIB} PATHS ${SVML_ROOT_DIR})
    if(${LIB}_PATH)
        if(WIN32)
		    set(SVM_LIBRARY_DLL ${${LIB}_PATH})
			message(STATUS "Debug: ${SVM_LIBRARY_DLL}")
			execute_process(COMMAND ${CMAKE_SOURCE_DIR}/scripts/dlltolib.bat "${SVM_LIBRARY_DLL}" ${CMAKE_SOURCE_DIR}/scripts/dlltolib.py ${CMAKE_BINARY_DIR})
			get_filename_component(SVML_LIBRARY_NAME ${SVM_LIBRARY_DLL} NAME)
			string(REPLACE ".dll" ".lib" SVML_LIBRARY_NEWNAME ${SVML_LIBRARY_NAME})
			set(SVML_LIBRARY ${CMAKE_BINARY_DIR}/${SVML_LIBRARY_NEWNAME})
			file(COPY "${SVM_LIBRARY_DLL}" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
		else(WIN32)
			set(SVML_LIBRARY ${SVML_LIBRARY} ${${LIB}_PATH})
		endif(WIN32)
        message(STATUS "Found SVML ${LIB} in: ${SVML_LIBRARY}")
    else()
        message(FATAL_ERROR "Could not find ${LIB}")
    endif()
endforeach()

if (NOT WIN32)
	# This is necessary at least on Linux and MacOS
	string(REPLACE "svml" "intlc" INTLC_LIBRARY ${SVML_LIBRARY})
        string(REPLACE ".so" ".so.5" INTLC_LIBRARY ${INTLC_LIBRARY})
endif()

set(INAC_DEPENDENCY_LIBS ${INAC_DEPENDENCY_LIBS} ${SVML_LIBRARY} ${INTLC_LIBRARY})
if(WIN32)
    set(INAC_DEPENDENCY_BINS ${SVM_LIBRARY_DLL})
else()
    set(INAC_DEPENDENCY_BINS ${SVML_LIBRARY} ${INTLC_LIBRARY})
endif()

