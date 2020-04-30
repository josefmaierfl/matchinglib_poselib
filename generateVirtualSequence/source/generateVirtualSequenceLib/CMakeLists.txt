
#
# External dependencies
#

option(USE_NON_FREE_CODE "use non free keypoint and descriptor types in the OpenCV library." ON)
option(USE_MANUAL_ANNOTATION "Manual annotate GTM for detected inaccuracies."         OFF)

find_package(Eigen REQUIRED)

find_package(OpenCV 4.2.0 REQUIRED)

if(USE_MANUAL_ANNOTATION)
	if(UNIX)
		find_package(Qt5 COMPONENTS Widgets REQUIRED)
	endif()
endif()

find_package(PCL 1.9.1 REQUIRED)
#message(STATUS ${PCL_VERSION})
if(PCL_FOUND)
	message(STATUS "Found PCL lib!")
	message(STATUS "PCL Version: ${PCL_VERSION}")
endif()

SET( Nanoflann_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nanoflann-1.3.0/include")

#
# Library name and options
#

# Target name
set(target generateVirtualSequenceLib)

# Exit here if required dependencies are not met
message(STATUS "Lib ${target}")

# Set API export file and macro
string(TOUPPER ${target} target_upper)
set(feature_file "include/${target}/${target}_features.h")
set(export_file  "include/${target}/${target}_api.h")
set(export_macro "${target_upper}_API")


#
# Sources
#

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")#/${target}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/source")

#file(GLOB_RECURSE header_files
#  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
#  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
#  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
#)
set(headers
#    ${include_path}/test.h
	#${include_path}/getStereoCameraExtr.h
	#${include_path}/generateSequence.h
	#${include_path}/helper_funcs.h
)

set(sources
    ${source_path}/io_data.cpp
	${source_path}/getStereoCameraExtr.cpp
	${source_path}/generateSequence.cpp
	${source_path}/helper_funcs.cpp
	${source_path}/polygon_helper.c
	${source_path}/side_funcs.cpp
	${source_path}/generateMatches.cpp
	${source_path}/imgFeatures.cpp
	${source_path}/descriptor-RIFF/riff.cpp
	${source_path}/descriptor-BOLD/bold.cpp
	${source_path}/GTM/base_matcher.cpp
	${source_path}/GTM/inscribeRectangle.cpp
	${source_path}/GTM/loadGTMfiles.cpp)

# Group source files
set(header_group "Header Files (API)")
set(source_group "Source Files")
source_group_by_path(${include_path} "\\\\.h$|\\\\.hpp$"
    ${header_group} ${headers})
source_group_by_path(${source_path}  "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$"
    ${source_group} ${sources})


#
# Create library
#

# Build library
add_library(${target}
    ${sources}
    ${headers}
    #${header_files}
)

# Create namespaced alias
add_library(${META_PROJECT_NAME}::${target} ALIAS ${target})

# Export library for downstream projects
export(TARGETS ${target} NAMESPACE ${META_PROJECT_NAME}:: FILE ${PROJECT_BINARY_DIR}/cmake/${target}/${target}-export.cmake)

# Create feature detection header
# Compilers: https://cmake.org/cmake/help/v3.1/variable/CMAKE_LANG_COMPILER_ID.html#variable:CMAKE_%3CLANG%3E_COMPILER_ID
# Feature: https://cmake.org/cmake/help/v3.1/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html
write_compiler_detection_header(
    FILE ${feature_file}
    PREFIX ${target_upper}
    COMPILERS AppleClang Clang GNU MSVC
    FEATURES cxx_alignas cxx_alignof cxx_constexpr cxx_final cxx_noexcept cxx_nullptr cxx_sizeof_member cxx_thread_local
)

# Create API export header
generate_export_header(${target}
    EXPORT_FILE_NAME  ${export_file}
    EXPORT_MACRO_NAME ${export_macro}
)


#
# Project options
#

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)


#
# Include directories
#
message(${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${target}
    PRIVATE
    ${PROJECT_BINARY_DIR}/source/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include

	${Nanoflann_INCLUDE_DIR}

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    #${FLANN_INCLUDE_DIRS}
    #${FLANN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIR}
    #${EIGEN3_INCLUDE_DIR}
	#${MATCHINGLIB_INCLUDE_DIRS}
	#${PFELIB_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}

    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


#
# Libraries
#
target_link_libraries(${target}
    PRIVATE


    PUBLIC
    ${DEFAULT_LIBRARIES}
    #${FLANN_LIBRARIES}
    ${OpenCV_LIBS}
	#${MATCHINGLIB_LIBRARIES}

	${PCL_LIBRARY_DIRS}
	${PCL_COMMON_LIBRARIES}
	${PCL_OCTREE_LIBRARIES}
	${PCL_IO_LIBRARIES}
	${PCL_KDTREE_LIBRARIES}
	${PCL_SEARCH_LIBRARIES}
	${PCL_FILTERS_LIBRARIES}
	${PCL_GEOMETRY_LIBRARIES}
	${PCL_SEGMENTATION_LIBRARIES}
	#${PCL_STEREO_LIBRARIES}
	${PCL_VISUALIZATION_LIBRARIES}
	#Eigen

    INTERFACE
)
if(USE_MANUAL_ANNOTATION)
	if(UNIX)
		target_link_libraries(${target}
			PUBLIC
			Qt5::Widgets
		)
	endif()
endif()
#
# Compile definitions
#

target_compile_definitions(${target}
    PRIVATE
    $<$<BOOL:${USE_NON_FREE_CODE}>:USE_NON_FREE_CODE>

    PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:${target_upper}_STATIC_DEFINE>
    ${DEFAULT_COMPILE_DEFINITIONS}
    #${FLANN_DEFINITIONS}
	${PCL_DEFINITIONS}

    INTERFACE
)


#
# Compile options
#

target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}


    INTERFACE
)


#
# Linker options
#

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
)


#
# Deployment
#

# Library
install(TARGETS ${target}
    EXPORT  "${target}-export"            COMPONENT dev
    RUNTIME DESTINATION ${INSTALL_BIN}    COMPONENT runtime
    LIBRARY DESTINATION ${INSTALL_SHARED} COMPONENT runtime
    ARCHIVE DESTINATION ${INSTALL_LIB}    COMPONENT dev
)

# Header files
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${target} DESTINATION ${INSTALL_INCLUDE}
    COMPONENT dev
)

# Generated header files
install(DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/include/${target} DESTINATION ${INSTALL_INCLUDE}
    COMPONENT dev
)

# CMake config
install(EXPORT ${target}-export
    NAMESPACE   ${META_PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKE}/${target}
    COMPONENT   dev
)
