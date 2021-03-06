
#
# External dependencies
#

option(USE_NON_FREE_CODE "use non free keypoint and descriptor types in the library." OFF)

find_package(Eigen REQUIRED)

find_package(OpenCV 4.2.0 REQUIRED)

SET( Nanoflann_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nanoflann-1.1.1/include")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#
# Library name and options
#

# Target name
set(target matchinglib)

# Exit here if required dependencies are not met
message(STATUS "Lib ${target}")

set(MAINTARGET ${target})

add_subdirectory(thirdparty/gms-1.0)

add_subdirectory(similarity_search)

# Set multithreading support for ANNOY lib
add_compile_definitions(ANNOYLIB_MULTITHREADED_BUILD)

# Set API export file and macro
string(TOUPPER ${target} target_upper)
set(feature_file "include/${target}/${target}_features.h")
set(export_file  "include/${target}/${target}_api.h")
set(export_macro "${target_upper}_API")


#
# Sources
#

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/source")

file(GLOB_RECURSE header_files
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
)
set(headers
   "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/matchinglib.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/matchinglib_imagefeatures.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/matchinglib_matchers.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/matchinglib_correspondences.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/vfcMatches.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/glob_includes.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/gms.h"
)

set(sources
    ${source_path}/matchers.cpp
    ${source_path}/correspondences.cpp
    ${source_path}/features.cpp
    ${source_path}/match_statOptFlow.cpp
    ${source_path}/match_opticalflow.cpp
    ${source_path}/vfc.cpp
    ${source_path}/vfcMatches.cpp
    ${source_path}/ratioMatches_Flann.cpp
	${source_path}/gms.cpp

    ${source_path}/CascadeHash/BucketBuilder.cc
    ${source_path}/CascadeHash/CasHashMatcher.cc
    ${source_path}/CascadeHash/DataPreProcessor.cc
    ${source_path}/CascadeHash/HashConvertor.cc
    ${source_path}/CascadeHash/MatchPairLoader.cc

    ${source_path}/descriptor-RIFF/riff.cpp
    ${source_path}/descriptor-BOLD/bold.cpp
)

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
    ${header_files}
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
target_include_directories(${target}
    PRIVATE
    ${PROJECT_BINARY_DIR}/source/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    ${CMAKE_CURRENT_SOURCE_DIR}/include/${target}
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${Nanoflann_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gms-1.0/include

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIR}

    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    #$<INSTALL_INTERFACE:include>
	$<INSTALL_INTERFACE:${INSTALL_INCLUDE}/${target}/include> # for client in install mode
	$<INSTALL_INTERFACE:${INSTALL_INCLUDE}/${target}>
	$<INSTALL_INTERFACE:${INSTALL_LIB}/${target}>
)

#message(STATUS "=====NonMetricSpaceLib ${NonMetricSpaceLib_LIBRARIES}")
#
# Libraries
#
target_link_libraries(${target}
    PRIVATE
    non_metric_space_lib
	  gms-feature-matching

    PUBLIC
    ${DEFAULT_LIBRARIES}
    ${OpenCV_LIBS}
    #Eigen

    INTERFACE
)


#
# Compile definitions
#

target_compile_definitions(${target}
    PRIVATE
    $<$<BOOL:${USE_NON_FREE_CODE}>:USE_NON_FREE_CODE>

    PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:${target_upper}_STATIC_DEFINE>
    ${DEFAULT_COMPILE_DEFINITIONS}

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
    EXPORT  "${target}-export"            			COMPONENT dev
    RUNTIME DESTINATION ${INSTALL_BIN}/${target}    COMPONENT runtime
    LIBRARY DESTINATION ${INSTALL_SHARED}/${target} COMPONENT runtime
    ARCHIVE DESTINATION ${INSTALL_LIB}/${target}    COMPONENT dev
)

# Header files
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${INSTALL_INCLUDE}/${target}
    COMPONENT dev
)
install(DIRECTORY
		${Nanoflann_INCLUDE_DIR} DESTINATION ${INSTALL_INCLUDE}/${target}
		COMPONENT dev
		)

# Generated header files
install(DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/include/${target} DESTINATION ${INSTALL_INCLUDE}/${target}
    COMPONENT dev
)

# CMake config
install(EXPORT ${target}-export
    NAMESPACE   ${META_PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKE}/${target}
    COMPONENT   dev
)
