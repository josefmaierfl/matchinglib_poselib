
#
# External dependencies
#

option(USE_NON_FREE_CODE "use non free keypoint and descriptor types in the library." OFF)

find_package(Eigen REQUIRED)

find_package(OpenCV 4.8.0 REQUIRED)

get_filename_component(PARENT_DIR ../ ABSOLUTE)
SET( Nanoflann_INCLUDE_DIR "${PARENT_DIR}/matchinglib/thirdparty/nanoflann/include")

set(DIRENT_DIR ${CMAKE_CURRENT_LIST_DIR}/thirdparty/dirent)
set(DIRENT_FILES
  ${DIRENT_DIR}/direntWindows.h
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#
# Library name and options
#

# Target name
set(target utilslib)

# Exit here if required dependencies are not met
message(STATUS "Lib ${target}")

set(MAINTARGET ${target})

# Set API export file and macro
string(TOUPPER ${target} target_upper)
# set(feature_file "include/${target}/${target}_features.h")
set(export_file  "include/${target}/${target}_api.h")
set(export_macro "${target_upper}_API")


#
# Sources
#

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/source")

file(GLOB_RECURSE header_files
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
)
set(headers
   "${CMAKE_CURRENT_SOURCE_DIR}/include/FileHelper.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/random_numbers.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/ransac.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/trees.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/utils_common.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/utils_cv.h"
)

set(sources
    ${source_path}/FileHelper.cpp
    ${source_path}/random_numbers.cpp
    ${source_path}/ransac.cpp
    ${source_path}/trees.cpp
    ${source_path}/utils_common.cpp
    ${source_path}/utils_cv.cpp
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

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if (COMPILER_SUPPORTS_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
# elseif (COMPILER_SUPPORTS_CXX14)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# elseif(COMPILER_SUPPORTS_CXX11)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# elseif(COMPILER_SUPPORTS_CXX0X)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build library
add_library(${target}
    ${sources}
    ${headers}
    ${header_files}
    ${DIRENT_FILES}
)

# Create namespaced alias
add_library(${META_PROJECT_NAME}::${target} ALIAS ${target})

# Export library for downstream projects
export(TARGETS ${target} NAMESPACE ${META_PROJECT_NAME}:: FILE ${PROJECT_BINARY_DIR}/cmake/${target}/${target}-export.cmake)

# Create feature detection header
# Compilers: https://cmake.org/cmake/help/v3.1/variable/CMAKE_LANG_COMPILER_ID.html#variable:CMAKE_%3CLANG%3E_COMPILER_ID
# Feature: https://cmake.org/cmake/help/v3.1/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html
# write_compiler_detection_header(
#     FILE ${feature_file}
#     PREFIX ${target_upper}
#     COMPILERS AppleClang Clang GNU MSVC
#     FEATURES cxx_alignas cxx_alignof cxx_constexpr cxx_final cxx_noexcept cxx_nullptr cxx_sizeof_member cxx_thread_local
# )

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

    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${Nanoflann_INCLUDE_DIR}

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

set(LIBRARIES
    ${OpenCV_LIBS}
    #Eigen
)
if(UNIX)
  set(LIBRARIES ${LIBRARIES} pthread)
endif()

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LIBRARIES}
    ${LIBRARIES}

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

# target_link_libraries(${target}
#     PRIVATE

#     PUBLIC
#     ${DEFAULT_LINKER_OPTIONS}

#     INTERFACE
# )


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
