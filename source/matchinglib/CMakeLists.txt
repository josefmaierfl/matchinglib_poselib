
#
# External dependencies
#

option(USE_NON_FREE_CODE "use non free keypoint and descriptor types in the library." OFF)

if(WIN32)
if("$ENV{THIRDPARTYROOT}" STREQUAL "")
  message(ERROR "Not able to build lib: Thirdpartyroot not found! set THIRDPARTYROOT")
endif()
include($ENV{THIRDPARTYROOT}/_buildtools/CMakeModules/TPMacros.cmake)
endif()

if(WIN32)
  find_package(Eigen REQUIRED)
else()
  find_package(PkgConfig)
  pkg_search_module(EIGEN REQUIRED eigen3)
endif()

find_package(Eigen REQUIRED)

if(NOT OpenCV_DIR)
    message(STATUS "could not find any opencv dir trying fallback")
    if(WIN32)
      set(OpenCV_DIR "C:/work/THIRDPARTY/opencv-3.0.0/build/install")
      find_package(OpenCV 3.0 REQUIRED NO_CMAKE_PATH)
    else()
      set(OpenCV_DIR "/home/jungr/installed/opencv-3.1.0/share/OpenCV/")
      find_package(OpenCV REQUIRED)
    endif()
else()
    find_package(OpenCV 3.0 REQUIRED)
endif()


find_package(Nanoflann REQUIRED)
#find_package(Flann REQUIRED)

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
#    ${include_path}/test.h
)

set(sources
    # ${source_path}/test.cpp
    ${source_path}/matchers.cpp
    ${source_path}/correspondences.cpp
    ${source_path}/features.cpp
    ${source_path}/match_statOptFlow.cpp
    ${source_path}/match_opticalflow.cpp
    ${source_path}/vfc.cpp
    ${source_path}/vfcMatches.cpp
    ${source_path}/ratioMatches_Flann.cpp

    ${source_path}/CascadeHash/BucketBuilder.cc
    ${source_path}/CascadeHash/CasHashMatcher.cc
    ${source_path}/CascadeHash/DataPreProcessor.cc
    ${source_path}/CascadeHash/HashConvertor.cc
    ${source_path}/CascadeHash/MatchPairLoader.cc

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
    ${CMAKE_CURRENT_BINARY_DIR}/include

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    #${FLANN_INCLUDE_DIRS}
    ${Nanoflann_INCLUDE_DIR}
    #${FLANN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}

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
    #${FLANN_DEFINITIONS}

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
