
#
# External dependencies
#

option(USE_NON_FREE_CODE "use non free keypoint and descriptor types in the library." OFF)
option(BUILD_WITH_AKAZE_CUDA "Builds the library with AKAZE on CUDA" OFF)

find_package(Eigen REQUIRED)

find_package(OpenCV 4.8.0 REQUIRED)

SET( Nanoflann_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nanoflann/include")

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

# add nmslib
# we have to force to build it statically as otherwise multiple singleton instances appear...
set(BUILD_SHARED_LIBS_ORIG  ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(thirdparty/nmslib-2.1.1/similarity_search)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_ORIG})

# Set multithreading support for ANNOY lib
add_compile_definitions(ANNOYLIB_MULTITHREADED_BUILD)

# Set API export file and macro
string(TOUPPER ${target} target_upper)
# set(feature_file "include/${target}/${target}_features.h")
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

if(BUILD_WITH_AKAZE_CUDA)
    set(headers ${headers} "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/cuda_affine_feature.h")
    set(sources ${sources} ${source_path}/cuda_affine_feature.cpp)
endif()

if(BUILD_WITH_AKAZE_CUDA)
  set(AKAZE_CUDA_DIR ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cuda_akaze/src/lib)
  set(AKAZE_CUDA_HEADERS
    ${AKAZE_CUDA_DIR}/AKAZEConfig.h
    ${AKAZE_CUDA_DIR}/AKAZE.h
    ${AKAZE_CUDA_DIR}/fed.h
    ${AKAZE_CUDA_DIR}/nldiffusion_functions.h
    ${AKAZE_CUDA_DIR}/utils.h
  )
  set(AKAZE_CUDA_CPP
    ${AKAZE_CUDA_DIR}/AKAZE.cpp
    ${AKAZE_CUDA_DIR}/fed.cpp
    ${AKAZE_CUDA_DIR}/nldiffusion_functions.cpp
    ${AKAZE_CUDA_DIR}/utils.cpp
  )
  set(AKAZE_CUDA_FILES ${AKAZE_CUDA_HEADERS} ${AKAZE_CUDA_CPP})
  set(AKAZE_CUDA_SRCS
    ${AKAZE_CUDA_DIR}/cudaImage.cu
    ${AKAZE_CUDA_DIR}/cuda_akaze.cu
  )
  set(headers ${headers} ${AKAZE_CUDA_HEADERS})
endif()

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

# if (COMPILER_SUPPORTS_CXX17)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
# # elseif (COMPILER_SUPPORTS_CXX14)
# #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# # elseif(COMPILER_SUPPORTS_CXX11)
# #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# # elseif(COMPILER_SUPPORTS_CXX0X)
# #   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
# else()
#   message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
# endif()

if (NOT COMPILER_SUPPORTS_CXX17)
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(BUILD_WITH_AKAZE_CUDA)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    enable_language(CUDA)
    set(CUDA_VERBOSE_BUILD ON)
  endif()
  #find_package(CUDA)
  find_package(CUDAToolkit)
  include_directories(${AKAZE_CUDA_DIR})
  add_library(AKAZE_CUDA ${AKAZE_CUDA_FILES})
  if(UNIX)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
      # set(CMAKE_CUDA_FLAGS_DEBUG "-g;-G")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
    else()
      LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_75;-lineinfo;--compiler-options;-O2")
      # LIST(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-pedantic -fPIC")
  endif()
  target_link_libraries(AKAZE_CUDA ${OpenCV_LIBS})
  set_target_properties(AKAZE_CUDA PROPERTIES DEBUG_POSTFIX "d" CUDA_ARCHITECTURES "75")

  set(CUDA_SEPARABLE_COMPILATION ON)
  cuda_add_library(AKAZE_CUDA_CUDA ${AKAZE_CUDA_SRCS})
  target_link_libraries(AKAZE_CUDA_CUDA ${OpenCV_LIBS})
  set_target_properties(AKAZE_CUDA_CUDA PROPERTIES CUDA_ARCHITECTURES "75")
  # target_link_libraries(AKAZE_CUDA_CUDA ${OpenCV_LIBS} ${CUDA_cusolver_LIBRARY})
  # target_compile_options(AKAZE_CUDA_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

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

    ${CMAKE_CURRENT_SOURCE_DIR}/include/${target}
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${Nanoflann_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gms-1.0/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nmslib-2.1.1/similarity_search/include

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

set(LIBRARIES
    ${OpenCV_LIBS}
    #Eigen
)
if(UNIX)
  set(LIBRARIES ${LIBRARIES} pthread)
endif()
if(BUILD_WITH_AKAZE_CUDA)
  set(LIBRARIES ${LIBRARIES} AKAZE_CUDA AKAZE_CUDA_CUDA)
  add_definitions(-DWITH_AKAZE_CUDA)
endif()

target_link_libraries(${target}
    PRIVATE
    NonMetricSpaceLib
	gms-feature-matching

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
# Deployment nmslib
#
# we have to export it here, as the libs cmake file doesn't

install(TARGETS NonMetricSpaceLib
    EXPORT  "NonMetricSpaceLib-export"            			COMPONENT dev
    # RUNTIME DESTINATION ${INSTALL_BIN}/NonMetricSpaceLib    COMPONENT runtime
    # LIBRARY DESTINATION ${INSTALL_SHARED}/NonMetricSpaceLib COMPONENT runtime
    # ARCHIVE DESTINATION ${INSTALL_LIB}/NonMetricSpaceLib   COMPONENT dev
)

# install(DIRECTORY
#     ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nmslib-2.1.1/similarity_search/include DESTINATION ${INSTALL_INCLUDE}/NonMetricSpaceLib
#     COMPONENT dev
# )

# CMake config
install(EXPORT NonMetricSpaceLib-export
    DESTINATION ${INSTALL_CMAKE}/NonMetricSpaceLib
    COMPONENT   dev
)

export(TARGETS NonMetricSpaceLib  FILE ${PROJECT_BINARY_DIR}/cmake/NonMetricSpaceLib/NonMetricSpaceLib-export.cmake)

#
# Deployment AKAZE CUDA
#
if(BUILD_WITH_AKAZE_CUDA)
install(TARGETS AKAZE_CUDA
    EXPORT  "AKAZE_CUDA-export"            			COMPONENT dev
)

# CMake config
install(EXPORT AKAZE_CUDA-export
    DESTINATION ${INSTALL_CMAKE}/AKAZE_CUDA
    COMPONENT   dev
)

export(TARGETS AKAZE_CUDA  FILE ${PROJECT_BINARY_DIR}/cmake/AKAZE_CUDA/AKAZE_CUDA-export.cmake)

#Deployment CUDA lib
install(TARGETS AKAZE_CUDA_CUDA
    EXPORT  "AKAZE_CUDA_CUDA-export"            			COMPONENT dev
)

# CMake config
install(EXPORT AKAZE_CUDA_CUDA-export
    DESTINATION ${INSTALL_CMAKE}/AKAZE_CUDA/CUDA
    COMPONENT   dev
)

export(TARGETS AKAZE_CUDA_CUDA  FILE ${PROJECT_BINARY_DIR}/cmake/AKAZE_CUDA/CUDA/AKAZE_CUDA_CUDA-export.cmake)
endif()

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
