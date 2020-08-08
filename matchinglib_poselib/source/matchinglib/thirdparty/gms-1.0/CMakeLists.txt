cmake_minimum_required(VERSION 3.7)
project(gms-feature-matching)


set(USE_GPU FALSE CACHE BOOL "use cuda for feature matching")

if(USE_GPU)
    find_package(OpenCV 4.1.1 REQUIRED core imgproc highgui cudafeatures2d)
else()
    find_package(OpenCV 4.1.1 REQUIRED core imgproc highgui features2d)
endif()

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# Target name
set(target gms-feature-matching)




#
# Sources
#

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE header_files
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
)
set(headers
   "${CMAKE_CURRENT_SOURCE_DIR}/include/MatchGMS.h"
   "${CMAKE_CURRENT_SOURCE_DIR}/include/ProcessImagePairs.h"
)
set(sources
    # ${source_path}/test.cpp
    ${source_path}/MatchGMS.cpp
    ${source_path}/ProcessImagePairs.cpp
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
add_library(${target} STATIC
    ${sources}
    ${headers}
    ${header_files}
)

#
# Project options
#

set_target_properties(${target}
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Create namespaced alias
add_library(${META_PROJECT_NAME}::${target} ALIAS ${target})

# Export library for downstream projects
export(TARGETS ${target} NAMESPACE ${META_PROJECT_NAME}:: FILE ${PROJECT_BINARY_DIR}/cmake/${target}/${target}-export.cmake)


# file(GLOB SOURCE_FILES "src/*.cpp" "include/*.h")

target_include_directories(${target}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  PUBLIC
  ${DEFAULT_INCLUDE_DIRECTORIES}
  ${OpenCV_INCLUDE_DIRS}

  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${INSTALL_INCLUDE}/${target}/include> # for client in install mode
  $<INSTALL_INTERFACE:${INSTALL_LIB}/${target}>
  )

# add_executable(gms-feature-matching ${SOURCE_FILES})

if(USE_GPU)
    add_definitions(-DUSE_GPU)
endif()

target_link_libraries(${target}
  ${OpenCV_LIBS}
)

# Library
install(TARGETS ${target}
    EXPORT  "${target}-export"            			     COMPONENT dev
    RUNTIME DESTINATION ${INSTALL_BIN}/${target}     COMPONENT runtime
    LIBRARY DESTINATION ${INSTALL_SHARED}/${target}  COMPONENT runtime
    ARCHIVE DESTINATION ${INSTALL_LIB}/${target}     COMPONENT dev
)

# Header files
install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${INSTALL_INCLUDE}/${target}
        COMPONENT dev
        )

# CMake config
install(EXPORT ${target}-export
    NAMESPACE   ${META_PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKE}/${target}
    COMPONENT   dev
)
