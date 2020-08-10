# sba CMake file; see http://www.cmake.org and
#                     http://www.insightsoftwareconsortium.org/wiki/index.php/CMake_Tutorial

#PROJECT(SBA)
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

set(CMAKE_MODULE_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GenerateExportHeader)
include(ExternalProject)
include(WriteCompilerDetectionHeader)
include(cmake/Custom.cmake)

set_policy(CMP0028 NEW) # ENABLE CMP0028: Double colon in target name means ALIAS or IMPORTED target.
set_policy(CMP0054 NEW) # ENABLE CMP0054: Only interpret if() arguments as variables or keywords when unquoted.
set_policy(CMP0042 NEW) # ENABLE CMP0042: MACOSX_RPATH is enabled by default.
set_policy(CMP0063 NEW) # ENABLE CMP0063: Honor visibility properties for all target types.

option(BUILD_SHARED_LIBS "Build shared instead of static libraries." OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(IDE_FOLDER "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

set(target "sba")
# Get project name
set(project ${target})

# Declare project
project(project C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

#
# Compiler settings and options
#

include(cmake/CompileOptions.cmake)

# Check for system dir install
set(SYSTEM_DIR_INSTALL FALSE)
if("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr" OR "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
    set(SYSTEM_DIR_INSTALL TRUE)
endif()

# Installation paths
if(UNIX AND SYSTEM_DIR_INSTALL)
    # Install into the system (/usr/bin or /usr/local/bin)
    set(INSTALL_ROOT      "share/${project}")       # /usr/[local]/share/<project>
    set(INSTALL_CMAKE     "share/${project}/cmake") # /usr/[local]/share/<project>
    set(INSTALL_DATA      "share/${project}")       # /usr/[local]/share/<project>
    set(INSTALL_BIN       "bin")                    # /usr/[local]/bin
    set(INSTALL_SHARED    "lib")                    # /usr/[local]/lib
    set(INSTALL_LIB       "lib")                    # /usr/[local]/lib
    set(INSTALL_INCLUDE   "include")                # /usr/[local]/include
    set(INSTALL_DOC       "share/doc/${project}")   # /usr/[local]/share/doc/<project>
    set(INSTALL_SHORTCUTS "share/applications")     # /usr/[local]/share/applications
    set(INSTALL_ICONS     "share/pixmaps")          # /usr/[local]/share/pixmaps
    set(INSTALL_INIT      "/etc/init")              # /etc/init (upstart init scripts)
else()
    # Install into local directory
    set(INSTALL_ROOT      ".")                      # ./
    set(INSTALL_CMAKE     "cmake")                  # ./cmake
    set(INSTALL_DATA      ".")                      # ./bin
    set(INSTALL_BIN       ".")                      # ./bin
    set(INSTALL_SHARED    "lib")                    # ./lib
    set(INSTALL_LIB       "lib")                    # ./lib
    set(INSTALL_INCLUDE   "include")                # ./include
    set(INSTALL_DOC       "doc")                    # ./doc
    set(INSTALL_SHORTCUTS "misc")                   # ./misc
    set(INSTALL_ICONS     "misc")                   # ./misc
    set(INSTALL_INIT      "misc")                   # ./misc
endif()

# Set runtime path
set(CMAKE_SKIP_BUILD_RPATH            FALSE) # Add absolute path to all dependencies for BUILD
set(CMAKE_BUILD_WITH_INSTALL_RPATH    FALSE) # Use CMAKE_INSTALL_RPATH for INSTALL
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE) # Do NOT add path to dependencies for INSTALL

if(NOT SYSTEM_DIR_INSTALL)
    # Find libraries relative to binary
    if(APPLE)
        set(CMAKE_INSTALL_RPATH "@loader_path/../../../${INSTALL_LIB}")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN/${INSTALL_LIB}")
    endif()
endif()

# Install cmake find script for the project
install(FILES ${target}-config.cmake DESTINATION ${INSTALL_ROOT} COMPONENT dev)

#EXECUTE_PROCESS( COMMAND bash -c "gcc --version | grep ^gcc | sed 's/^.* //g'" OUTPUT_VARIABLE GCC_VERSION)
#STRING(STRIP GCC_VERSION ${GCC_VERSION})
#string(REGEX REPLACE "\n$" "" GCC_VERSION "${GCC_VERSION}")
#if(BUILD_SHARED_LIBS)
  #set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/linux64gcc${GCC_VERSION})
#ELSE(BUILD_SHARED_LIBS)
  #set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/linux64gcc${GCC_VERSION})
#ENDIF(BUILD_SHARED_LIBS)

#does not have any effect on executables on its own only with
#   set_target_properties(${PROJ_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
#   after the target definition (see end of file)
#set(CMAKE_DEBUG_POSTFIX _d)

# f2c is sometimes equivalent to libF77 & libI77; in that case, set HAVE_F2C to 0
#SET(HAVE_F2C 1 CACHE BOOL "Do we have f2c or F77/I77?" )

# the directory where the lapack/blas/f2c libraries reside
#SET(LAPACKBLAS_DIR /usr/lib CACHE PATH "Path to lapack/blas libraries")

# actual names for the lapack/blas/f2c libraries
#SET(LAPACK_LIB lapack CACHE STRING "The name of the lapack library")
#SET(BLAS_LIB blas CACHE STRING "The name of the blas library")
#IF(HAVE_F2C)
  #SET(F2C_LIB f2c CACHE STRING "The name of the f2c library")
#ELSE(HAVE_F2C)
  #SET(F77_LIB libF77 CACHE STRING "The name of the F77 library")
  #SET(I77_LIB libI77 CACHE STRING "The name of the I77 library")
#ENDIF(HAVE_F2C)

string(TOUPPER ${target} target_upper)
set(feature_file "include/${target}/${target}_features.h")

find_package(Clapack REQUIRED)

# sba library source files
add_library(${target}
  sba_levmar.c sba_levmar_wrap.c sba_lapack.c sba_crsm.c sba_chkjac.c
  sba.h sba_chkjac.h compiler.h
)

# Export library for downstream projects
export(TARGETS ${target} FILE ${PROJECT_BINARY_DIR}/cmake/${target}/${target}-export.cmake)

# Create feature detection header
# Compilers: https://cmake.org/cmake/help/v3.1/variable/CMAKE_LANG_COMPILER_ID.html#variable:CMAKE_%3CLANG%3E_COMPILER_ID
# Feature: https://cmake.org/cmake/help/v3.1/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html
write_compiler_detection_header(
    FILE ${feature_file}
    PREFIX ${target_upper}
    COMPILERS AppleClang Clang GNU MSVC
    FEATURES cxx_alignas cxx_alignof cxx_constexpr cxx_final cxx_noexcept cxx_nullptr cxx_sizeof_member cxx_thread_local
)

#if(MSVC)
	#foreach(flag_var
			#CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
			#CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
			#CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
	   #if(${flag_var} MATCHES "/MT")
		  #string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
	   #endif(${flag_var} MATCHES "/MT")
	#endforeach(flag_var)
#endif()

set_target_properties(${target}
    PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
)

#set_target_properties(${target} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

target_include_directories(${target}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}

  PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${CLAPACK_INCLUDE_DIRS}

  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INSTALL_INCLUDE}/${target}> # for client in install mode
    $<INSTALL_INTERFACE:${INSTALL_LIB}/${target}>
    )

target_link_libraries(${target}
		PUBLIC
    ${DEFAULT_LIBRARIES}
		${CLAPACK_LIBRARIES}

    INTERFACE
    )

target_compile_definitions(${target}
    PRIVATE

    PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:${target_upper}_STATIC_DEFINE>
    ${DEFAULT_COMPILE_DEFINITIONS}

    INTERFACE
)

# Library
install(TARGETS ${target}
    EXPORT  "${target}-export"                      COMPONENT dev
    RUNTIME DESTINATION ${INSTALL_BIN}/${target}    COMPONENT runtime
    LIBRARY DESTINATION ${INSTALL_SHARED}/${target} COMPONENT runtime
    ARCHIVE DESTINATION ${INSTALL_LIB}/${target}    COMPONENT dev
)
# Header files
#list(APPEND INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/")
file(GLOB INCLUDE_FILES
    "*.h"
    "*.hpp"
)
install(FILES
    ${INCLUDE_FILES} DESTINATION ${INSTALL_INCLUDE}/${target} COMPONENT dev
)
# Generated header files
install(DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/include/${target} DESTINATION ${INSTALL_INCLUDE}/${target}
    COMPONENT dev
)
# CMake config
install(EXPORT ${target}-export
    DESTINATION ${INSTALL_CMAKE}/${target}
    COMPONENT   dev
)

#ADD_SUBDIRECTORY(demo)
#SUBDIRS(demo)
