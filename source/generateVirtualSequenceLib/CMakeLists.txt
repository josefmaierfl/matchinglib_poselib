
#
# External dependencies
#

#option(USE_NON_FREE_CODE "use non free keypoint and descriptor types in the library." ON)

if(WIN32)
if("$ENV{THIRDPARTYROOT}" STREQUAL "")
  message(ERROR "Not able to build lib: Thirdpartyroot not found! set THIRDPARTYROOT")
endif()
#if("$ENV{PFELIBROOT}" STREQUAL "")
	#message(ERROR "Not able to build lib: Root of pfelib not found! set PFELIBROOT")
#endif()
include($ENV{THIRDPARTYROOT}/_buildtools/CMakeModules/TPMacros.cmake)
endif()

#Eigen is included in PCL:
#if(WIN32)
  #find_package(Eigen REQUIRED)
#else()
  #find_package(PkgConfig)
  #pkg_search_module(EIGEN REQUIRED eigen3)
#endif()

find_package(Eigen REQUIRED)

if("${OpenCV_DIR}" STREQUAL "")
    message(STATUS "could not find any opencv dir trying fallback")
    if(WIN32)
      # set(OpenCV_DIR "C:/work/THIRDPARTY/opencv-3.0.0/build/install")
      # find_package(OpenCV 3.0 REQUIRED NO_CMAKE_PATH)
	  find_package(OpenCV 3.2 REQUIRED)
    else()
      set(OpenCV_DIR "/home/jungr/installed/opencv-3.1.0/share/OpenCV/")
      find_package(OpenCV REQUIRED)
    endif()
else()
    find_package(OpenCV 3.2 REQUIRED)
endif()

if(WIN32)
	set(PCL_DIR "C:/work/THIRDPARTY/pcl-1.8.1/pcl-1.8.1-dl/PCL 1.8.1/cmake/")
	find_package(PCL 1.8 REQUIRED)# COMPONENTS common filters)
else()
	find_package(PCL 1.8 REQUIRED)
endif()
#message(STATUS ${PCL_VERSION})
if(PCL_FOUND)
	message(STATUS "Found PCL lib!")
	message(STATUS "PCL Version: ${PCL_VERSION}")
endif()

find_package(Nanoflann REQUIRED)
#find_package(Flann REQUIRED)

#find_package(OpenMP)
#if (OPENMP_FOUND)
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

# set(MATCHINGLIB_FOUND "YES")
# set(MATCHINGLIB_ROOT_DIR "C:/work/navigation/matching")
# set(MATCHINGLIB_INCLUDE_DIR ${MATCHINGLIB_ROOT_DIR}/source/matchinglib/include)
# set(MATCHINGLIB_INCLUDE_DIRS 
# ${MATCHINGLIB_INCLUDE_DIR}
# ${MATCHINGLIB_ROOT_DIR}/buildVS15/source/matchinglib/include
# )
# SET(MATCHINGLIB_LIBRARY "${MATCHINGLIB_ROOT_DIR}/buildVS15/RelWithDebInfo/matchinglib.lib")
# set(MATCHINGLIB_LIBRARY_DEBUG "${MATCHINGLIB_ROOT_DIR}/buildVS15/Debug/matchinglibd.lib")
# set(MATCHINGLIB_LIBRARIES "debug;${MATCHINGLIB_LIBRARY_DEBUG};optimized;${MATCHINGLIB_LIBRARY}")

#set(matching_DIR  "C:/work/navigation/matching/install")
#find_package(matching REQUIRED)


#set(PFELIB_ROOT_DIR $ENV{PFELIBROOT})
#set(PFELIB_INCLUDE_DIR ${PFELIB_ROOT_DIR}/include)
#set(PFELIB_LIBRARY 
#${PFELIB_ROOT_DIR}/lib/win7x64vs14/pfe_utils_WIN64.lib
#${PFELIB_ROOT_DIR}/lib/win7x64vs14/pfe_base_gen_WIN64.lib
#)
#set(PFELIB_LIBRARY_DEBUG 
#${PFELIB_ROOT_DIR}/lib/win7x64vs14/pfe_utils_WIN64_D.lib
#${PFELIB_ROOT_DIR}/lib/win7x64vs14/pfe_base_gen_WIN64_D.lib
#)
#set(PFELIB_LIBRARIES "debug;${PFELIB_LIBRARY_DEBUG};optimized;${PFELIB_LIBRARY}")

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

file(GLOB_RECURSE header_files
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
)
set(headers
#    ${include_path}/test.h
	#${include_path}/getStereoCameraExtr.h
	#${include_path}/generateSequence.h
	#${include_path}/helper_funcs.h
)

set(sources
    #${source_path}/argvparser.cpp
    ${source_path}/base_matcher.cpp
    ${source_path}/eval_start.cpp
    ${source_path}/io_data.cpp
    #${source_path}/main.cpp

	${source_path}/loadGTMfiles.cpp
	
	${source_path}/getStereoCameraExtr.cpp
	${source_path}/generateSequence.cpp
	${source_path}/helper_funcs.cpp

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
	#${PFELIB_LIBRARIES}
	#matching::matchinglib
	
	${PCL_LIBRARY_DIRS}
	${PCL_COMMON_LIBRARIES}
	${PCL_OCTREE_LIBRARIES}
	${PCL_IO_LIBRARIES}
	${PCL_KDTREE_LIBRARIES}
	${PCL_SEARCH_LIBRARIES}
	${PCL_FILTERS_LIBRARIES}
	${PCL_GEOMETRY_LIBRARIES}
	${PCL_SEGMENTATION_LIBRARIES}

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
