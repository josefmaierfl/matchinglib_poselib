
#
# External dependencies
#

if("$ENV{THIRDPARTYROOT}" STREQUAL "")
  message(ERROR "Not able to build lib: Thirdpartyroot not found! set THIRDPARTYROOT")
endif()
include($ENV{THIRDPARTYROOT}/_buildtools/CMakeModules/TPMacros.cmake)

find_package(Eigen REQUIRED)
if (Eigen_FOUND STREQUAL "YES")
  message("Eigen found!")
else()
  message(FATAL_ERROR "Eigen not found!")
endif()


if(WIN32)
  set(OpenCV_DIR "C:/work/THIRDPARTY/opencv-3.0.0/build/install")
  find_package(OpenCV 3.0 REQUIRED NO_CMAKE_PATH)
else()
  set(OpenCV_DIR "/home/jungr/installed/opencv-3.1.0/share/OpenCV/")
  find_package(OpenCV 3.1 REQUIRED)
endif()

#set(OpenCV_DIR "C:/work/THIRDPARTY/opencv-3.0.0/build/install")
#find_package(OpenCV 3.0)
# find_package(OpenCV 2.4.9 REQUIRED)
# if (OpenCV_FOUND STREQUAL "ON")
  # message("OpenCV found!")
# else()
  # message(FATAL_ERROR "OpenCV not found!")
# endif()

# find_package(Nanoflann REQUIRED)
# if (Nanoflann_FOUND STREQUAL "YES")
  # message("Nanoflann found!")
# else()
  # message(FATAL_ERROR "Nanoflann not found!")
# endif()

# find_package(Flann 1.8.4 REQUIRED)
# if (FLANN_FOUND STREQUAL "YES")
  # message("Flann found!")
# else()
  # message(FATAL_ERROR "Flann not found!")
# endif()

find_package(Clapack REQUIRED)
if (CLAPACK_FOUND STREQUAL "YES")
  message("Clapack found!")
else()
  message(FATAL_ERROR "Clapack not found!")
endif()

find_package(SBA REQUIRED)
if (SBA_FOUND STREQUAL "YES")
  message("SBA found!")
else()
  message(FATAL_ERROR "SBA not found!")
endif()


#
# Library name and options
#

# Target name
set(target poselib)

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

#set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/source")

set(headers
	${include_path}/BA_driver.h
	${include_path}/imgproj.h
	${include_path}/glob_includes.h
	${include_path}/pose_estim.h
	${include_path}/pose_helper.h
	
	${include_path}/arrsac/arrsac.h
	${include_path}/arrsac/estimator.h
	${include_path}/arrsac/prosac_sampler.h
	${include_path}/arrsac/quality_measurement.h
	${include_path}/arrsac/random_sampler.h
	${include_path}/arrsac/sample_consensus_estimator.h
	${include_path}/arrsac/sampler.h
	${include_path}/arrsac/sequential_probability_ratio.h
	
	${include_path}/five-point-nister/_modelest.h
	${include_path}/five-point-nister/five-point.hpp
	${include_path}/five-point-nister/precomp.hpp
	
	${include_path}/HomographyAlignment.h
	${include_path}/pose_homography.h
)

set(sources
	${source_path}/BA_driver.cpp
	${source_path}/imgproj.c
	${source_path}/pose_estim.cpp
	${source_path}/pose_helper.cpp
	
	${source_path}/arrsac/sequential_probability_ratio.cc
	
	${source_path}/five-point-nister/five-point.cpp
	${source_path}/five-point-nister/modelest.cpp
	${source_path}/five-point-nister/precomp.cpp
	
	${source_path}/HomographyAlignment.cpp
	${source_path}/pose_homography.cpp
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
    #${Nanoflann_INCLUDE_DIR}
    #${FLANN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIR}
	${CLAPACK_INCLUDE_DIR}
	${SBA_INCLUDE_DIR}

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
	${CLAPACK_LIBRARIES}
	${SBA_LIBRARIES}

    INTERFACE
)


#
# Compile definitions
#

target_compile_definitions(${target}
    PRIVATE

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
