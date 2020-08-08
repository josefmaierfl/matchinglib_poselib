
#
# External dependencies
#

find_package(Eigen REQUIRED)

find_package(OpenCV 4.2.0 REQUIRED)

find_package(Clapack REQUIRED)
#if (CLAPACK_FOUND STREQUAL "YES")
#  message(STATUS "Clapack found!")
#else()
#  message(FATAL_ERROR "Clapack not found!")
#endif()

find_package(SBA REQUIRED)
#if (SBA_FOUND STREQUAL "YES")
#  message(STATUS "SBA found!")
#else()
#  message(FATAL_ERROR "SBA not found!")
#endif()

SET( Nanoflann_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/nanoflann-1.3.0/include")

#
# Library name and options
#

# Target name
set(target poselib)

# Exit here if required dependencies are not met
message(STATUS "Lib ${target}")

set(MAINTARGET ${target})
add_subdirectory(thirdparty/opengv)

# Set API export file and macro
string(TOUPPER ${target} target_upper)
set(feature_file "include/${target}/${target}_features.h")
set(export_file  "include/${target}/${target}_api.h")
set(export_macro "${target_upper}_API")


#
# Sources
#

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/source")

set(headers
  ${include_path}/BA_driver.h
  ${include_path}/imgproj.h
  ${include_path}/${target}/glob_includes.h
  ${include_path}/${target}/pose_estim.h
  ${include_path}/${target}/pose_helper.h
  ${include_path}/${target}/pose_linear_refinement.h
  ${include_path}/${target}/stereo_pose_refinement.h
  ${include_path}/${target}/stereo_pose_types.h
  ${include_path}/${target}/nanoflannInterface.h

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
  ${include_path}/${target}/pose_homography.h

  ${include_path}/usac/config/ConfigParams.h
  ${include_path}/usac/config/ConfigParamsFundmatrix.h
  ${include_path}/usac/config/ConfigParamsHomog.h
  ${include_path}/usac/config/ConfigParamsEssentialMat.h
  ${include_path}/usac/config/ConfigParamsRotationMat.h
  ${include_path}/usac/estimators/FundmatrixEstimator.h
  ${include_path}/usac/estimators/HomogEstimator.h
  ${include_path}/usac/estimators/EssentialMatEstimator.h
  ${include_path}/usac/estimators/RotationMatEstimator.h
  ${include_path}/usac/estimators/USAC.h
  ${include_path}/usac/utils/FundmatrixFunctions.h
  ${include_path}/usac/utils/PoseFunctions.h
  ${include_path}/usac/utils/HomographyFunctions.h
  ${include_path}/usac/utils/MathFunctions.h
  ${include_path}/usac/utils/Timer.h
  ${include_path}/usac/utils/weightingEssential.h
  ${include_path}/usac/usac_estimations.h
)

set(sources
  ${source_path}/BA_driver.cpp
  ${source_path}/imgproj.c
  ${source_path}/pose_estim.cpp
  ${source_path}/pose_helper.cpp
  ${source_path}/pose_linear_refinement.cpp
  ${source_path}/stereo_pose_refinement.cpp
  ${source_path}/nanoflannInterface.cpp

  ${source_path}/arrsac/sequential_probability_ratio.cc

  ${source_path}/five-point-nister/five-point.cpp
  ${source_path}/five-point-nister/modelest.cpp
  ${source_path}/five-point-nister/precomp.cpp

  ${source_path}/HomographyAlignment.cpp
  ${source_path}/pose_homography.cpp

  ${source_path}/usac/utils/FundmatrixFunctions.cpp
  ${source_path}/usac/utils/HomographyFunctions.cpp
  ${source_path}/usac/utils/PoseFunctions.cpp
  ${source_path}/usac/utils/MathFunctions.cpp
  ${source_path}/usac/utils/weightingEssential.cpp
  ${source_path}/usac/usac_estimations.cpp
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
message(STATUS "clapack: ${CLAPACK_INCLUDE_DIRS}")
target_include_directories(${target}
    PRIVATE
    ${PROJECT_BINARY_DIR}/source/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opengv/include
    ${Nanoflann_INCLUDE_DIR}

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
        ${CLAPACK_INCLUDE_DIRS}

    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    #$<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:${INSTALL_INCLUDE}/${target}/include> # for client in install mode
    $<INSTALL_INTERFACE:${INSTALL_INCLUDE}/${target}>
    $<INSTALL_INTERFACE:${INSTALL_LIB}/${target}>
)


#
# Libraries
#
message(STATUS "clapack: ${CLAPACK_LIBRARIES}")
target_link_libraries(${target}
    PRIVATE
        #${SBA_LIBRARIES}
        #${CLAPACK_LIBRARIES}

    opengv

    PUBLIC
    ${DEFAULT_LIBRARIES}
    ${OpenCV_LIBS}
    ${CLAPACK_LIBRARIES}
        sba
    #Eigen

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
    EXPORT  "${target}-export"                      COMPONENT dev
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
