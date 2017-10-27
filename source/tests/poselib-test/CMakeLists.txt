
#
# Executable name and options
#

# Target name
set(target poselib-test)
message(STATUS "Test ${target}")


#
# Sources
#

set(sources
    main.cpp
	io_data.cpp
	argvparser.cpp
)

#find_package(Clapack REQUIRED)
#SET( Nanoflann_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/source/poselib/thirdparty/nanoflann-1.3.0/include")

#
# Create executable
#

# Build executable
add_executable(${target}
    ${sources}
)

# Create namespaced alias
add_executable(${META_PROJECT_NAME}::${target} ALIAS ${target})


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
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${PROJECT_BINARY_DIR}/source/include
    ${CLAPACK_INCLUDE_DIR}
    ${SBA_INCLUDE_DIR}
	#${Nanoflann_INCLUDE_DIR}
)


#
# Libraries
#

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::matchinglib
    #${CLAPACK_LIBRARIES}
    ${META_PROJECT_NAME}::poselib
    gmock-dev
    ${SBA_LIBRARIES}
    ${CLAPACK_LIBRARIES}
#    blas
)


#
# Compile definitions
#

target_compile_definitions(${target}
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}
)


#
# Compile options
#

target_compile_options(${target}
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)


#
# Linker options
#

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)
