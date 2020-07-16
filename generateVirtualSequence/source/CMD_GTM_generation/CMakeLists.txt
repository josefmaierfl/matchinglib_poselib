
#
# Executable name and options
#

# Target name
set(target virtualSequenceLib-GTM-interface)
message(STATUS "${target}")

#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.71.0 REQUIRED COMPONENTS
		thread
		filesystem
		regex
		system)

find_package(VTK 7.2 QUIET NO_MODULE)
if(NOT VTK_FOUND)
	find_package(VTK 8.2.0 REQUIRED NO_MODULE)
endif()
message(STATUS "VTK Version: ${VTK_VERSION}")
message(STATUS "VTK Lib location: ${VTK_LIBRARIES}")

find_package(Ceres REQUIRED)

#
# Sources
#

set(sources
    main.cpp
	argvparser.cpp
)


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
	${Boost_INCLUDE_DIRS}
	${VTK_USE_FILE}
	${CERES_INCLUDE_DIRS}
)


#
# Libraries
#

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::generateVirtualSequenceLib
	${Boost_LIBRARIES}
	${VTK_LIBRARIES}
	${CERES_LIBRARIES}
#    gmock-dev
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
