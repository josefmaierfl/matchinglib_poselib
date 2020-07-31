
#
# Executable name and options
#

option(COPY_TEST_MATCH_IMGS  "Copy images for testing to binary dir" ON)

# Target name
set(target matchinglib-test)
message(STATUS "Test ${target}")


#
# Sources
#

set(sources
    main.cpp
	io_data.cpp
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

if(${COPY_TEST_MATCH_IMGS})
	file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/imgs DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

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
)


#
# Libraries
#

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::matchinglib
    gmock-dev
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

#
# Deployment
#

# Executable
install(TARGETS ${target}
    RUNTIME DESTINATION ${INSTALL_BIN}/${META_PROJECT_NAME} COMPONENT bin
    BUNDLE  DESTINATION ${INSTALL_BIN}/${META_PROJECT_NAME} COMPONENT bin
)