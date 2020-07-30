
#
# Executable name and options
#

# Target name
set(target noMatch_poselib-test)
message(STATUS "Test ${target}")


#
# Sources
#

set(sources
    main.cpp
	io_data.cpp
	argvparser.cpp)

#find_package(Clapack REQUIRED)
#find_package(Boost COMPONENTS filesystem system date_time python36 REQUIRED)
find_package(Boost COMPONENTS filesystem system date_time REQUIRED)
#message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
#message("Libs of boost: " ${Boost_LIBRARIES} )

#find_package(Python 3.6 COMPONENTS Interpreter REQUIRED)
#find_package(PythonLibs REQUIRED)
#message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
#message("Libs of Python: " ${PYTHON_LIBRARIES} )

#find_package(PkgConfig REQUIRED)
#pkg_check_modules (deps REQUIRED IMPORTED_TARGET glib-2.0>=2.2.5)

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
	${Boost_INCLUDE_DIRS}
	#${PYTHON_INCLUDE_DIRS}
)


#
# Libraries
#

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::matchinglib
    ${META_PROJECT_NAME}::poselib
    gmock-dev
    ${SBA_LIBRARIES}
    ${CLAPACK_LIBRARIES}
	${Boost_LIBRARIES}
#	${PYTHON_LIBRARIES}
#	PkgConfig::deps
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