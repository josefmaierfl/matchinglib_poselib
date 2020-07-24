
# 
# External dependencies
# 

# find_package(THIRDPARTY REQUIRED)


# 
# Executable name and options
# 

# Target name
set(target loadData)

# Exit here if required dependencies are not met
message(STATUS "Example ${target}")

find_package(Boost 1.71.0 REQUIRED COMPONENTS
        thread
        filesystem
        regex
        system)

find_package(Eigen REQUIRED)

find_package(OpenCV 4.2.0 REQUIRED)

find_package(VTK 7.2 QUIET NO_MODULE)
if(NOT VTK_FOUND)
    find_package(VTK 8.2.0 REQUIRED NO_MODULE)
endif()
message(STATUS "VTK Version: ${VTK_VERSION}")

find_package(PCL 1.11.0 REQUIRED)
#message(STATUS ${PCL_VERSION})
if(PCL_FOUND)
    message(STATUS "Found PCL lib!")
    message(STATUS "PCL Version: ${PCL_VERSION}")
endif()

# 
# Sources
# 

set(sources
    main.cpp
    argvparser.cpp
    io_data.cpp
)


# 
# Create executable
# 

# Build executable
add_executable(${target}
    #MACOSX_BUNDLE
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
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIR}
    ${VTK_USE_FILE}
    ${PCL_INCLUDE_DIRS}
)


# 
# Libraries
# 

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LIBRARIES}
    ${META_PROJECT_NAME}::generateVirtualSequenceLib
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    ${PCL_LIBRARY_DIRS}
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
    ${PCL_GEOMETRY_LIBRARIES}
    ${PCL_VISUALIZATION_LIBRARIES}
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
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)
