
#
# Configuration for all sub-projects
#

# Generate version-header
string(TOUPPER ${META_PROJECT_NAME} META_PRO  JECT_NAME_UPPER)
configure_file(version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/${META_PROJECT_NAME}/${META_PROJECT_NAME}-version.h)


#
# Sub-projects
#

# Libraries
set(IDE_FOLDER "")
option(ENABLE_GENVIRTSEQULIB  "build with testframework" ON)
option(ENABLE_CMD_INTERFACE "build with command line interface for the virtual sequence library" ON)
option(ENABLE_GTM_INTERFACE "build with command line interface for generating ground truth matches and flow for the MegaDepth dataset" ON)


if(${ENABLE_GENVIRTSEQULIB})
  add_subdirectory(generateVirtualSequenceLib)
endif()

if(${ENABLE_CMD_INTERFACE})
  add_subdirectory(CMD_Interface)
endif()

if(${ENABLE_GTM_INTERFACE})
    add_subdirectory(CMD_GTM_generation)
endif()

# Examples
 set(IDE_FOLDER "Examples")
 add_subdirectory(examples)

# Tests
set(IDE_FOLDER "Tests")
add_subdirectory(tests)


#
# Deployment
#

# Deploy generated headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${META_PROJECT_NAME} DESTINATION include COMPONENT dev)
