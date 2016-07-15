
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
option(ENABLE_MATCHINGLIB  "build with matchinglib" ON)
option(ENABLE_POSELIB  "build with poselib" OFF)
option(ENABLE_BASELIB  "build with poselib" OFF)
option(ENABLE_FIBLIB  "build with poselib" OFF)
# add_subdirectory(baselib)
# add_subdirectory(fiblib)


if(${ENABLE_MATCHINGLIB})
  
  add_subdirectory(matchinglib)
endif()

if(${ENABLE_POSELIB})
  add_subdirectory(poselib)
endif()

# Examples
# set(IDE_FOLDER "Examples")
# add_subdirectory(examples)

# Tests
set(IDE_FOLDER "Tests")
add_subdirectory(tests)


#
# Deployment
#

# Deploy generated headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${META_PROJECT_NAME} DESTINATION include COMPONENT dev)
