target_include_directories(lshkit
  ${PROJECT_SOURCE_DIR}/lshkit/include
  ${PROJECT_SOURCE_DIR}/include
  ${Boost_INCLUDE_DIR}

  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lshkit/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${INSTALL_INCLUDE}/lshkit/include> # for client in install mode
  $<INSTALL_INTERFACE:${INSTALL_LIB}/lshkit>
  )

set(lshkit_SRCS
  src/mplsh.cpp
  src/mplsh-model.cpp
  src/apost.cpp
  src/char_bit_cnt.cpp
  src/vq.cpp
  src/kdtree.cpp)

add_library(lshkit ${lshkit_SRCS})

# Create namespaced alias
add_library(${META_PROJECT_NAME}::lshkit ALIAS lshkit)

# Export library for downstream projects
export(TARGETS lshkit NAMESPACE ${META_PROJECT_NAME}:: FILE ${PROJECT_BINARY_DIR}/cmake/lshkit/lshkit-export.cmake)

target_link_libraries(lshkit
  ${CMAKE_REQUIRED_LIBRARIES}

  INTERFACE
  )

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/release/")
    set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/release/")
else ()
    set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/debug/")
    set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/debug/")
endif ()

#ADD_SUBDIRECTORY("tools")

#install(TARGETS lshkit
  #ARCHIVE DESTINATION lib
  #LIBRARY DESTINATION lib
  #COMPONENT library
#)

# Library
install(TARGETS lshkit
    EXPORT  "lshkit-export"            			     COMPONENT dev
    RUNTIME DESTINATION ${INSTALL_BIN}/lshkit    COMPONENT runtime
    LIBRARY DESTINATION ${INSTALL_SHARED}/lshkit COMPONENT runtime
    ARCHIVE DESTINATION ${INSTALL_LIB}/lshkit    COMPONENT dev
)

# Header files
install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/lshkit/include DESTINATION ${INSTALL_INCLUDE}/lshkit
        COMPONENT dev
        )

# CMake config
install(EXPORT $lshkit-export
    NAMESPACE   ${META_PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKE}/lshkit
    COMPONENT   dev
)
