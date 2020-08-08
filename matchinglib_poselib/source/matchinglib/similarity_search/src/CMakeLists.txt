#
# Non-metric Space Library
#
# Authors: Bilegsaikhan Naidan, Leonid Boytsov.
#
# This code is released under the
# Apache License Version 2.0 http://www.apache.org/licenses/.
#
#

file(GLOB PROJ_HDR_FILES ${PROJECT_SOURCE_DIR}/include/*.h ${PROJECT_SOURCE_DIR}/include/method/*.h ${PROJECT_SOURCE_DIR}/include/space/*.h ${PROJECT_SOURCE_DIR}/include/factory/*.h ${PROJ_HDR_FILES}/include/factory/*/*.h)
file(GLOB OTH_HDR_FILES)
set(HDR_FILES ${PROJ_HDR_FILES} ${OTH_HDR_FILES})
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cc ${PROJECT_SOURCE_DIR}/src/space/*.cc ${PROJECT_SOURCE_DIR}/src/method/*.cc)
list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/main.cc)
list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/tune_vptree.cc)
# The dummy application file also needs to be removed from the list
# of library source files:
list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/dummy_app.cc)

if (NOT WITH_EXTRAS)
  # Extra methods
  list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/method/lsh.cc)
  list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/method/lsh_multiprobe.cc)
  list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/method/lsh_space.cc)
  list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/method/falconn.cc)
  list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/method/nndes.cc)

  # Extra spaces
  list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/space/space_sqfd.cc)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
message(STATUS "Header files: ${HDR_FILES}")
message(STATUS "Source files: ${SRC_FILES}")

link_directories(${Boost_LIBRARY_DIRS})

find_package (Threads)
if (Threads_FOUND)
    message (STATUS "Found Threads.")
else (Threads_FOUND)
    message (STATUS "Could not locate Threads.")
endif (Threads_FOUND)

add_library (non_metric_space_lib STATIC ${SRC_FILES} ${HDR_FILES})

# Create namespaced alias
add_library(${META_PROJECT_NAME}::non_metric_space_lib ALIAS non_metric_space_lib)

# Export library for downstream projects
export(TARGETS non_metric_space_lib NAMESPACE ${META_PROJECT_NAME}:: FILE ${PROJECT_BINARY_DIR}/cmake/non_metric_space_lib/non_metric_space_lib-export.cmake)

target_include_directories(non_metric_space_lib
PUBLIC
 #${PROJECT_SOURCE_DIR}/include

 INTERFACE
 $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
 $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
$<INSTALL_INTERFACE:${INSTALL_INCLUDE}/non_metric_space_lib/include> # for client in install mode
$<INSTALL_INTERFACE:${INSTALL_LIB}/non_metric_space_lib>
 )

if (WITH_EXTRAS)
  add_dependencies (non_metric_space_lib lshkit)
  set(LSHKIT_LIB "lshkit")
endif()

if(${ENABLE_NMSLIBTEST})
add_executable (experiment main.cc)
add_executable (tune_vptree tune_vptree.cc)
# The following line is necessary to create an executable for the dummy application:
add_executable (dummy_app dummy_app.cc)

target_link_libraries (experiment non_metric_space_lib ${LSHKIT_LIB} ${Boost_LIBRARIES} ${GSL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (tune_vptree non_metric_space_lib ${LSHKIT_LIB} ${Boost_LIBRARIES} ${GSL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
# What are the libraries that we need to link with for dummy_app?
target_link_libraries (dummy_app non_metric_space_lib ${LSHKIT_LIB}
                                                          ${Boost_LIBRARIES}
                                                          ${GSL_LIBRARIES}
                                                          ${CMAKE_THREAD_LIBS_INIT})
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/release/")
    set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/release/")
else ()
    set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/debug/")
    set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/debug/")
endif ()

# string(LENGTH ${PROJECT_SOURCE_DIR} PREFIX_LEN)
# MATH(EXPR PREFIX_LEN "${PREFIX_LEN}+1")
# foreach(F ${PROJ_HDR_FILES})
  # GET_FILENAME_COMPONENT(FP ${F} PATH)
  # string(SUBSTRING ${FP} ${PREFIX_LEN} -1 FS)
  ##message(${PREFIX_LEN} ":" ${FS})
  # install(FILES "${F}" DESTINATION ${FS})
# endforeach(F)

#install(TARGETS non_metric_space_lib
 # ARCHIVE DESTINATION lib
  #LIBRARY DESTINATION lib
  #COMPONENT library
#)

# Library
install(TARGETS non_metric_space_lib
    EXPORT  "non_metric_space_lib-export"            			      COMPONENT dev
    RUNTIME DESTINATION ${INSTALL_BIN}/non_metric_space_lib     COMPONENT runtime
    LIBRARY DESTINATION ${INSTALL_SHARED}/non_metric_space_lib  COMPONENT runtime
    ARCHIVE DESTINATION ${INSTALL_LIB}/non_metric_space_lib     COMPONENT dev
)

set(non_metric_space_lib_LIBRARIES non_metric_space_lib ${LSHKIT_LIB} ${GSL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
message(STATUS "non_metric_space_lib_LIBRARIES: ${non_metric_space_lib_LIBRARIES}")
set(non_metric_space_lib_LIBRARIES ${non_metric_space_lib_LIBRARIES} PARENT_SCOPE)
