project(python_bindings)

cmake_minimum_required (VERSION 2.8)
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif ()


# ============================================================================ #
# Build Type. By default is release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug, Release" FORCE)


# ============================================================================ #
# Where to output binaries and libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")


set(libraries)

# Boost
find_package(Boost REQUIRED COMPONENTS python)
include_directories(${Boost_INCLUDE_DIR})
list( APPEND libraries ${Boost_PYTHON_LIBRARY} )
message(${Boost_PYTHON_LIBRARY})

# Python
find_package(PythonLibs REQUIRED)
list( APPEND libraries ${PYTHON_LIBRARIES})
include_directories(${PYTHON_INCLUDE_DIRS})

# mac
if (APPLE)
  include_directories(/usr/local/lib/python2.7/site-packages/numpy/core/include)
  set(CMAKE_MACOSX_RPATH ON)
endif()

# opencv
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
list( APPEND libraries ${OpenCV_LIBS} )
message(${OpenCV_LIBS})


include_directories(.)

# cuda
find_package(CUDA)
list( APPEND libraries ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_npps_LIBRARY} )

# akaze
include_directories(/usr/local/include)
list( APPEND libraries /usr/local/lib/libAKAZE.a /usr/local/lib/libAKAZE_CUDA.a)

add_library(akaze_pybindings SHARED python_bindings.cpp pyboost_cv2_converter.cpp pyboost_cv3_converter.cpp)

if (APPLE)
  set_target_properties(akaze_pybindings PROPERTIES INSTALL_RPATH "/usr/local/lib")
endif()


target_link_libraries(akaze_pybindings ${libraries})

