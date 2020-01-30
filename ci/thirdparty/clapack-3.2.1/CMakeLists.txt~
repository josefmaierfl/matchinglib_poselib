cmake_minimum_required(VERSION 2.6)
project(CLAPACK C)

option(CLAPACK_BUILD_TESTING "Build CLAPACK tests" ON)

include(CTest)

if(WIN32 AND NOT CYGWIN)
  set(SECOND_SRC  ${CLAPACK_SOURCE_DIR}/INSTALL/winsecond.c)
  set(DSECOND_SRC  ${CLAPACK_SOURCE_DIR}/INSTALL/windsecnd.c)
  add_definitions(-DNO_ISATTY -DMSDOS -DUSE_CLOCK)
else()
  set(SECOND_SRC  ${CLAPACK_SOURCE_DIR}/INSTALL/second.c)
  set(DSECOND_SRC  ${CLAPACK_SOURCE_DIR}/INSTALL/dsecnd.c)
endif()

option(USE_BLAS_WRAP "pre-pend f2c_ to each function in blas" OFF)
if(NOT USE_BLAS_WRAP)
# _zrotg_ seems to be missing in the wrap header
  add_definitions(-DNO_BLAS_WRAP)
endif()

if (NOT CLAPACK_INSTALL_LIB_DIR)
  set(CLAPACK_INSTALL_LIB_DIR lib)
endif()

# lib install macro
macro( install_library TARGET_NAME )
    install(TARGETS ${TARGET_NAME}
        EXPORT ${PROJECT_NAME} 
        ARCHIVE DESTINATION ${CLAPACK_INSTALL_LIB_DIR} 
        LIBRARY DESTINATION ${CLAPACK_INSTALL_LIB_DIR}
        RUNTIME DESTINATION bin
        )
endmacro( install_library )

# set compiler options (run once)
if( NOT SETUP_COMPILER_FLAGS_HAS_RUN )
  if(MSVC) # silence some warnings
    set( WARN_FLAGS "${WARN_FLAGS} /wd4244" ) # conversion from 'doublereal' to 'real', possible loss of data
    set( WARN_FLAGS "${WARN_FLAGS} /wd4554" ) # check operator precedence for possible error; use parentheses to clarify precedence
    set( WARN_FLAGS "${WARN_FLAGS} /wd4996" ) # This function or variable may be unsafe. Consider using freopen_s instead.
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARN_FLAGS}"
         CACHE STRING "Flags used by the C compiler during all build types." FORCE )
    set( CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Append this string to debug lib names")
  elseif (CMAKE_COMPILER_IS_GNUCC)
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC"
             CACHE STRING "Flags used by the C compiler during all build types." FORCE )
  elseif (CMAKE_COMPILER_IS_GNUCXX)
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC"
             CACHE STRING "Flags used by the C compiler during all build types." FORCE )
  endif()
  set( SETUP_COMPILER_FLAGS_HAS_RUN TRUE CACHE INTERNAL "one-time setup of compiler options" FORCE)
endif( NOT SETUP_COMPILER_FLAGS_HAS_RUN )

include_directories(${CLAPACK_SOURCE_DIR}/INCLUDE)
add_subdirectory(INCLUDE)
add_subdirectory(F2CLIBS)
add_subdirectory(BLAS)
add_subdirectory(SRC)
if (CLAPACK_BUILD_TESTING)
    enable_testing()
    add_subdirectory(TESTING)
endif()
set(CLAPACK_VERSION 3.2.1)
set(CPACK_PACKAGE_VERSION_MAJOR 3)
set(CPACK_PACKAGE_VERSION_MINOR 2)
set(CPACK_PACKAGE_VERSION_PATCH 1)

include(CPack)

configure_file(${CLAPACK_SOURCE_DIR}/clapack-config-version.cmake.in
  ${CLAPACK_BINARY_DIR}/clapack-config-version.cmake @ONLY)

set(CLAPACK_INCLUDE_DIRECTORIES ${CMAKE_INSTALL_PREFIX}/include)
set(CLAPACK_LIBRARY_DIRECTORIES ${CMAKE_INSTALL_PREFIX}/${CLAPACK_INSTALL_LIB_DIR})
set(CLAPACK_LIBRARIES blas lapack f2c)

set(INSTALL_CMAKE_DIR share/cmake/clapack)
if (WIN32)
    set(INSTALL_CMAKE_DIR cmake)
elseif(APPLE)
    set(INSTALL_CMAKE_DIR .framework/Resources/CMake/)
endif()

configure_file(${CLAPACK_SOURCE_DIR}/clapack-config.cmake.in
  ${CLAPACK_BINARY_DIR}/clapack-config.cmake @ONLY)

install(EXPORT ${PROJECT_NAME} DESTINATION ${INSTALL_CMAKE_DIR})

install(FILES ${CLAPACK_BINARY_DIR}/clapack-config.cmake 
    DESTINATION ${INSTALL_CMAKE_DIR})
