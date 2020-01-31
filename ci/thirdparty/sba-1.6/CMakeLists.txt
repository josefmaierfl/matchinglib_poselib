# sba CMake file; see http://www.cmake.org and
#                     http://www.insightsoftwareconsortium.org/wiki/index.php/CMake_Tutorial

PROJECT(SBA)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

EXECUTE_PROCESS( COMMAND bash -c "gcc --version | grep ^gcc | sed 's/^.* //g'" OUTPUT_VARIABLE GCC_VERSION)
STRING(STRIP GCC_VERSION ${GCC_VERSION})
string(REGEX REPLACE "\n$" "" GCC_VERSION "${GCC_VERSION}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/linux64gcc${GCC_VERSION})
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/linux64gcc${GCC_VERSION})
#does not have any effect on executables on its own only with
#   set_target_properties(${PROJ_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
#   after the target definition (see end of file)
set(CMAKE_DEBUG_POSTFIX _d)



# f2c is sometimes equivalent to libF77 & libI77; in that case, set HAVE_F2C to 0
SET(HAVE_F2C 1 CACHE BOOL "Do we have f2c or F77/I77?" )

# the directory where the lapack/blas/f2c libraries reside
SET(LAPACKBLAS_DIR /usr/lib CACHE PATH "Path to lapack/blas libraries")

# actual names for the lapack/blas/f2c libraries
SET(LAPACK_LIB lapack CACHE STRING "The name of the lapack library")
SET(BLAS_LIB blas CACHE STRING "The name of the blas library")
IF(HAVE_F2C)
  SET(F2C_LIB f2c CACHE STRING "The name of the f2c library")
ELSE(HAVE_F2C)
  SET(F77_LIB libF77 CACHE STRING "The name of the F77 library")
  SET(I77_LIB libI77 CACHE STRING "The name of the I77 library")
ENDIF(HAVE_F2C)


IF(UNIX)
  set_target_properties(${PROJ_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
  #add_definitions(-DSBA_APPEND_UNDERSCORE_SUFFIX)
ENDIF(UNIX)

########################## NO CHANGES BEYOND THIS POINT ##########################

INCLUDE_DIRECTORIES(.)
# sba library source files
ADD_LIBRARY(sba STATIC
  sba_levmar.c sba_levmar_wrap.c sba_lapack.c sba_crsm.c sba_chkjac.c
  sba.h sba_chkjac.h compiler.h
)

if(MSVC)
	foreach(flag_var
			CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
			CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
			CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
	   if(${flag_var} MATCHES "/MT")
		  string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
	   endif(${flag_var} MATCHES "/MT")
	endforeach(flag_var)
endif()

set_target_properties(sba
    PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
)

set_target_properties(sba PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})


#ADD_SUBDIRECTORY(demo)
#SUBDIRS(demo)
