# if ("$ENV{THIRDPARTYROOT}" STREQUAL "")
# 	SET(THIRDPARTYROOT "/home/maierj/work/thirdpartyroot")
#     #message(FATAL_ERROR "THIRDPARTYROOT not set!" )
# else()
# 	SET(THIRDPARTYROOT "$ENV{THIRDPARTYROOT}")
# endif()
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
SET(THIRDPARTYROOT "${PARENT_DIR}/thirdparty")
EXECUTE_PROCESS( COMMAND bash -c "gcc --version | grep ^gcc | sed 's/^.* //g'" OUTPUT_VARIABLE GCC_VERSION)
STRING(STRIP GCC_VERSION ${GCC_VERSION})
string(REGEX REPLACE "\n$" "" GCC_VERSION "${GCC_VERSION}")


SET( SBA_FOUND "YES" )
SET( SBA_INCLUDE_DIR "${THIRDPARTYROOT}/sba-1.6")
SET( SBA_LIBRARY_DIR "${THIRDPARTYROOT}/sba-1.6/lib/linux64gcc${GCC_VERSION}")


SET(SBAlibs "sba")
set(SBA_DEBUG_POSTFIX "_d")

if(WIN32)
	set(LIBEXT ".lib")
elseif(UNIX)
	set(LIBPRE "lib")
	set(LIBEXT ".a")
endif()

FOREACH(onelib ${SBAlibs})
	set(_debug ${SBA_LIBRARY_DIR}/${LIBPRE}${onelib}${SBA_DEBUG_POSTFIX}${LIBEXT})
	set(_release ${SBA_LIBRARY_DIR}/${LIBPRE}${onelib}${LIBEXT})
	LIST(APPEND SBA_LIBRARIES "debug;${_debug};optimized;${_release}")
ENDFOREACH()
