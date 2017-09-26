if ("$ENV{THIRDPARTYROOT}" STREQUAL "")
    message(FATAL_ERROR "THIRDPARTYROOT not set!" )
endif()
SET(THIRDPARTYROOT "$ENV{THIRDPARTYROOT}")

SET( SBA_FOUND "YES" )
SET( SBA_INCLUDE_DIR "${THIRDPARTYROOT}/sba-1.6")
SET( SBA_LIBRARY_DIR "${THIRDPARTYROOT}/sba-1.6/lib/linux64gcc48")



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
