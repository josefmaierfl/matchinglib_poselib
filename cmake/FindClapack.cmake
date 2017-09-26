if ("$ENV{THIRDPARTYROOT}" STREQUAL "")
    message(FATAL_ERROR "THIRDPARTYROOT not set!" )
endif()
SET(THIRDPARTYROOT "$ENV{THIRDPARTYROOT}")

SET( CLAPACK_FOUND "YES" )
SET( CLAPACK_LIBRARY_DIR "${THIRDPARTYROOT}/clapack-3.2.1/lib/linux64gcc48")
SET( CLAPACK_INCLUDE_DIR "${THIRDPARTYROOT}/clapack-3.2.1/INCLUDE")


SET(CLAPACKlibs "blas" "lapack" "tmglib")

set(CLAPACK_DEBUG_POSTFIX "d")

if(WIN32)
	LIST(APPEND CLAPACKlibs "libf2c")
	set(LIBEXT ".lib")
elseif(UNIX)
	LIST(APPEND CLAPACKlibs "f2c")
	set(LIBPRE "lib")
	set(LIBEXT ".a")
endif()

FOREACH(onelib ${CLAPACKlibs})
	set(_debug ${CLAPACK_LIBRARY_DIR}/${LIBPRE}${onelib}${CLAPACK_DEBUG_POSTFIX}${LIBEXT})
	set(_release ${CLAPACK_LIBRARY_DIR}/${LIBPRE}${onelib}${LIBEXT})
	LIST(APPEND CLAPACK_LIBRARIES "debug;${_debug};optimized;${_release}")
ENDFOREACH()

SET( Clapack_INCLUDE_DIR "${CLAPACK_INCLUDE_DIR}")
SET( Clapack_LIBRARIES "${CLAPACK_LIBRARIES}")
