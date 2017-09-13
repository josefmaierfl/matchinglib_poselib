# OpenGV: https://github.com/laurentkneip/opengv
# Example for linking in your project:
#
# find_package(OpenGV REQUIRED)
# if (OPENGV_FOUND STREQUAL "YES")
#  message("OpenGV found!")
# else()
#  message(FATAL_ERROR "OpenGV not found!")
# endif()
#
# target_include_directories(${target} ${OPENGV_INCLUDE_DIRS})
# target_link_libraries(${target} ${OPENGV_LIBRARIES})


# SET( OPENGV_FOUND "YES" )
# SET( OPENGV_INCLUDE_DIRS "${THIRDPARTYROOT_DETECTED}/OpenGV/include")
# SET( OPENGV_LIBRARIES "${THIRDPARTYROOT_DETECTED}/OpenGV/lib/win7x64vs10/opengv.lib"
                      # "${THIRDPARTYROOT_DETECTED}/OpenGV/lib/win7x64vs10/random_generators.lib"
                      # )
					  
SET( OPENGV_FOUND "YES" )
SET( OPENGV_INCLUDE_DIR "/home/martin/thirdpartyroot/OpenGV/include")
SET( OPENGV_LIBRARY_DIR "/home/martin/thirdpartyroot/OpenGV/lib/linux64gcc48")

SET(OPENGVlibs "opengv" "random_generators")
set(OPENGV_DEBUG_POSTFIX "_d")

if(WIN32)
	set(LIBEXT ".lib")
elseif(UNIX)
	set(LIBPRE "lib")
	set(LIBEXT ".a")
endif()

FOREACH(onelib ${OPENGVlibs})
	set(_debug ${OPENGV_LIBRARY_DIR}/${LIBPRE}${onelib}${OPENGV_DEBUG_POSTFIX}${LIBEXT})
	set(_release ${OPENGV_LIBRARY_DIR}/${LIBPRE}${onelib}${LIBEXT})
	LIST(APPEND OPENGV_LIBRARIES "debug;${_debug};optimized;${_release}")
ENDFOREACH()
