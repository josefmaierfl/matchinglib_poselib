#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "matching::poselib" for configuration "Debug"
set_property(TARGET matching::poselib APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(matching::poselib PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/poselibd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/./poselibd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS matching::poselib )
list(APPEND _IMPORT_CHECK_FILES_FOR_matching::poselib "${_IMPORT_PREFIX}/lib/poselibd.lib" "${_IMPORT_PREFIX}/./poselibd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
