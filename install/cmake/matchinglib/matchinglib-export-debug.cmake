#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "matching::matchinglib" for configuration "Debug"
set_property(TARGET matching::matchinglib APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(matching::matchinglib PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/matchinglibd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/./matchinglibd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS matching::matchinglib )
list(APPEND _IMPORT_CHECK_FILES_FOR_matching::matchinglib "${_IMPORT_PREFIX}/lib/matchinglibd.lib" "${_IMPORT_PREFIX}/./matchinglibd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
