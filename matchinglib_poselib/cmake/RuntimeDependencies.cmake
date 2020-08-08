
# 
# Default dependencies for the runtime-package
# 

# Install 3rd-party runtime dependencies into runtime-component
#find_package(Clapack REQUIRED)
#message(STATUS "Installing: ${CLAPACK_INCLUDE_DIR}")
#install(FILES ${CLAPACK_INCLUDE_DIR}/blaswrap.h COMPONENT runtime)


# 
# Full dependencies for self-contained packages
# 

if(OPTION_SELF_CONTAINED)

    # Install 3rd-party runtime dependencies into runtime-component
    # install(FILES ... COMPONENT runtime)

endif()
