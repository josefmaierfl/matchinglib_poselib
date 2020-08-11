#!/bin/bash

root_dir=`pwd`
autocalib_dir=${root_dir}/matchinglib_poselib
build_dir=${autocalib_dir}/build

BUILD_TESTS="-DOPTION_BUILD_TESTS=OFF"
BUILD_SHARED_LIBS="-DBUILD_SHARED_LIBS=ON"
INSTALL_EXE=0
if [ $# -ne 0 ]; then
  for (( i = 1; i <= "$#"; i++ )); do
    if [ "${!i}" == "exe" ]; then
      INSTALL_EXE="$((${INSTALL_EXE} + 1))"
    elif [ "${!i}" == "install" ]; then
      INSTALL_EXE="$((${INSTALL_EXE} + 2))"
    fi
  done
fi

if [ ${INSTALL_EXE} -eq 1 ] || [ ${INSTALL_EXE} -eq 3 ]; then
  BUILD_TESTS="-DOPTION_BUILD_TESTS=ON"
fi

# Build project matchinglib_poselib
mkdir ${build_dir}
cd ${build_dir}
cmake ../ -DCMAKE_BUILD_TYPE=Release ${BUILD_SHARED_LIBS} ${BUILD_TESTS}
if [ $? -ne 0 ]; then
    exit 1
fi
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
if [ ${INSTALL_EXE} -ge 2 ]; then
  sudo make install
  if [ $? -ne 0 ]; then
      exit 1
  fi
fi
