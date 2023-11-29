#!/bin/bash

#-----------------------------------
# Ceres
#-----------------------------------

root_dir=`pwd`
# thirdparty_dir=${root_dir}/thirdparty

BUILD_FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON"
if [[ $# -ne 0 ]] && [[ "$1" == "-cuda" ]]; then
  BUILD_FLAGS="${BUILD_FLAGS} -DUSE_CUDA=ON"
else
  BUILD_FLAGS="${BUILD_FLAGS} -DUSE_CUDA=OFF"
fi

cd ${root_dir}
CERES_VERSION=2.2.0
wget -q http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz
tar zxf ceres-solver-${CERES_VERSION}.tar.gz
rm -rf ceres-solver-${CERES_VERSION}.tar.gz
cd ceres-solver-${CERES_VERSION}
mkdir build
cd build
cmake .. ${BUILD_FLAGS}
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
# make test
# if [ $? -ne 0 ]; then
    # exit 1
# fi
make install

if [ $? -ne 0 ]; then
    exit 1
fi
