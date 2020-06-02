#!/bin/bash

#-----------------------------------
# Ceres
#-----------------------------------

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty

cd ${thirdparty_dir}
CERES_VERSION=1.14.0
wget -q http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz
tar zxf ceres-solver-${CERES_VERSION}.tar.gz
rm -rf ceres-solver-${CERES_VERSION}.tar.gz
cd ceres-solver-${CERES_VERSION}
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
make install

if [ $? -ne 0 ]; then
    exit 1
fi
