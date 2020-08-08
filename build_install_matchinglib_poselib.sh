#!/bin/bash

root_dir=`pwd`
autocalib_dir=${root_dir}/matchinglib_poselib
build_dir=${autocalib_dir}/build

# Build project matchinglib_poselib
mkdir ${build_dir}
cd ${build_dir}
FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOPTION_BUILD_TESTS=OFF"
cmake ../ ${FLAGS}
if [ $? -ne 0 ]; then
    exit 1
fi
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
sudo make install
if [ $? -ne 0 ]; then
    exit 1
fi
