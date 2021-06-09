#!/bin/bash

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty
mkdir ${thirdparty_dir}

#-----------------------------------
# Eigen
#-----------------------------------

cd ${root_dir}
./build_eigen.sh

if [ $? -ne 0 ]; then
    exit 1
fi

#-----------------------------------
# Boost
#-----------------------------------
# cd ${root_dir}
# ./build_boost.sh
#
# if [ $? -ne 0 ]; then
#     exit 1
# fi
# ldconfig

#-----------------------------------
# OpenCV
#-----------------------------------

cd ${thirdparty_dir}
${root_dir}/make_opencv.sh

if [ $? -ne 0 ]; then
    exit 1
fi
ldconfig
