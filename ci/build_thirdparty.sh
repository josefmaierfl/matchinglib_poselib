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
# Clapack 3.2.1
#-----------------------------------

cd ${thirdparty_dir}/clapack-3.2.1/build/generic
./build.sh

#-----------------------------------
# SBA 1.6
#-----------------------------------

cd ${thirdparty_dir}/sba-1.6/build/generic
./build.sh

#-----------------------------------
# Copy necessary files of Clapack 3.2.1 and SBA 1.6
#-----------------------------------

# cd ${root_dir} && ./copy_thirdparty.sh

#-----------------------------------
# PCL
#-----------------------------------

cd ${root_dir}
./build_pcl.sh

if [ $? -ne 0 ]; then
    exit 1
fi

#-----------------------------------
# Ceres
#-----------------------------------

cd ${root_dir}
./build_ceres.sh

if [ $? -ne 0 ]; then
    exit 1
fi

#-----------------------------------
# OpenCV
#-----------------------------------

cd ${thirdparty_dir}
${root_dir}/make_opencv.sh
ldconfig
