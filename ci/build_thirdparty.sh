#!/bin/bash

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty
mkdir ${thirdparty_dir}

#-----------------------------------
# Eigen
#-----------------------------------

cd ${thirdparty_dir}
Eigen_VERSION=3.3.7
wget -q http://bitbucket.org/eigen/eigen/get/${Eigen_VERSION}.tar.bz2
tar xf ${Eigen_VERSION}.tar.bz2
rm -rf ${Eigen_VERSION}.tar.bz2
mv eigen-eigen-* eigen-eigen-${Eigen_VERSION}
cd eigen-eigen-${Eigen_VERSION}
mkdir -p build && cd build
cmake ..  -DCMAKE_BUILD_TYPE=Release
make -j "$(nproc)"
make install

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

cd ${thirdparty_dir}
PCL_VERSION=1.9.1
git clone https://github.com/PointCloudLibrary/pcl.git -b pcl-${PCL_VERSION} pcl-${PCL_VERSION}
mkdir -p pcl-${PCL_VERSION}/build
cd pcl-${PCL_VERSION}/build
cmake ..  -DBUILD_2d:BOOL=ON -DBUILD_CUDA:BOOL=OFF -DBUILD_GPU:BOOL=OFF -DBUILD_apps:BOOL=OFF -DBUILD_common:BOOL=ON -DBUILD_examples:BOOL=OFF -DBUILD_features:BOOL=OFF -DBUILD_filters:BOOL=ON -DBUILD_geometry:BOOL=ON -DBUILD_global_tests:BOOL=OFF -DBUILD_io:BOOL=ON -DBUILD_kdtree:BOOL=ON -DBUILD_keypoints:BOOL=OFF -DBUILD_ml:BOOL=OFF -DBUILD_octree:BOOL=ON -DBUILD_outofcore:BOOL=OFF -DBUILD_people:BOOL=OFF -DBUILD_recognition:BOOL=OFF -DBUILD_registration:BOOL=OFF -DBUILD_sample_consensus:BOOL=ON -DBUILD_search:BOOL=ON -DBUILD_segmentation:BOOL=OFF -DBUILD_simulation:BOOL=OFF -DBUILD_stereo:BOOL=OFF -DBUILD_surface:BOOL=OFF -DBUILD_tools:BOOL=OFF -DBUILD_tracking:BOOL=OFF -DBUILD_visualization:BOOL=ON
make -j "$(nproc)"
make install

#-----------------------------------
# OpenCV
#-----------------------------------

cd ${thirdparty_dir}
${root_dir}/make_opencv.sh
