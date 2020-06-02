#!/bin/bash

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty
mkdir ${thirdparty_dir}

cd ${thirdparty_dir}
PCL_VERSION=1.11.0
git clone https://github.com/PointCloudLibrary/pcl.git -b pcl-${PCL_VERSION} pcl-${PCL_VERSION}
mkdir -p pcl-${PCL_VERSION}/build
cd pcl-${PCL_VERSION}/build
PCL_FLAGS="-DPCL_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS=-std=c++14"
PCL_MODULES="-DBUILD_2d:BOOL=ON -DBUILD_CUDA:BOOL=OFF -DBUILD_GPU:BOOL=OFF -DBUILD_apps:BOOL=OFF -DBUILD_common:BOOL=ON -DBUILD_examples:BOOL=OFF -DBUILD_features:BOOL=OFF -DBUILD_filters:BOOL=ON -DBUILD_geometry:BOOL=ON -DBUILD_global_tests:BOOL=OFF -DBUILD_io:BOOL=ON -DBUILD_kdtree:BOOL=ON -DBUILD_keypoints:BOOL=OFF -DBUILD_ml:BOOL=OFF -DBUILD_octree:BOOL=ON -DBUILD_outofcore:BOOL=OFF -DBUILD_people:BOOL=OFF -DBUILD_recognition:BOOL=OFF -DBUILD_registration:BOOL=OFF -DBUILD_sample_consensus:BOOL=ON -DBUILD_search:BOOL=ON -DBUILD_segmentation:BOOL=OFF -DBUILD_simulation:BOOL=OFF -DBUILD_stereo:BOOL=OFF -DBUILD_surface:BOOL=OFF -DBUILD_tools:BOOL=OFF -DBUILD_tracking:BOOL=OFF -DBUILD_visualization:BOOL=ON"
cmake .. $PCL_FLAGS $PCL_MODULES
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
make install

if [ $? -ne 0 ]; then
    exit 1
fi
