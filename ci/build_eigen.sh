#!/bin/bash

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty
mkdir ${thirdparty_dir}

#-----------------------------------
# Eigen
#-----------------------------------

cd ${thirdparty_dir}
Eigen_VERSION=3.4.0
#wget -q http://bitbucket.org/eigen/eigen/get/${Eigen_VERSION}.tar.bz2
#tar xf ${Eigen_VERSION}.tar.bz2
#rm -rf ${Eigen_VERSION}.tar.bz2
#mv eigen-eigen-* eigen-eigen-${Eigen_VERSION}
#cd eigen-eigen-${Eigen_VERSION}
mkdir eigen-${Eigen_VERSION}
git clone --depth 1 --branch ${Eigen_VERSION} https://gitlab.com/libeigen/eigen.git eigen-${Eigen_VERSION}
cd eigen-${Eigen_VERSION}
mkdir -p build && cd build
cmake ..  -DCMAKE_BUILD_TYPE=Release -DOpenGL_GL_PREFERENCE=GLVND
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
sudo make install

if [ $? -ne 0 ]; then
    exit 1
fi
