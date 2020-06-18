#!/bin/bash

#-----------------------------------
# VTK
#-----------------------------------

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty

cd ${thirdparty_dir}
VTK_VERSION=8.2.0
VTK_ROOT="vtk_$(echo ${VTK_VERSION} | tr . _)"
git clone https://gitlab.kitware.com/vtk/vtk.git ${VTK_ROOT}
cd ${VTK_ROOT}
git submodule update --init
git checkout "v${VTK_VERSION}"
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DQT_QMAKE_EXECUTABLE="$(whereis qmake)"
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
make install

if [ $? -ne 0 ]; then
    exit 1
fi
