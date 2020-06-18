#!/bin/bash

root_dir=`pwd`
autocalib_dir=${root_dir}/matchinglib_poselib
build_dir=${autocalib_dir}/build

# Build project matchinglib_poselib
mkdir ${build_dir}
cd ${build_dir}
FLAGS="-DCMAKE_BUILD_TYPE=Release \
-DPYTHON_INCLUDE_DIR=$(python -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
-DPYTHON_LIBRARY=$(python -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))")"
cmake ../ ${FLAGS}
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi

copy_dir=${root_dir}/tmp/matchinglib_poselib/build
mkdir -p ${copy_dir}
find ${build_dir} -type f \( -executable -o -name \*.so -o -name \*.a \) -exec cp {} ${copy_dir} \;
