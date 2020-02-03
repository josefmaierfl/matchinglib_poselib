#!/bin/bash

root_dir=`pwd`
virtsequ_dir=${root_dir}/generateVirtualSequence
build_dir=${virtsequ_dir}/build

# Build project generateVirtualSequence
mkdir ${build_dir}
cd ${build_dir}
cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j8

copy_dir=${root_dir}/tmp/generateVirtualSequence/build
mkdir -p ${copy_dir}
find ${build_dir} -type f \( -executable -o -name \*.so \) -exec cp {} ${copy_dir} \;
