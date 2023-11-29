#!/bin/bash

root_dir=`pwd`
autocalib_dir=${root_dir}/matchinglib_poselib
build_dir=${autocalib_dir}/build

CUDA=false
USE_CERES=true
if [ $# -ne 0 ]; then
  for (( i = 1; i <= "$#"; i++ )); do
    if [ "${!i}" == "-cuda" ]; then
      CUDA=true
    elif [ "${!i}" == "-noceres" ]; then
      USE_CERES=false
    fi
  done
fi

# Build project matchinglib_poselib
mkdir ${build_dir}
cd ${build_dir}
FLAGS="-DCMAKE_BUILD_TYPE=Release"
#FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DOPTION_BUILD_TESTS=OFF"
if [[ "${CUDA}" == true ]]; then
  FLAGS="${FLAGS} -DBUILD_WITH_AKAZE_CUDA=ON"
fi
if [[ "${USE_CERES}" == true ]]; then
  FLAGS="${FLAGS} -DBUILD_WITH_CERES=ON"
else
  FLAGS="${FLAGS} -DBUILD_WITH_CERES=OFF"
fi


cmake ../ ${FLAGS}
if [ $? -ne 0 ]; then
    exit 1
fi
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
#make install

copy_dir=${root_dir}/tmp/matchinglib_poselib/build
mkdir -p ${copy_dir}
find ${build_dir} -type f \( -executable -o -name \*.so -o -name \*.a \) -exec cp {} ${copy_dir} \;
IMG_DIR=${build_dir}/imgs
if [ -d ${IMG_DIR} ]; then
  cp -r ${IMG_DIR} ${copy_dir}
fi
