#!/bin/bash

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty
mkdir ${thirdparty_dir}

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
# Ceres
#-----------------------------------

if [[ "${USE_CERES}" == true ]]; then
  cd ${root_dir}
  # if [[ "${CUDA}" == true ]]; then
    # echo "Installing Ceres with CUDA support"
    # ./build_ceres.sh -cuda
  # else
    echo "Installing Ceres"
    ./build_ceres.sh
  # fi
fi
if [ $? -ne 0 ]; then
    exit 1
fi

#-----------------------------------
# OpenCV
#-----------------------------------

cd ${thirdparty_dir}
if [[ "${CUDA}" == true ]]; then
  echo "Installing OpenCV with CUDA support"
  ${root_dir}/make_opencv_cuda.sh
else
  echo "Installing OpenCV"
  ${root_dir}/make_opencv.sh
fi
if [ $? -ne 0 ]; then
    exit 1
fi
ldconfig
