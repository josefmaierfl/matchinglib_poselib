#!/bin/bash

OPT_BUILD_SHARED_LIBS="ON"
FLAG_BUILD_SHARED_LIBS="-DBUILD_SHARED_LIBS=${OPT_BUILD_SHARED_LIBS}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BLAS_P="$( cd $DIR && cd ../../../clapack-3.2.1/BLAS && pwd )"
pushd $DIR
cd ../..
mkdir lib
cd ${DIR}
cd ../
mkdir linux
cd linux
rm ./CMakeCache.txt
cmake ../../ -DCMAKE_BUILD_TYPE=Release -DLAPACKBLAS_DIR="${BLAS_P}" ${FLAG_BUILD_SHARED_LIBS}
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
sudo make install
popd

cd ../
mkdir linux_debug
cd linux_debug
rm ./CMakeCache.txt
cmake ../../ -DCMAKE_BUILD_TYPE=Debug -DLAPACKBLAS_DIR="${BLAS_P}" ${FLAG_BUILD_SHARED_LIBS}
make -j "$(nproc)"
sudo make install
if [ $? -ne 0 ]; then
    exit 1
fi
