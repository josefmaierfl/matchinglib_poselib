#!/bin/bash

OPT_BUILD_SHARED_LIBS="OFF"
FLAG_BUILD_SHARED_LIBS="-DBUILD_SHARED_LIBS=${OPT_BUILD_SHARED_LIBS}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#BLAS_P="$( cd $DIR && cd ../../../clapack-3.2.1/BLAS && pwd )"
pushd $DIR
cd ../..
mkdir lib
cd ${DIR}
cd ../
mkdir linux
cd linux
CACHEFILE=./CMakeCache.txt
if [ -f "$CACHEFILE" ]; then
    rm $CACHEFILE
fi
cmake ../../ -DCMAKE_BUILD_TYPE=Release ${FLAG_BUILD_SHARED_LIBS}
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
sudo make install
popd

cd ../
mkdir linux_debug
cd linux_debug
if [ -f "$CACHEFILE" ]; then
    rm $CACHEFILE
fi
cmake ../../ -DCMAKE_BUILD_TYPE=Debug ${FLAG_BUILD_SHARED_LIBS}
make -j "$(nproc)"
sudo make install
if [ $? -ne 0 ]; then
    exit 1
fi
