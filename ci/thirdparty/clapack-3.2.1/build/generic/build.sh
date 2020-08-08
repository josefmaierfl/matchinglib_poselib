#!/bin/bash

#packages
#sudo apt-get install cmake build-essential

OPT_BUILD_SHARED_LIBS="OFF"
FLAG_BUILD_SHARED_LIBS="-DBUILD_SHARED_LIBS=${OPT_BUILD_SHARED_LIBS}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GCC_VER="$(gcc --version | grep ^gcc | sed 's/^.* //g')"
pushd $DIR
cd ../
mkdir linux
cd linux
rm ./CMakeCache.txt
cmake ../../ -DCMAKE_BUILD_TYPE=Release ${FLAG_BUILD_SHARED_LIBS}
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
sudo make install
if [ $? -ne 0 ]; then
    exit 1
fi
popd

mkdir -p "../../lib/linux64gcc${GCC_VER}"
if [ "${OPT_BUILD_SHARED_LIBS}" == "OFF" ]; then
  find ../linux -name  \*.a -type f -exec cp {} "../../lib/linux64gcc${GCC_VER}" \;
else
  find ../linux -name  \*.so -type f -exec cp {} "../../lib/linux64gcc${GCC_VER}" \;
fi

pushd $DIR
cd ../
rm -Rf linux_debug
mkdir linux_debug
cd linux_debug
rm ./CMakeCache.txt
cmake ../../ -DCMAKE_BUILD_TYPE=Debug ${FLAG_BUILD_SHARED_LIBS}
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
sudo make install
if [ $? -ne 0 ]; then
    exit 1
fi
popd

#if [ "${OPT_BUILD_SHARED_LIBS}" == "OFF" ]; then
  #find ../linux_debug -name  \*.a -type f -exec  sh -c 'mv ${0} ${0%.a}d.a' {}  \;
  #find ../linux_debug -name  \*.a -type f -exec cp {} "../../lib/linux64gcc${GCC_VER}" \;
#else
  #find ../linux_debug -name  \*.so -type f -exec  sh -c 'mv ${0} ${0%.so}d.so' {}  \;
  #find ../linux_debug -name  \*.so -type f -exec cp {} "../../lib/linux64gcc${GCC_VER}" \;
#fi
