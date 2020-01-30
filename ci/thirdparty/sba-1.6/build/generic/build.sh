#dependencies
#sudo apt-get install cmake build-essential libopencv-dev

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BLAS_P="$( cd "$( dirname "../../../$DIR/clapack-3.2.1/BLAS" )" && pwd )"
pushd $DIR
cd ../
mkdir linux
cd linux
rm ./CMakeCache.txt
cmake ../../ -DCMAKE_BUILD_TYPE=Release
make -j8
popd

cd ../
mkdir linux_debug
cd linux_debug
rm ./CMakeCache.txt
cmake ../../ -DCMAKE_BUILD_TYPE=Debug -DLAPACKBLAS_DIR="${BLAS_P}"
make -j8
