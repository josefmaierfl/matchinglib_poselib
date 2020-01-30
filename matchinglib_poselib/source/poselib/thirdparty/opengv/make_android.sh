BUILD_TYPE=Release
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VSLAMFW_PATH=$THISDIR


# ANDROID
TARGET=android

# 3rdParty
export ANDROID_NDK=/home/schoerghuberm/work/android-ndk-r12
OPENCV=-DOpenCV_DIR=/host/dev/thirdparty/OpenCV-3.0/platforms/build_android_arm/install/sdk/native/jni
EIGEN=-DEIGEN3_INCLUDE_DIR=/host/dev/3rdParty/Eigen-3.1.3

# build
TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE=/home/schoerghuberm/work/vslamfw/platforms/toolchains/android.toolchain.cmake
CMAKEEXT="$TOOLCHAIN $OPENCV $EIGEN -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=19"


# vslamfw
cd $THISDIR
BUILD_DIR=build_vslamfw_$TARGET
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
echo cmake $CMAKEEXT $VSLAMFW_PATH -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=install -DBUILD_DEMO=OFF
cmake $CMAKEEXT $VSLAMFW_PATH -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX=install -DBUILD_DEMO=OFF
make
make install
