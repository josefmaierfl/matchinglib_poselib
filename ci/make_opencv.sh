#!/usr/bin/env bash

# ===========================================
# VERSION
# ===========================================
VERSION=${VERSION:=4.2.0}

OPENCV_DIR=$PWD/opencv-$VERSION
OPENCV_CONTRIB_DIR=$PWD/opencv_contrib-$VERSION

# ===========================================
# Download
# ===========================================

git clone https://github.com/opencv/opencv $OPENCV_DIR -b $VERSION
git clone https://github.com/opencv/opencv_contrib $OPENCV_CONTRIB_DIR -b $VERSION

# ===========================================
# CONFIG
# ===========================================

BUILD_DIR=opencv-$VERSION/build
#mkdir -p $BUILD_DIR
CMAKE_OPENCV_EXTRA_MODULES=-DOPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB_DIR/modules

FLAGS="-DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_APPS=OFF -DBUILD_TESTS=OFF -DWITH_CUDA=OFF -DWITH_IPP=ON -DWITH_TBB=ON -DWITH_V4L=ON -DWITH_QT=ON -DWITH_OPENGL=ON -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 -DENABLE_AVX=ON \
-DHAVE_opencv_python3=ON -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DINSTALL_PYTHON_EXAMPLES=OFF \
-DBUILD_EXAMPLES=OFF \
-DPYTHON3_EXECUTABLE=$(which python3) \
-DPYTHON_EXECUTABLE=$(which python3) \
-DPYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
-DPYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
-DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
-DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")"

# ===========================================
# BUILD
# ===========================================
mkdir -p $BUILD_DIR
cd $BUILD_DIR

OPENCV_MODULES="-DBUILD_opencv_apps:BOOL=OFF -DBUILD_opencv_aruco:BOOL=OFF -DBUILD_opencv_bgsegm:BOOL=OFF -DBUILD_opencv_bioinspired:BOOL=OFF -DBUILD_opencv_calib3d:BOOL=ON -DBUILD_opencv_ccalib:BOOL=ON -DBUILD_opencv_core:BOOL=ON -DBUILD_opencv_datasets:BOOL=OFF -DBUILD_opencv_dnn:BOOL=OFF -DBUILD_opencv_dnn_objdetect:BOOL=OFF -DBUILD_opencv_dpm:BOOL=OFF -DBUILD_opencv_face:BOOL=OFF -DBUILD_opencv_features2d:BOOL=ON -DBUILD_opencv_flann:BOOL=ON -DBUILD_opencv_freetype:BOOL=OFF -DBUILD_opencv_fuzzy:BOOL=OFF -DBUILD_opencv_gapi:BOOL=OFF -DBUILD_opencv_hdf:BOOL=ON -DBUILD_opencv_hfs:BOOL=OFF -DBUILD_opencv_highgui:BOOL=ON -DBUILD_opencv_img_hash:BOOL=OFF -DBUILD_opencv_imgcodecs:BOOL=ON -DBUILD_opencv_imgproc:BOOL=ON -DBUILD_opencv_java_bindings_generator:BOOL=OFF -DBUILD_opencv_js:BOOL=OFF -DBUILD_opencv_line_descriptor:BOOL=OFF -DBUILD_opencv_ml:BOOL=OFF -DBUILD_opencv_objdetect:BOOL=OFF -DBUILD_opencv_optflow:BOOL=ON -DBUILD_opencv_phase_unwrapping:BOOL=OFF -DBUILD_opencv_photo:BOOL=OFF -DBUILD_opencv_plot:BOOL=OFF -DBUILD_opencv_python_bindings_generator:BOOL=OFF -DBUILD_opencv_python_tests:BOOL=OFF -DBUILD_opencv_quality:BOOL=OFF -DBUILD_opencv_reg:BOOL=OFF -DBUILD_opencv_rgbd:BOOL=OFF -DBUILD_opencv_saliency:BOOL=OFF -DBUILD_opencv_shape:BOOL=OFF -DBUILD_opencv_stereo:BOOL=ON -DBUILD_opencv_stitching:BOOL=OFF -DBUILD_opencv_structured_light:BOOL=OFF -DBUILD_opencv_superres:BOOL=OFF -DBUILD_opencv_surface_matching:BOOL=OFF -DBUILD_opencv_text:BOOL=OFF -DBUILD_opencv_tracking:BOOL=ON -DBUILD_opencv_video:BOOL=ON -DBUILD_opencv_videoio:BOOL=ON -DBUILD_opencv_videostab:BOOL=OFF -DBUILD_opencv_viz:BOOL=OFF -DBUILD_opencv_world:BOOL=OFF -DBUILD_opencv_xfeatures2d:BOOL=ON -DBUILD_opencv_ximgproc:BOOL=ON -DBUILD_opencv_xobjdetect:BOOL=OFF -DBUILD_opencv_xphoto:BOOL=OFF"

cmake $OPENCV_DIR $CMAKE_OPENCV_EXTRA_MODULES $FLAGS $PACK_FLAGS ${OPENCV_MODULES}
make -j "$(nproc)"
if [ $? -ne 0 ]; then
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    exit 1
fi
