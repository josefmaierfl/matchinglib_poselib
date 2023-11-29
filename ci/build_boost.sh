#!/bin/bash

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty

cd ${thirdparty_dir}
Boost_VERSION=1.71.0
Boost_ROOT="boost_$(echo ${Boost_VERSION} | tr . _)"
Boost_FILEN="${Boost_ROOT}.tar.bz2"
wget -q https://dl.bintray.com/boostorg/release/${Boost_VERSION}/source/${Boost_FILEN}
tar --bzip2 -xf ${Boost_FILEN}
rm -rf ${Boost_FILEN}

cd ${Boost_ROOT}
unset CPLUS_INCLUDE_PATH
PYTHON_ROOT_PATH="$(python -c "from distutils.sysconfig import get_config_h_filename; from os.path import dirname; print(dirname(get_config_h_filename()))")"
PYTHON_VERSION="$(python -c "from distutils.sysconfig import get_python_version; print(get_python_version())")"
#  --with-libraries=python
./bootstrap.sh --with-python="$(which python)" --with-python-root="${PYTHON_ROOT_PATH}" --with-python-version="${PYTHON_VERSION}"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:${PYTHON_ROOT_PATH}"
./b2 install
if [ $? -ne 0 ]; then
    exit 1
fi
TO_BASH="export CPLUS_INCLUDE_PATH=\\\$CPLUS_INCLUDE_PATH:${PYTHON_ROOT_PATH}"
id -u conan
if [ $? -ne 1 ]; then
  sudo -H -u conan bash -c "echo ${TO_BASH} >> ~/.profile"
fi
echo "export CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:${PYTHON_ROOT_PATH}" >> ~/.bashrc
