#!/bin/bash

root_dir=`pwd`
thirdparty_dir=${root_dir}/thirdparty
tmp_dir=${root_dir}/tmp
mkdir ${tmp_dir}
cd ${tmp_dir}
mkdir thirdparty && cd thirdparty

mkdir sba-1.6
cd sba-1.6 && cp -r ${thirdparty_dir}/sba-1.6/lib ./
find ${thirdparty_dir}/sba-1.6 -type f \( -name \*.h -o -name \*.c \) -exec cp {} . \;

#mkdir clapack-3.2.1
#cd ../clapack-3.2.1 && cp -r ${thirdparty_dir}/clapack-3.2.1/lib ./ && cp -r ${thirdparty_dir}/clapack-3.2.1/INCLUDE ./
