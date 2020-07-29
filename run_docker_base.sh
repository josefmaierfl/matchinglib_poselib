#!/bin/bash

RES_SV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )/res_save_compressed"
if [ ! -d ${RES_SV_DIR} ]; then
  mkdir ${RES_SV_DIR}
fi

SECOND_ARG="$2"
if [ "${SECOND_ARG}" == "RESDIR" ]; then
  RES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )/$3"
  if [ -d ${RES_DIR} ]; then
    shift 2
  else
    echo "Given directory for storing results does not exist"
    exit 1
  fi
else
  RES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )/results"
  if [ ! -d ${RES_DIR} ]; then
    mkdir ${RES_DIR}
  fi
fi
# -c $(echo "${@:2}")
xhost +local:root
#docker run -v `pwd`/images:/app/images:ro -v `pwd`/py_test_scripts:/app/py_test_scripts -v ${RES_DIR}:/app/results -v ${RES_SV_DIR}:/app/res_save_compressed -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro ac_test_package:1.0 /bin/bash
docker run -v `pwd`/images:/app/images:ro -v `pwd`/py_test_scripts:/app/py_test_scripts -v ${RES_DIR}:/app/results -v ${RES_SV_DIR}:/app/res_save_compressed -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY poselib:1.0 /app/start_testing.sh "${@:2}"
xhost -local:root
