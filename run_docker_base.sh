#!/bin/bash

RES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )/results"
mkdir ${RES_DIR}
RES_SV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )/res_save_compressed"
mkdir ${RES_SV_DIR}

xhost +local:
#nvidia-docker run -v `pwd`:/app -v /home/maierj:/home/maierj -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:ro git-service.ait.ac.at:8010/maierj/generatevirtualsequence /bin/bash
#docker run -v `pwd`:/app -v /home/maierj:/home/maierj -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro ac_test_package:1.0 /bin/bash
docker run -v `pwd`/images:/app/images:ro -v `pwd`/py_test_scripts:/app/py_test_scripts -v ${RES_DIR}:/app/results -v ${RES_SV_DIR}:/app/res_save_compressed -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro ac_test_package:1.0 /bin/bash

# Shut down if asked for
if [ $# -ne 0 ]; then
    ARG_NAME="$1"
    if [ "${ARG_NAME}" == "shutdown" ]; then
        echo "Shutting down"
        sudo shutdown -h now
    fi
fi
