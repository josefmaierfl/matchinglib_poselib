#!/usr/bin/env bash
xhost +local:
#nvidia-docker run -v `pwd`:/app -v /home/maierj:/home/maierj -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:ro git-service.ait.ac.at:8010/maierj/generatevirtualsequence /bin/bash
docker run -v `pwd`:/app -v /home/maierj:/home/maierj -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro git-service.ait.ac.at:8010/maierj/generatevirtualsequence /bin/bash
