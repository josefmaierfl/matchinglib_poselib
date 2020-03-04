#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Arguments are required"
  exit 1
fi
FIRST_ARG="$1"
DRIVE_NAME="$2"
#Connect to server and install dependencies
scp -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" /home/maierj/work/autocalib_test_package/prepare_os.sh "ubuntu@${FIRST_ARG}":/home/ubuntu
ssh -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" "ubuntu@${FIRST_ARG}" '/home/ubuntu/prepare_os.sh && exit'
# if [ $? -ne 0 ]; then
#   exit 1
# fi
# scp -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" /home/maierj/work/autocalib_test_package/initial_setup.sh "ubuntu@${FIRST_ARG}":/home/ubuntu
# ssh -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" "ubuntu@${FIRST_ARG}" '/home/ubuntu/initial_setup.sh && exit'
