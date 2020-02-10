#!/bin/bash

#Connect to server and install dependencies
scp -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" /home/maierj/work/autocalib_test_package/prepare_os.sh ubuntu@ec2-3-121-98-109.eu-central-1.compute.amazonaws.com:/data
ssh -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" ubuntu@ec2-3-121-98-109.eu-central-1.compute.amazonaws.com '/data/prepare_os.sh && exit'
scp -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" /home/maierj/work/autocalib_test_package/initial_setup.sh ubuntu@ec2-3-121-98-109.eu-central-1.compute.amazonaws.com:/data
ssh -i "/home/maierj/work/aws/ait_calib_frankfurt.pem" ubuntu@ec2-3-121-98-109.eu-central-1.compute.amazonaws.com '/data/initial_setup.sh && exit'
