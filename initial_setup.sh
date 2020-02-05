#!/bin/bash

WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/work"
mkdir ${WORK_DIR}
cd ${WORK_DIR}

git clone https://github.com/josefmaierfl/autocalib_test_package.git
cd autocalib_test_package
./build_docker_base.sh
