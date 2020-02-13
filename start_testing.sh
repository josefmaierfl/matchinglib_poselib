#!/bin/bash

PY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/py_test_scripts"
cd ${PY_DIR}
#echo "works $1"
python main.py "$@"
