#!/bin/bash
if [ $# -ne 0 ]; then
    ARG_NAME="$1"
    if [ "${ARG_NAME}" == "shutdown" ]; then
        (sleep 2; shutdown -h now) &
    elif [ "${ARG_NAME}" == "exit" ]; then
        (sleep 2; exit) &
    fi
fi
