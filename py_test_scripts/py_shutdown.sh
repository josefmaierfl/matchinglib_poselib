#!/bin/bash
if [ $# -ne 0 ]; then
    ARG_NAME="$1"
    if [ ${ARG_NAME} -eq "shutdown" ]; then
        (sleep 2; shutdown -h now) &
    elif [ ${ARG_NAME} -eq "exit" ]; then
        (sleep 2; exit) &
    fi
fi
