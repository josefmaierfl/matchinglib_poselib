#!/bin/bash
if [ $# -ne 0 ]
  then
    ARG_NAME="$1"
    if [ ${ARG_NAME} -eq "shutdown" ]
      then
        shutdown -h +2
    elif [ ${ARG_NAME} -eq "exit" ]
      then
        then
          (sleep 2; exit) &
    fi
fi
