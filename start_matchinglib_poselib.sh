#!/bin/bash

EXE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/matchinglib_poselib/build"
cd ${EXE_DIR}

EXE_MATCHING=0
if [ "$1" == "match" ]; then
  EXE_MATCHING=1
elif [ "$1" == "pose" ]; then
  EXE_MATCHING=2
elif [ "$1" == "test" ]; then
  EXE_MATCHING=3
else
  echo "Provide 'match', 'pose', or 'test' as first argument to switch between executables for feature matching only, feature matching and pose estimation, or testing of pose estimation algorithms utilizing data from SemiRealSequence"
  exit 1
fi
shift 1

if [ "${EXE_MATCHING}" -eq 1 ]; then
  ./matchinglib-test $@
elif [ "${EXE_MATCHING}" -eq 2 ]; then
  ./poselib-test $@
elif [ "${EXE_MATCHING}" -eq 3 ]; then
  ./noMatch_poselib-test $@
fi
