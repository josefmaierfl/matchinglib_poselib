#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Provide 'match', 'pose', or 'test' as first argument to switch between executables for feature matching only, feature matching and pose estimation, or testing of pose estimation algorithms utilizing data from SemiRealSequence"
  exit 1
fi

SKIP_NEXT=0
IMG_PATH_GIVEN=0
SEMIRS=0
TEST=0
if [ "$1" == "test" ]; then
  TEST=1
fi

OPTS=()
for (( i = 1; i <= "$#"; i++ )); do
  if [ "${SKIP_NEXT}" -eq 1 ]; then
    SKIP_NEXT=0
  elif [ "${!i}" == "--img_path" ]; then
    i2="$((${i} + 1))"
    if [ $# -lt "${i2}" ]; then
      echo "Argument 'image path' after --img_path required"
      exit 1
    fi
    IMG_PATH="${!i2}"
    if [ ! -d ${IMG_PATH} ]; then
      echo "Folder ${IMG_PATH} does not exist"
      exit 1
    fi
    OPTS+=("--img_path" "/app/images")
    IMG_PATH_GIVEN=1
    SKIP_NEXT=1
  elif [ "${!i}" == "--sequ_path" ]; then
    i2="$((${i} + 1))"
    if [ $# -lt "${i2}" ]; then
      echo "Argument 'path to semi-real sequence 3D data' after --sequ_path required"
      exit 1
    fi
    SEQU_PATH="${!i2}"
    if [ ! -d ${SEQU_PATH} ]; then
      echo "Folder ${SEQU_PATH} does not exist"
      exit 1
    fi
    OPTS+=("--sequ_path" "/app/sequence")
    SEMIRS="$((${SEMIRS} + 1))"
    SKIP_NEXT=1
  elif [ "${!i}" == "--output_path" ]; then
    i2="$((${i} + 1))"
    if [ $# -lt "${i2}" ]; then
      echo "Argument 'result path' after --output_path required"
      exit 1
    fi
    OUT_PATH="${!i2}"
    if [ ! -d ${OUT_PATH} ]; then
      echo "Folder ${OUT_PATH} does not exist"
      exit 1
    fi
    OPTS+=("--output_path" "/app/results")
    SEMIRS="$((${SEMIRS} + 1))"
    SKIP_NEXT=1
  else
    OPTS+=("${!i}")
  fi
done

if [ "${TEST}" -eq 1 ] && [ "${SEMIRS}" -gt 0 ] && [ "${SEMIRS}" -ne 2 ]; then
  echo "Arguments --sequ_path and --output_path must be provided"
  exit 1
fi

xhost +local:root
#docker run -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY poselib:1.0 /bin/bash
if [ "${SEMIRS}" -gt 0 ] && [ "${IMG_PATH_GIVEN}" -eq 1 ]; then
  docker run -v ${OUT_PATH}:/app/results -v ${IMG_PATH}:/app/images:ro -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY poselib:1.0 /app/start_matchinglib_poselib.sh ${OPTS[@]}
elif [ "${IMG_PATH_GIVEN}" -eq 1 ]; then
  docker run -v ${IMG_PATH}:/app/images:ro -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY poselib:1.0 /app/start_matchinglib_poselib.sh ${OPTS[@]}
elif [ "${SEMIRS}" -eq 2 ]; then
  docker run -v ${SEQU_PATH}:/app/sequence:ro -v ${OUT_PATH}:/app/results -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY poselib:1.0 /app/start_matchinglib_poselib.sh ${OPTS[@]}
else
  docker run -it -v /tmp/.X11-unix/:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY poselib:1.0 /app/start_matchinglib_poselib.sh ${OPTS[@]}
fi
xhost -local:root
