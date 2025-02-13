#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${CURRENT_DIR}/source.sh

docker run --gpus all -ti $PARAMS $VOLUMES $NAME $@
