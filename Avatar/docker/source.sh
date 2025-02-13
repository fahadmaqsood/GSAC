#!/usr/bin/env bash
GSAC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../.."

PORT=8087

PARAMS="--net=host --ipc=host -u $(id -u ):$(id -g)"
NAME="gsac"
# VOLUMES="-v /${GSAC_DIR}:/gsac -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"
VOLUMES="-v /${GSAC_DIR}:/gsac"
