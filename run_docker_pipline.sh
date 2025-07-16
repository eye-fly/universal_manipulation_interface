#!/bin/bash

# set vars that are needed for running dockers containers on host machine from inside docker container

# Get absolute path to this script (resolves symlinks)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SLAM_SET_PATH="$SCRIPT_DIR/src/universal_manipulation_interface/slam"
echo $SLAM_SET_PATH


VIDEO_FULLPATH="$(realpath ~/Wideo/vid2)"
