#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [-n <string>] [-y] <path>

Upload videos to Hugging Face dataset with optional cleanup

Options:
  -n <string>  Hugging Face dataset name (format: username/dataset-name)
  -y           Automatically remove local dataset cache from ~/.cache/huggingface/lerobot/<ds-name>
  <path>       Relative or absolute path to videos folder (required)

Examples:
  $0 -n johndoe/my-videos -y ./videos
  $0 /data/videos
EOF
    exit 1
}

# Initialize variables
DATASET_NAME=""
AUTO_CONFIRM=false
# Parse arguments
while getopts ":n:y" opt; do
    case "${opt}" in
        n)
            DATASET_NAME="${OPTARG}"
            # Validate dataset name format
            if [[ ! "$DATASET_NAME" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$ ]]; then
                echo "Error: Dataset name must be in format 'username/dataset-name'" >&2
                usage
            fi
            ;;
        y)
            AUTO_CONFIRM=true
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

# Verify required path argument
if [ $# -eq 0 ]; then
    echo "Error: Missing required path argument" >&2
    usage
fi
VIDEO_PATH="$1"


confirm() {
    if "$AUTO_CONFIRM"; then
        echo "Auto-confirmed: $1"
        return 0
    fi

    local message="${1:-Are you sure?}"
    while true; do
        read -p "$message [y/N] " yn
        case "$yn" in
            [Yy]*) return 0 ;;  # Return success (0) for Yes
            [Nn]*) return 1 ;;  # Return failure (1) for No
            *) echo "Please answer yes or no." ;;
        esac
    done
}



#------------------------------------------------------------------------------------------------------


# set vars that are needed for running dockers containers on host machine from inside docker container

# Get absolute path to this script (resolves symlinks)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SLAM_SET_PATH="$SCRIPT_DIR/src/universal_manipulation_interface/slam"
# echo $SLAM_SET_PATH


# if ! VIDEO_FULLPATH=$(realpath "$1" 2>/dev/null); then



VIDEO_FULLPATH="$(realpath $1)"
DC_RUN="docker run --rm -it -v  $VIDEO_FULLPATH:$VIDEO_FULLPATH \
    -e SLAM_SET_PATH="$SLAM_SET_PATH" \
    -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock \
    umi bash -c"
if ! [ -d "$VIDEO_FULLPATH" ]; then
   echo "$VIDEO_FULLPATH: No such directory"
else
  echo "extracting trajectories from: $VIDEO_FULLPATH"

  # docker run --rm -it -v  $VIDEO_FULLPATH:$VIDEO_FULLPATH \
  #   -e VIDEO_FULLPATH="$VIDEO_FULLPATH" -e SLAM_SET_PATH="$SLAM_SET_PATH" \
  #   -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock \
  #   umi
  $DC_RUN "conda run -n umi python /workspace/UMI/src/universal_manipulation_interface/run_slam_pipeline.py $VIDEO_FULLPATH"


  if [ "$DATASET_NAME" ]; then
    echo "in"

    # remove cached dataset if present
    CACHE_PATH="$HOME/.cache/huggingface/lerobot/$DATASET_NAME"
    if [ -d "$CACHE_PATH" ]; then

      if  confirm "Delete all files at $CACHE_PATH ?"; then
        echo "Removing existing directory: $CACHE_PATH"
        rm -rf "$CACHE_PATH"
      fi
    fi

    # reformat data and upload ds
    $DC_RUN "uv run pipline_scripts/07_generate_el_dataset.py --repo_id $DATASET_NAME $VIDEO_FULLPATH"
    
  fi
  



fi
