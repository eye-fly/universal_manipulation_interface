XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH

# Load .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs -d '\n')
fi


docker run --runtime=nvidia  --rm -it -v  /home/jacek-muszynski/Videos:/home/jacek-muszynski/Videos     -e HF_TOKEN="$HF_TOKEN" \
   -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock     --privileged --network=host -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY=$DISPLAY   --gpus all \
   -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --shm-size=1gb umi