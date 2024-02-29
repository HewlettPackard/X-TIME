#! /usr/bin/env bash

set -e

function usage() {
    echo "Usage: ./dockerctl.sh [OPTION]"
    echo "  -b, --build <TAG> <DOCKERFILE> <PORT> <PROXY>  Builds and tags a docker image"
    echo "  -r, --run <IMAGE> <CONTAINER> <PATH> <PORT>    Starts container"
    echo "  -s, --launch <IMAGE> <CONTAINER> <PATH> <PORT> Gets a shell on container"
    echo "  -rm, --remove <CONTAINER>                      Stops and removes container"
    echo "  -h, --help                                     Print this message"
    exit 0
}

if [ $# -eq 0 ]
then
    usage
    exit
fi

JUPYTER_PORT=8888

while test $# -gt 0
do
    case "$1" in
        -b|--build)
            docker build -t "$2" \
                   --build-arg=PORTS="$4" \
                   --build-arg=http_proxy="$5" \
                   --build-arg=https_proxy="$5" \
                   -f "$3" .
            shift
            shift
            shift
            ;;
        -r|--run)
            docker run --rm -p "$5":"${JUPYTER_PORT}" \
		 --user root \
                 --ipc=host \
		 --privileged \
                 --name "$3" \
                 -v "$4":/home/"${USER}"/xtime-sst \
		 -e NB_USER="${USER}" \
		 -e NB_UID="$(id -u)" \
		 -e NB_GID="$(id -g)"  \
		 -e CHOWN_HOME=yes \
		 -e CHOWN_EXTRA="/home/${USER}/xtime-sst" \
                 "$2"
            shift
            shift
            shift
            shift
            ;;
        -s|--shell)
            docker run --rm -it -p "$5":"${JUPYTER_PORT}" \
		 --user root \
         --ipc=host \
		 --privileged \
         --name "$3" \
         -v "$4":/home/"${USER}"/xtime-sst \
		 -e NB_USER="${USER}" \
		 -e NB_UID="$(id -u)" \
		 -e NB_GID="$(id -g)"  \
		 -e CHOWN_HOME=yes \
		 -e CHOWN_EXTRA="/home/${USER}/xtime-sst" \
		 -e CHOWN_EXTRA_OPTS='-R' \
                 "$2" \
                 /bin/bash
            shift
            shift
            shift
            shift
            ;;
        -rm|--remove)
            docker container rm -f "$2"
            shift
            ;;
        --help|-h|*)
            usage
            ;;
    esac
    shift
done

exit 0
