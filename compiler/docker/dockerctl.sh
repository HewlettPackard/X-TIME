#! /usr/bin/env bash

set -e

function usage() {
    echo "Usage: ./dockerctl.sh [OPTION]"
    echo "  -b, --build <TAG> <DOCKERFILE> <PORT>          Builds and tags a docker image"
    echo "  -r, --run <IMAGE> <CONTAINER> <PATH> <PORT>    Starts container"
    echo "  -s, --launch <IMAGE> <CONTAINER> <PATH> <PORT> Gets a shell on container"
    echo "  -rm, --remove <CONTAINER>                      Stops and removes container"
    echo "  -l, --local-build <TAG> <DOCKERFILE> <PORT>    Builds and tags a docker image for local use"
    echo "  -h, --help                                     Print this message"
    exit 0
}

if [ $# -eq 0 ]
then
    usage
    exit
fi

MLFLOW_TRACKING_URI=http://10.93.226.108:10000
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PROXY_URL=http://web-proxy.corp.hpecorp.net:8080/

JUPYTER_PORT=8888
PROJECT_NAME=xtime-hpca23
CODE_TARGET_DIR=/home/jovyan/"${PROJECT_NAME}"

KAGGLE_CRED=/home/"${USER}"/.kaggle
CONTAINER_DIRS=../gbm-datasets

while test $# -gt 0
do
    case "$1" in
        -b|--build)
            docker build -t "$2" \
                   --build-arg=PORTS="$4" \
                   --build-arg=http_proxy="${PROXY_URL}" \
                   --build-arg=https_proxy="${PROXY_URL}" \
                   --build-arg=MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
                   --build-arg=PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION}" \
                   --build-arg=HOST_USER="${USER}" \
                   --build-arg=MODE="" \
                   --build-arg=HUID=$(id -u) \
                   --build-arg=HGID=$(id -g) \
                   -f "$3" .
            shift
            shift
            shift
            ;;
        -v|--build-vim)
            docker build -t "$2" \
                   --build-arg=PORTS="$4" \
                   --build-arg=http_proxy="${PROXY_URL}" \
                   --build-arg=https_proxy="${PROXY_URL}" \
                   --build-arg=MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
                   --build-arg=PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION}" \
                   --build-arg=HOST_USER="${USER}" \
                   --build-arg=MODE="vim" \
                   --build-arg=HUID=$(id -u) \
                   --build-arg=HGID=$(id -g) \
                   -f "$3" .
            shift
            shift
            shift
            ;;
        -l|--local-build)
            docker build -t "$2" \
                   --build-arg=PORTS="$4" \
                   -f "$3" .
            shift
            shift
            shift
            ;;
        -r|--run)
            mkdir -p "${CONTAINER_DIRS}"

            docker run --init --rm -p "$5":"${JUPYTER_PORT}" \
		 --user root \
                 --ipc=host \
		 --privileged \
		 --runtime=nvidia \
                 --name "$3" \
                 -v "$4":/home/"${USER}"/"${PROJECT_NAME}" \
                 -v "${KAGGLE_CRED}":"${KAGGLE_CRED}" \
                 -v "$(pwd)/${CONTAINER_DIRS}":/opt/gbm-datasets \
                 -v /opt/mlflow:/opt/mlflow \
                 -v /data:/data \
		 -e NB_USER="${USER}" \
		 -e NB_UID="$(id -u)" \
		 -e NB_GID="$(id -g)"  \
		 -e CHOWN_HOME=yes \
		 -e CHOWN_EXTRA="/home/${USER}/${PROJECT_NAME}" \
                 "$2"
            shift
            shift
            shift
            shift
            ;;
        -s|--shell)
            mkdir -p "${CONTAINER_DIRS}"

            docker run --init --rm -it -p "$5":"${JUPYTER_PORT}" \
		 --user "${USER}" \
                 --ipc=host \
		 --net=host \
		 --privileged \
		 --runtime=nvidia \
                 --user root \
                 --name "$3" \
                 -v "$4":/home/"${USER}"/"${PROJECT_NAME}" \
                 -v "${KAGGLE_CRED}":"${KAGGLE_CRED}" \
		 -v "${HOME}"/.Xauthority:/home/"${USER}"/.Xauthority:rw \
		 -e DISPLAY="${DISPLAY}" \
                 -v "$(pwd)/${CONTAINER_DIRS}":/opt/gbm-datasets \
		 -e NB_USER="${USER}" \
		 -e NB_USER="${USER}" \
		 -e NB_UID="$(id -u)" \
		 -e NB_GID="$(id -g)"  \
		 -e CHOWN_HOME=yes \
		 -e CHOWN_EXTRA="/home/${USER}/${PROJECT_NAME}" \
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
