#!/bin/bash
set -e
useradd --shell /bin/bash -u $NB_UID -o -c "" -m $NB_USER
export HOME=/home/$NB_USER
chown -R "$NB_UID" /home/
cd $HOME
exec /usr/local/bin/gosu $NB_USER "$@"