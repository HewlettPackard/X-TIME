# XTIME Compiler dcoker environment

## Building the Environment

Go to the `docker` directory and build the base image:

```bash
cd docker
./dockerctl.sh -b cuda-jupyter-hpca23:latest Dockerfile.base <PORT>
```
Where `PORT` is the port you wish to use to access the running container, such as 8888.
Then, build the `xtime-compiler` image:

```bash
./dockerctl.sh -b xtime-compiler:latest Dockerfile <PORT>
```
Where `PORT` is the port you wish to use to access the running container, such as 8888.
Finally, get the JupyterLab login token with:

```bash
./dockerctl.sh -r xtime-compiler:latest <CONTAINER> <PATH> <PORT>
```
Where `CONTAINER` is a name given to the launched container and `PATH` is an absolute path, such as ""$(pwd)/.." in this case, pointing to a directory you wish to mount on the container, and `PORT` is the port you wish to use to access the running container, such as 8888.