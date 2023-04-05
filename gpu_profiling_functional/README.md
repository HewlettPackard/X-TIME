# XTIME Environment Setup

## Building the Environment

Go to the `docker` directory and build the base image:

```bash
cd docker
./dockerctl.sh -b cuda-jupyter-xtime:latest Dockerfile.base 8888
```

Then, build the `xtime` image:

```bash
./dockerctl.sh -b xtime:latest Dockerfile 8888
```

Finally, get the JupyterLab login token with:

```bash
./dockerctl.sh -r xtime:latest xtime-container "$(pwd)/.." 8888
```

After that you can open any of the notebooks at the `notebooks`
directory from the Jupyter Lab interface.

## MLflow server
An MLflow server should be setup, using:
- SQlite backend store: `/opt/mlflow/mlruns.db`.
- Filesystem artifact store: `/opt/mlflow/mlruns`.

### Start MLflow server
```shell
# Run in a separate screen session
screen -S mlflow_server
# Go to the root directory and activate python virtual environment with mlflow package
cd /opt/mlflow/
source ./.mlflow/bin/activate
# Export several environment variables
export http_proxy=
export https_proxy=
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Run MLflow server on port 10000 and bind it to all network interfaces so that it's available from remote machines
mlflow server --backend-store-uri sqlite:////opt/mlflow/mlruns.db --default-artifact-root=file:///opt/mlflow/mlruns --host=0.0.0.0 --port=10000
```

### Start MLflow Web UI
```shell
# Run in a separate screen session
screen -S mlflow_ui
# Go to the root directory and activate python virtual environment with mlflow package
cd /opt/mlflow/
source ./.mlflow/bin/activate
# Export several environment variables
export http_proxy=
export https_proxy=
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Run MLflow WebUI on port 10001 and bind it to all network interfaces so that it's available from remote machines
mlflow ui --backend-store-uri sqlite:////opt/mlflow/mlruns.db --default-artifact-root=file:///opt/mlflow/mlruns --host=0.0.0.0 --port=10001
```

### Mount MLflow artifact store on other machines
Experiment metadata will be available via MLflow API. But in order to access artifacts, the MLflow artifact store must
be mounted on each development machine under the same exact path. This is what worked on `xtime-1` and `xtime-3`:
```shell
sudo sshfs ${USER}@xtime-2:/opt/mlflow/datasets /opt/mlflow/datasets -o allow_other -o ro -o IdentityFile=${HOME}/.ssh/id_rsa
sudo sshfs ${USER}@xtime-2:/opt/mlflow/mlruns /opt/mlflow/mlruns -o allow_other -o ro -o IdentityFile=${HOME}/.ssh/id_rsa
```
