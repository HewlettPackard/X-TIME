FROM cuda-jupyter-xtime

ARG HOST_USER
ARG HUID=1000
ARG HGID=1000
ARG MODE

ARG MLFLOW_TRACKING_URI
ARG PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION

RUN if [ "$MODE" = "vim" ] ; then \
        pip install \
        jupyterlab-vim \
        mlflow \
        ray[tune] \
        onnxmltools\
        onnxruntime-gpu \
        catboost \
        kaggle \
        tqdm ; \
    else \
        pip install \
        mlflow \
        ray[tune] \
        onnxmltools\
        onnxruntime-gpu \
        catboost \
        kaggle \
        tqdm ; \
    fi

RUN if [ "$MODE" = "vim" ] ; then jupyter lab build; fi

USER root

# From: https://github.com/NVIDIA/gbm-bench/blob/master/Dockerfile

RUN mamba install --quiet --yes \
    -c conda-forge -c rapidsai -c nvidia -c defaults \
    dask \
    cudf \
    dask-cuda \
    rmm \
    librmm \
    rapids-xgboost \
    cuml \
    pycuda \
    lightgbm \
    r-svglite \
    r-extrafont \
    r-patchwork \
    r-ggforce && \
    mamba clean --all -f -y && \
    npm cache clean --force && \
    rm -rf "/home/${NB_USER}/.cache/yarn" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
RUN pip install \
    conifer
RUN groupadd -g $HGID -o $HOST_USER
RUN useradd -l -m -u $HUID -g $HGID -o -s /bin/bash $HOST_USER

ENV MLFLOW_TRACKING_URI $MLFLOW_TRACKING_URI
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION $PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION

USER $HOST_USER

ENV KAGGLE_PROXY $https_proxy

ARG PORTS
EXPOSE $PORTS
