FROM ubuntu:20.04

LABEL maintainer="John Moon <jmoon@hpe.com>"
ARG NB_USER="sst"
ARG NB_UID="1000"
ARG NB_GID="100"

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && apt-get upgrade --yes &&\
        apt-get install --yes\
        git\
        wget\
        gcc g++\
        make autoconf\
        libtool libtool-bin\
        mpich libopenmpi-dev openmpi-bin

RUN apt-get install --yes python python2 python3 python-dev

RUN apt-get install --yes python-is-python3

ENV SHELL=/bin/bash \
        NB_USER="${NB_USER}" \
        NB_UID=${NB_UID} \
        NB_GID=${NB_GID}

ENV HOME=/home/${NB_USER}\
        MPIHOME=$HOME/local/packages/OpenMPI-4.0.5

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
        # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
        echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc

# Create NB_USER with name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN useradd -l -m -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" && \
        chown "${NB_USER}:${NB_GID}" "${HOME}" && \
        chmod g+w /etc/passwd 

RUN mkdir -p $HOME/scratch/src &&\
    mkdir -p $HOME/local/packages 

RUN cd $HOME/scratch/src \
        && wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz \
        && tar xfz openmpi-4.0.5.tar.gz\
        && cd openmpi-4.0.5\
        && ./configure --prefix=$MPIHOME\
        && make all install

ENV PATH=$MPIHOME/bin:$PATH\
        MPICC=mpicc\
        MPICXX=mpicxx\
        PMIX_MCA_gds=hash\
        LD_LIBRARY_PATH=$MPIHOME/lib:$LD_LIBRARY_PATH\
        DYLD_LIBRARY_PATH=$MPIHOME/lib:$DYLD_LIBRARY_PATH\
        MANPATH=$MPIHOME/share/man:$DYLD_LIBRARY_PATH\
        SST_CORE_HOME=$HOME/local/sstcore-11.1.0\
        SST_CORE_ROOT=$HOME/scratch/src/sstcore-11.1.0

RUN cd $HOME/scratch/src \
        && wget https://github.com/sstsimulator/sst-core/releases/download/v11.1.0_Final/sstcore-11.1.0.tar.gz\
        && tar xfz sstcore-11.1.0.tar.gz\
        && cd sstcore-11.1.0\
        && ./configure --prefix=$SST_CORE_HOME\
        && make all\
        && make install

ENV PATH=$SST_CORE_HOME/bin:$PATH

RUN apt-get install --yes\
        python3-numpy

RUN apt update

RUN apt-get install --yes gpg

RUN wget -O /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && wget -O /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

RUN apt-get install --yes libboost-dev

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]