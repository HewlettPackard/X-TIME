# xtime-sst simulation

## Introduction

The `xtime` Structural Simulation Toolkit (SST) simulation module is a functional and cycle-approximate simulator of an in-memory engine for accelerating tree-based machine learning (ML) models with content-addressable memory (CAM). 
It serves to explore not only the architectural design choices but the component-wise functional implementation choices. There are several compiling options to map the tree-based machine learning model to the CAMs and run the system available. In addition to the functional verification of the simulated system, it evaluates the performance (accruacy, latency, throughput, energy efficiency, area, utilization) of the system. The system performance depends on both the hardware itself (e.g. architecture) and the compiler, which brings the hardware and software (e.g. tree-based ML model) together. 

## Table of Contents
1. [Design](#Design)
2. [Building and Installation](#Building-and-Installation)
    1. [Dependencies](#Dependencies)
    2. [Building](#Building)
    3. [Install](#Install)
    4. [Verify](#Verify)
    5. [Documentation](#Documentation)
3. [Usage](#Usage)
4. [Example](#Example)

## Design
![schematic](https://github.hpe.com/labs/xtime-sst/blob/main/slides/xtime-sst-schematic.png)

Presentation showing design details
[xtime-sst-slides.pptx](https://github.hpe.com/labs/xtime-sst/blob/main/slides/xtime-sst-slides.pptx)

## Building and Installation
### Dependencies
This repository builds a loadable module in SST. To use SST, you may build the SST simulation engine using docker image or install it on your workspace directly following the installation guide.

[SST Installation guide](http://sst-simulator.org/SSTPages/SSTBuildAndInstall_12dot1dot0_SeriesDetailedBuildInstructions/)

#### Building from docker image
```sh
git clone git@github.hpe.com:labs/xtime-sst.git
cd docker
./dockerctl.sh -b sst:latest Dockerfile <PORT>
```
Where `PORT` is the port you wish to use to access the running container, such as 8888.

#### Building From Source
To build SST Core from source, from the SST Github page:
##### Centos/RHEL 7
```sh
sudo yum install gcc gcc-c++ python3 python3-devel make automake git libtool libtool-ltdl-devel openmpi openmpi-devel zlib-devel
mkdir sst-core && cd sst-core
git clone https://github.com/sstsimulator/sst-core.git sst-core-src
(cd sst-core-src && ./autogen.sh)
mkdir build && cd build
../sst-core-src/configure \
  MPICC=/usr/lib64/openmpi/bin/mpicc \
  MPICXX=/usr/lib64/openmpi/bin/mpic++ \
  --prefix=$PWD/../sst-core-install
make install 
```
 
##### Ubuntu 20.04
```sh
DEBIAN_FRONTEND=noninteractive sudo apt install openmpi-bin openmpi-common libtool libtool-bin autoconf python3 python3-dev automake build-essential git 
mkdir sst-core && cd sst-core
git clone https://github.com/sstsimulator/sst-core.git sst-core-src
(cd sst-core-src && ./autogen.sh)
mkdir build && cd build
../sst-core-src/configure --prefix=$PWD/../sst-core-install
make install 
```

Once you install the SST-Core, verify the SST-Core installation is correct.
```sh
export SST_CORE_HOME=[path to sst-core build location ('--prefix' in sst-core build)]
export PATH=$SST_CORE_HOME/bin:$PATH
which sst
sst --version
sst-info
sst-test-core
```

### Building
If you install SST-Core on your workspace directly, this similarly requires `autogen.sh` and `run_cofnig.sh` to be run.
```sh
git clone git@github.hpe.com:labs/xtime-sst.git
cd xtime-sst
./autogen.sh
./run_config.sh
make 
```

If you use the docker image, then launch a container and get a shell on the container with:
```sh
cd ..
./docker/dockerctl.sh -s sst:latest <CONTAINER> <PATH> <PORT>
```
Where `CONTAINER` is a name given to the launched container and `PATH` is an absolute path, such as "$(pwd)" in this case, pointing to a directory you wish to mount on the container.

Then, in the shell, run `autogen.sh` and `run_cofnig.sh`.
```sh
cd xtime-sst
./autogen.sh
./run_config.sh
make 
```

### Install
Installation is really registering your module into SST's module database. Once you install your module, it informs SST where it is located on the system and then you only have to reference the main SST executable. Whenever you make change to the source/header files (e.g. control.cc/control.h), it should be installed again.
```
make install
```
### Verify
You can verify the module built, registered, and is loadable by SST by running `sst-info`, a small program to parse and dump all registerd module information known to SST.
```sh
# Dump all module info
sst-info

# Dump and verify xtime
sst-info xtime

# Filter a specific component in xtime
sst-info xtime.acam
```
[sst-info doc](http://sst-simulator.org/SSTPages/SSTToolSSTInfo/)

### Documentation
You can find the details of `xtime-sst` in the documentation here [xtime-sst-documentation](https://pages.github.hpe.com/labs/xtime-sst/html/index.html).

You may modify the python/c++ files to run different testbench or change the comments. To update the documentation, you can run doxygen.
```sh
doxygen Doxyfile
```

## Usage
You need to program a Python script to create SST components and connect them together. The SST will execute the Python config file and run the simulation. 
The format of SST command is:
```sh
sst [SST options] <test file>
```
`[SST options]` is the simulator options, which is optional. For example, the number of parallel threads per rank (-n). You can find the details at the website (http://sst-simulator.org/SSTPages/SSTUserHowToRunSST/) or by typing `sst -h`
`<test file>` is the Python file that creates SST compoenents, assigns their parameters, defines the links connecting them, and interconnect them. The parameters for Python config file may follow it.

The example command of this repository is:
```sh
sst ./tests/test_general.py -- --dataset=churn --model=catboost --numTree=202
```
`./src/tests/test_general.py` is a Python config file building the in-memory engine for tree-based model inference. `-- --dataset=churn --model=catboost --numTree=202` is the paramters for Python config file, which means the dataset is 'churn', the ML model is 'catboost', and the number of trees per class is '202'.

To see the parameters of Python config file, you can do:
```sh
sst ./tests/test_general.py -- -h

usage: sstsim.x [-h] [--dataset DATASET] [--task {classification,regression, SHAP}] [--model {catboost,xgboost,rf}] [--numTree NUMTREE] [--numSample NUMSAMPLE]
                [--numBatch NUMBATCH] [--config CONFIG] [--noc {tree,bus}] [--verbose {0,1,2,3,4,5}]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of dataset
  --task {classification,regression, SHAP}
                        Type of task
  --model {catboost,xgboost,rf}
                        Type of model
  --numTree NUMTREE     Number of tree per class
  --numSample NUMSAMPLE
                        Number of test samples
  --numBatch NUMBATCH   Batch size
  --config CONFIG       Name of configuration file
  --noc {tree,bus}      Type of NoC
  --verbose {0,1,2,3,4,5}
                        Output verbosity
```

### Parameters
Below is an explanation of the options you can pass into the current script.

Parameter | Default | Description
--- | --- | ---
--dataset | churn | The dataset name.
--task | classification | The task type. 'classification'/'regression'/'SHAP' available now.
--model | catboost | The model type. 'catboost'/'xgboost'/'rf' available now.
--numTree | 202 | The number of trees per class. 
--numSample | 2000 | The number of test samples.
--numBatch | 1 | The input batch size. 
--config | default | The name of configuration file containing NoC, link, component parameters.
--noc | tree | The NoC type. 'tree'/'bus' available now.
--verbose | 1 | The verbosity level of info statements. (0: Summary, 1: Every samples, 2: Activity of control component, 3: Activity of control/demux/accumulator component, 4: Activity of every component, 5: Print everything on output files)

### Data/Model format
The groud truth of dataset is either a class label in classification task or a predicted value in regression task, which is indepenent to the ML model. It is located in `./data/{dataset}/y_test.npy`. The ground truth file in numpy format(.npy) will be converted to data format(.dat) in `utils.genPacket()`.

The input test samples may depend on the model as it can be trained with input/threshold quantization, which corresponds to the maximum number of discrete bins to bucket continuous features in tree-based ML model (e.g. `max_bin` of xgboost model). Current version assumes the model uses 8bit quantization (256 bins) so that the test input file is `./data/{dataset}/{model}_nTree{numTree}/x_test_8bit.npy`. The data packet in `xtime` simulator, which is conisted of raw input data, sample index, and routing information, is generated and converted to data format(.dat) in `utils.genPacket()`.

The ML model is pre-trained with some parameters (e.g. the maximum number of discrete bins, the maximum number of leaves, etc) and extracted so that each branch is expressed by a list of low and high thresholds of conditions, leaf value, class index, and tree index. The dimension of extracted model (`./data/{dataset}/{model}_nTree{numTree}/model_8bit.npy`) is [(Total number of leaves), (2*Number of features + 3)]. For example, one of branches in the first tree of the second class has a list of conditions(`Feature#0 < 0.5`, `5.5 <= Feature#1 < 10.5`) and a leaf value,`3.1`.
```
# model_8bit.npy format
# Th_Low (F#0), Th_High (F#0),  Th_Low (F#1), Th_High (F#1),  Th_Low (F#2), Th_High (F#2),  Leaf value, Class ID, Tree ID
 [np.nan,       0.5,            5.5,          10.5,           np.nan,       np.nan,         3.1,        1,        0       ]
```
Each branch is mapped to the core according to the compiling options (`mappingCore` and `mappingNoc` in `config` file). Only assumption is that every branch in the core should have same class index. 

### Component configuration format
The configuration file (`config`) is a json format file (.json) that defines NoC layout parameters and parameter dictionaries for each component.

#### Network on Chip (NoC)
The NoC describes the layout of NoC and the link latency between the components.
```
{
    # Tree NoC
    "tree" : {
        "numPortControl" : 16,            # Number of ports of control
        "numPort" : 4,                    # Number of ports of demux/accumulator
        "numLevel" : 4,                   # Number of levels of sub-tree
        "mappingCore" : "numTreePerCore", # Compiling option for core   {"continuous", "numTreePerCore"}
        "mappingNoc" : "numCoreClass",    # Compiling option for NoC    {"continuous", "numCoreBatch", "numCoreClass"}
        "numCoreClass" : 256,             # Minimum number of cores per class (Active when "mappingNoc" is "numCoreClass")
        "numCoreBatch" : 1280,            # Minimum number of cores per batch (Active when "mappingNoc" is "numCoreClass" or "numCoreBatch")
        "numTreePerCore" : 2,             # Number of trees per core  (Active when "mappingCore" is "numTreePerCore")
        "sizeBus" : 32,                   # Size of bus (flit)
    },
    # Bus NoC
    "bus" : {
        "numPortControl" : 4,             # Number of ports of control
        "numPort" : 4,                    # Number of ports of demux/accumulator
        "numLevel" : 5,                   # Number of levels of sub-tree
        "mappingCore" : "numTreePerCore", # Compiling option for core   {"continuous", "numTreePerCore"}
        "mappingNoc" : "numCoreClass",    # Compiling option for NoC    {"continuous", "numCoreBatch", "numCoreClass"}
        "numCoreClass" : 512,             # Minimum number of cores per class (Active when "mappingNoc" is "numCoreClass")
        "numCoreBatch" : 256,             # Minimum number of cores per batch (Active when "mappingNoc" is "numCoreClass" or "numCoreBatch")
        "numTreePerCore" : 1,             # Number of trees per core  (Active when "mappingCore" is "numTreePerCore")
        "sizeBus" : 32,                   # Size of bus (flit)
    },
    "linkLatency" : {
        # If latency is initialized as 0, it will be updated to latency depending on the data size from "utils.configureNoC()". Otherwise, it is fixed as given value in ns scale.
        "Control-Demux" : 0,              # Latency between control and demux
        "Demux-Core" : 0,                 # Latency between demux and core
        "Core-Accumulator" : 0,           # Latency between core and accumulator
        "Accumulator-Control" : 0,        # Latency between accumulator and control
        "Demux-Demux" : 0,                # Latency between demuxs
        "Accumulator-Accumulator" : 0,    # Latency between accumulators

        "Driver-Acam(EN)" : 1,            # Latency between driver and acam (enable signal)
        "Driver-Acam(DL)" : 1,            # Latency between driver and acam (data signal)
        "Acam-Mmr" : 1,                   # Latency between acam and mmr
        "Mmr-Memory" : 1,                 # Latency between mmr and memory
        "Memory-Adder" : 1,               # Latency between memory and adder
        "Acam-Acam" : 1,                  # Latency between acams
    },
}
```

##### Compiling options for core/NoC


#### Components
Here are the brief descriptions of each component.
- The control sends data packets to demux, receives result packets from accumulator, and predict a class label(classification)/a value(regression).
- The demux receives a data packet, and distributes its copies to the ports according to 'mode' and routing info in the data packet.
- The driver receives a data packet from demux, and apply the data and enable signal to acams in the core.
- The acam evaluates a list of conditions (low threshold <= data line value < high threshold) per row when the enable singal vector is arrived, and send a data packet (match vector) to mmr.
- The mmr receives a data packet (match vector) from acam, converts the match vector to the address, and sends a data packet (address) to memory.
- The memory receives a data packet (address) from mmr, and sends a result packet (logit) to adder.
- The adder recieves result packets from memories, accumulates the results, and sends a result packet to accumulator.
- The accumulator receives result packets, accumulates them, which have same sample index and class index, and sends a result packet.

The configuration parameters defined explicitly in `config` file are `latency`, which is the number of cycles to process data, and `freq`, which is the component frequency. Other parameters are automatically initialized from `utils.configureNoc()` based on the arguments (`utils.readArg()`), the NoC configuration (`utils.readComponent()`), and the ML model (`utils.readModel()`).
```
"control" : {
    "latency" : 1,    # Number of cycles to process data
    "freq" : "1GHz"   # Component frequency
}
```
The acam requires additional configuration parameters for acam layout.
```
"acam" : {
    "latency" : 4,    # Number of cycles to process data
    "freq" : "1GHz",  # Component frequency
    "acamQueue" : 1,  # Number of queued acams in a core
    "acamStack" : 1,  # Number of stacked acams in a core
    "acamCol" : 32,   # Number of acam column
    "acamRow" : 256,  # Number of acam row
}
```

### Parallel Execution
SST is built from the ground up around parallel execution. Each component can be divided across different processes/threads and synced on a communication time boundary defined by their links. To execute SST in parallel, prefix the `sst` command with `mpirun`:
```bash
mpirun -n <num_procs> sst <test.py> -- <params>
```

### Parametric Simulation
The tool allows us to run series of simulations with different configuration parameters. The example command is:
```sh
python ./tests/test_auto.py --numRank=50 --numThread=2 --dataset=gesture --task=classification --model=xgboost --numTree=379 --numSample=1975 --numBatch=1 --noc=tree --config=default --hyperparam=hyperparam  --verbose=0 --printFlag=no
```
The example command launches 50 parallel processors with 2 threads each, and simulates the classification task for gesture dataset with 1975 test samples. The pre-trained model is xgboost model with 379 trees per class. The list of parameters we want to test with different values is given in `hyperparam.json`, and the parameters not included in `hyperparam.json` is initialized from `default.json`. The intermediate results of each configuration can be printed if `printFlage` is `yes`. The result will be saved in `./log/{dataset}/{model}_nTree{numTree}/<simulation start time>/`.

The example json file for parametric simulation is:
```
{
    "tree" : {
        "numPortControl" : {
            "type"  : "choice",
            "value" : [1, 4, 16]
        },
        "numPort" : {
            "type"  : "constant",
            "value" : 4
        },
        "numLevel" : {
            "type"  : "range",
            "min"   : 2,
            "max"   : 5,
            "step"  : 1
        },
        "mappingCore" : {
            "type"  : "choice",
            "value" : ["numTreePerCore"]
        }
    }
}
```
The example code will run 12 simulations with different configurations since there are 3 possible values (1, 4, 16) for "numPortControl" and 4 possible values (2, 3, 4, 5) for "numLevel". 

## Example

### Example outputs
```
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=gesture --model=xgboost --numTree=379 --numSample=1975 --numBatch=1 --config=default --noc=tree --task=classification --verbose=0

Reading dataset and model...
Dataset: gesture (#Feature: 32, #Class: 5), Task: classification
Model: xgboost (#Tree: 379)

Reading components...
NoC : tree
Number of control ports: 1
Number of ports, levels: 4, 6
Number of total available cores: 4096

Mapping option for core and NoC: numTreePerCore, numCoreClass
Number of trees per core: 1
Number of cores per batch: 256
Number of cores per class: 512

Configuring NoC...
Number of cores per class: [379 379 379 379 379]
Input configuration (#Batch: [Control, Demux at level 1, Demux at level 2, ..., Demux at level numLevel]):
{0: [[0, 0, 0, 0, 0, 0, 0]]}

Generating input packet...

Building SST modules...
Total time for Python API: 32.4s

Running SST...

Accuracy : 1381/1975 = 0.699
Throughput : 1975/(17895 ns) --> 110.366 M samples/s, Latency : 128 ns
Time : 30.0
Simulation is complete, simulated time: 17.895 us
```

```
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=gesture --model=xgboost --numTree=379 --numSample=1975 --numBatch=1 --config=default --noc=tree --task=classification --verbose=1

...
17858 [0:0] [Control0]:          ID#1970 Truth 1, Predict 1 (1.494) -> 1377/1971 = 0.699
17867 [0:0] [Control0]:          ID#1971 Truth 2, Predict 2 (1.468) -> 1378/1972 = 0.699
17876 [0:0] [Control0]:          ID#1972 Truth 2, Predict 2 (1.338) -> 1379/1973 = 0.699
17885 [0:0] [Control0]:          ID#1973 Truth 0, Predict 0 (3.286) -> 1380/1974 = 0.699
17894 [0:0] [Control0]:          ID#1974 Truth 4, Predict 4 (2.007) -> 1381/1975 = 0.699

Accuracy : 1381/1975 = 0.699
Throughput : 1975/(17895 ns) --> 110.366 M samples/s, Latency : 128 ns
Time : 30.0
Simulation is complete, simulated time: 17.895 us
```

#### Example codes
```sh
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=churn --model=catboost --numTree=202 --numSample=2000 --numBatch=1 --config=default --noc=tree --task=classification --verbose=0
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=eye --model=xgboost --numTree=784 --numSample=2188 --numBatch=1 --config=default --noc=tree --task=classification --verbose=0
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=forest --model=xgboost --numTree=193 --numSample=116203 --numBatch=1 --config=default --noc=tree --task=classification --verbose=0
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=gas --model=rf --numTree=226 --numSample=2782 --numBatch=1 --config=default --noc=tree --task=classification --verbose=0
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=gesture --model=xgboost --numTree=379 --numSample=1975 --numBatch=1 --config=default --noc=tree --task=classification --verbose=0
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=telco --model=xgboost --numTree=159 --numSample=1407 --numBatch=1 --config=default --noc=tree --task=classification --verbose=0
mpirun -n 50 sst --num_threads=2 --timebase=1ns ./tests/test_general.py -- --dataset=rossmann --model=xgboost --numTree=2017 --numSample=100000 --numBatch=1 --config=default --noc=tree --task=regression --verbose=0
```

#### Results of example codes

| Dataset   | Model     | #Features | #Classes  | #Trees per class  | Accuracy<br>(%)   | Throughput<br>(M samples/s)   | Latency<br>(ns)   |
| :---      | :---      | :---      | :---      | :---              | :---              | :---                          | :---              |
| churn     | catboost  | 10        | 2         | 202               | 86.0              | 247.5                         | 85                |
| eye       | xgboost   | 26        | 3         | 784               | 73.1              | 124.2                         | 117               |
| forest    | xgboost   | 54        | 7         | 193               | 95.7              | 66.7                          | 174               |
| gas       | rf        | 129       | 6         | 226               | 98.8              | 30.2                          | 298               |
| gesture   | xgboost   | 32        | 5         | 379               | 69.9              | 110.4                         | 128               |
| telco     | xgboost   | 19        | 2         | 159               | 81.7              | 164.8                         | 99                |
| rossmann  | xgboost   | 29        | 1         | 2017              | 530.5 (RMSE)      | 111.1                         | 120               |
