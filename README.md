# X-TIME: An In-memory Engine for Accelerating Machine Learning on Tabular Data with CAMs

## Introduction
X-TIME is a simulator framework for tree based machine learning (ML) accelerator
based on analog content addressable memories (CAM).

At  Hewlett   Packard  Labs   we  recently  developed   analog  CAMs   based  on
memristors[^1]. Compared  to traditional CAM  that can store values,  the analog
CAM stores a  range returning a match  if the analog input is  within the stored
range. We  demonstrated that analog CAM  can map and accelerate  decision trees,
resulting in  tremendous speedup  comparaed to other  accelerators thank  to the
massively parallel threshold look-up operation[^2].

This repository  contains the  code for  our recent  work "X-TIME:  An in-memory
engine for accelerating machine learning on tabular data with CAMs"[^3]

Abstract:  "Structured, or  tabular,  data is  the most  common  format in  data
science.  While deep  learning models  have proven  formidable in  learning from
unstructured data such as images or  speech, they are less accurate than simpler
approaches  when learning  from  tabular data.  In  contrast, modern  tree-based
Machine  Learning (ML)  models  shine in  extracting  relevant information  from
structured data.  An important requirement  in data  science is to  reduce model
inference latency  in cases where, for  example, models are used  in closed loop
with  simulation  to  accelerate  scientific discovery.  However,  the  hardware
accelera- tion community has mostly focused on deep neural networks, and largely
ignored other forms of machine learning.  Previous work has described the use of
an analog  content addressable  memory (CAM)  component for  efficiently mapping
random forest. In this work, we  focus on an overall analog-digital architecture
implementing a novel  increased precision analog CAM and  a programmable network
on chip allowing  the inference of state- of-the-art tree-based  ML models, such
as XGBoost and  CatBoost. Results evaluated in a single  chip at 16nm technology
show  119× lower  latency  at 9740×  higher throughput  compared  with a  state-
of-the-art GPU, with a 19W peak power consumption."

contact: [giacomo.pedretti@hpe.com](giacomo.pedretti@hpe.com)

## Training machine learning models
The [training](./training)  Python-based subproject implements scripts  to train
machine learning  models and  optimize their hyperparameters.  The documentation
provides details on what datasets and  machine learning models are available out
of the box and how to train new models.

## Compiling machine learning models for the X-TIME architecture
The  [compiler](./compiler)   Python-based  subproject  implements   scripts  to
compiler  pre-trained machine  learning  models  (currently supporting  sklearn,
XGBoost and CatBoost) in a format  accepted by the cycle accurate simulator, and
currently support all models trained with the [training](./training) subproject.

## Running functional and cycle-accurate simulation
The  [cycle_accurate](./cycle_accurate)  subproject  implements C++  scripts  to
build functional blocks and Python scripts to create X-TIME architecture and run
the cycle-accurate simulation. In addition to the functional verification of the
simulated system,  it evaluates the performance  (accuracy, latency, throughput,
energy efficiency, area, utilization) of the system.

The [functional  simulator and  profiler](./gpu_profiling_functional) subproject
contains a CUDA kernel that simulates  the execution of tree inference models on
an analog CAM and a python pycuda interface that enables evaluating the accuracy
of a tree model ported to the aCAM by launching the CUDA kernel from python code
and  jupyter  notebooks. Additionally,  this  subproject  contains utilities  to
profile  tree models  using  CUDA and  NVIDIA Rapids,  parsing  the results  and
generating tables and data for comparisons to the aCAM results.

## Contributing to XTIME
Thank you for your interest in contributing to XTIME! If you have found a bug or
want to ask a question or discuss a  new feature (such as adding support for new
datasets        or        ML        models),        please        open        an
[issue](https://github.com/HewlettPackard/X-TIME/issues).   Once  a new  feature
has  been implemented  and tested,  or a  bug has  been fixed,  please submit  a
[pull](https://github.com/HewlettPackard/X-TIME/pulls) request.

- `Python`. Some of the subprojects provide unit tests. Make sure they run. It's
  also a good idea  to provide unit tests along with  new features. We generally
  follow   [PEP-8](https://peps.python.org/pep-0008/)  programming   guidelines.
  Development environments such as PyCharm and  VSCode can help you follow these
  guidelines.               Some              subprojects               (example
  [pyproject.toml](https://github.com/HewlettPackard/X-TIME/blob/main/training/pyproject.toml)
  file)      provide       configurations      for      tools       such      as
  [isort](https://pycqa.github.io/isort/),
  [black](https://black.readthedocs.io/en/stable/)                           and
  [flake8](https://flake8.pycqa.org/en/latest/).  Please make  sure to run these
  tools before submitting a pull request.

In order for us to accept your pull request, you will need to `sign-off` your commit.
This [page](https://wiki.linuxfoundation.org/dco) contains more information about it.
In short, please add the following line at the end of your commit message:
```text
Signed-off-by: First Second <email>
```

## License
XTIME is licensed under [Apache 2.0](https://github.com/HewlettPackard/X-TIME/blob/master/LICENSE) license.


[^1]: Li, C., Graves, C.E., Sheng, X. et al. Analog content-addressable memories with memristors. Nat Commun 11, 1638 (2020). https://doi.org/10.1038/s41467-020-15254-4
[^2]: Pedretti, G., Graves, C.E., Serebryakov, S. et al. Tree-based machine learning performed in-memory with memristive analog CAM. Nat Commun 12, 5806 (2021). https://doi.org/10.1038/s41467-021-25873-0
[^3]: Pedretti, G., Moon, J., Bruel, P. et al. X-TIME: An in-memory engine for accelerating machine learning on tabular data with CAMs. arXiv:2304.01285 (2023). https://arxiv.org/abs/2304.01285
