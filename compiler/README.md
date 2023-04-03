# X-TIME compiler

Receives as input tree based ML models from sklearn, xgboost and catboost and compile that into threshold mappable to X-TIME architecture. X-TIME accepts thresholds and inputs in ubyte format, thus the thresholds should be converted to ubyte as well. The compiler folder contains three subfolders

### docker

The `docker` folder contains the container files. We strongly suggest to use Docker for running our code in order to avoid path, dependencies and version issues. Follow the instruction in the `docker` folder for building and running the container.

### src

The `src` folder contains the source code for the compiler. The compiler accepts models from xgboost, sklearn random forest and catboost in the format of classifiers or regressors. `src/compiler.py` contains two main funtions:
1. `extract_thresholds` which receives a model as input and returns the threshold maps in X-TIME compatible shape
2. `map_to_ubyte` which receives the threshold maps from the previous functions and the test inputs, and returns a unsigned byte version of them. Note that threshold maps and inputs have to be converted together in order to assure correct binning.

### notebooks

The `notebooks` folder contains a jupyter notebook with some example of X-TIME compiler usage, starting from skratch, train and compile a model, and loading a pre-trained model before compiling it.

