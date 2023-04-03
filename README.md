# X-TIME: An in-memory engine for accelerating machine learning on tabular data with CAMs

## Introduction
X-TIME is a simulator framework for tree based machine learning (ML) accelerator based on analog content addressable memories (CAM). 

At Hewlett Packard Labs we recently developed analog CAMs based on memristors[^1]. Compared to traditional CAM that can store values, the analog CAM stores a range returning a match if the analog input is within the stored range. We demonstrated that analog CAM can map and accelerate decision trees, resulting in tremendous speedup comparaed to other accelerators thank to the massively parallel threshold look-up operation[^2].

This repository contains the code for our recent work "X-TIME: An in-memory engine for accelerating machine learning on tabular data with CAMs"[^3]

Abstract: "Structured, or tabular, data is the most common format in data science. While deep learning models have proven formidable in learning from unstructured data such as images or speech, they are less accurate than simpler approaches when learning from tabular data. In contrast, modern tree-based Machine Learning (ML) models shine in extracting relevant information from structured data. An important requirement in data science is to reduce model inference latency in cases where, for example, models are used in closed loop with simulation to accelerate scientific discovery. However, the hardware accelera- tion community has mostly focused on deep neural networks, and largely ignored other forms of machine learning. Previous work has described the use of an analog content addressable memory (CAM) component for efficiently mapping random forest. In this work, we focus on an overall analog-digital architecture implementing a novel increased precision analog CAM and a programmable network on chip allowing the inference of state- of-the-art tree-based ML models, such as XGBoost and CatBoost. Results evaluated in a single chip at 16nm technology show 119× lower latency at 9740× higher throughput compared with a state- of-the-art GPU, with a 19W peak power consumption."

contact: [giacomo.pedretti@hpe.com](giacomo.pedretti@hpe.com)


## Training machine learning models
The [training](./training) Python-based subproject implements scripts to train machine learning models and
optimize their hyperparameters. The documentation provides details on what datasets and machine learning models 
are available out of the box and how to train new models.


## License
XTIME is licensed under [Apache 2.0](https://github.com/HewlettPackard/X-TIME/blob/master/LICENSE) license.


## References
[^1]: Li, C., Graves, C.E., Sheng, X. et al. Analog content-addressable memories with memristors. Nat Commun 11, 1638 (2020). https://doi.org/10.1038/s41467-020-15254-4
[^2]: Pedretti, G., Graves, C.E., Serebryakov, S. et al. Tree-based machine learning performed in-memory with memristive analog CAM. Nat Commun 12, 5806 (2021). https://doi.org/10.1038/s41467-021-25873-0
[^3]: Pedretti, G., Moon, J., Bruel, P. et al. X-TIME: An in-memory engine for accelerating machine learning on tabular data with CAMs
