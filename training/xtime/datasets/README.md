# Datasets

We selected datasets for benchmarks based upon papers for the area of applying machine learning and deep learning 
methods for tabular data. These datasets are hosted on different platforms, and depending on where these datasets are
coming from, users might need to download them manually. To get a list of datasets, run the following command:

```shell
python -m xtime.main dataset list
```

Each dataset has a name and version (`<dataset_name>[:<version>]`). Version is defined in the context of this project
and can represent various preprocessed versions of the original datasets. Two common identifies for versions are:
- `default`: preprocessed dataset that might contain categorical features.
- `numerical`: dataset where features have been converted to numerical values.

## Downloading datasets
Datasets are automatically downloaded when users call respective APIs (e.g., `build_dataset`). It's very likely that
all datasets are cached, and subsequent calls do not trigger the download process. If user environment is behind a 
proxy firewall, users might need to configure proxy servers. To configure proxy servers, export two environment 
variables `HTTP_PROXY` and `HTTPS_PROXY`. 

> OpenML datasets use `minio` library that seems to be not using proxy servers. The `xtime` patches this library on
> the fly to make sure that proxy servers are used.

> Datasets hosted on Kaggle platform are not downloaded automatically. Users need to download them manually and copy
> to appropriate locations. Respective dataset builders provide this information.
 
 
## List of datasets.
The following is the list of datasets as of 2023.03.29 (the doc strings in source code provide more information 
including references to publications that were the source for us to identify needed preprocessing and hyperparameters 
search spaces).


|          Dataset          | Task                       | Input Shape  | Output        |     Versions |  Source                                                                                               | Download                                                 | 
|---------------------------|----------------------------|--------------|---------------|---------------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
|      churn_modelling      | binary classification      | (10000, 10)  | num_classes=2 | default, numerical   | [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)                             | manual (`~/.cache/kaggle/datasets/shrutime`)             |
|       eye_movements       | multi-class classification | (10936, 26)  | num_classes=3 | default, numerical   | [OpenML](openml.org/d/1044)                                                                           | automatic                                                |
|     forest_cover_type     | multi-class classification | (581012, 54) | num_classes=7 | default, numerical   | [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html) | automatic                                                |
|     gas_concentrations    | multi-class classification | (13910, 129) | num_classes=6 | default, numerical   | [OpenML](openml.org/d/1477)                                                                           | automatic                                                |
| gesture_phase_segmentation| multi-class classification | (9873, 32)   | num_classes=5 | default, numerical   | [OpenML](openml.org/d/4538)                                                                           | automatic                                                |
|    rossmann_store_sales   | regression                 | (610235, 29) | num_outputs=1 | default, numerical   | [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales)                                    | manual (`~/.cache/kaggle/datasets/rossmann_store_sales`) |
|    telco_customer_churn   | binary classification      | (7032, 19)   | num_classes=2 | default, numerical   | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)                              | manual (`~/.cache/kaggle/datasets/blastchar`)            |
|    year_prediction_msd    | regression                 | (515345, 90) | num_outputs=1 | default              | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)                        | automatic                                                |

