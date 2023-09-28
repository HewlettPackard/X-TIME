#!.venv/bin/python3

from xgboost import XGBClassifier
from math import floor, ceil

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, load_wine
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from xtimec import XTimeModel
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import bisect
import pdb

def print_histo(data, suffix="out"):
    plt.hist(data, bins=15)
    plt.savefig(f"histo_{suffix}.png")
    plt.clf()

def quantize_leaves(leaves, multiplier=2**23):
    return (leaves * multiplier).round().astype(int)

def quantize_thresholds(X, X_test, T, max_state, verbose=True):
    T_quant = np.copy(T)

    #min_ts_val = min(X.flatten())
    #max_ts_val = max(X.flatten())
    #ts_val_span = max_ts_val - min_ts_val

    X = np.array(X)
    X_test = np.array(X_test)

    min_max_per_column = [(min(c), max(c)) for c in X.transpose()]
    sorted_values_per_column = [sorted(c) for c in X.transpose()]
    spans_per_column = [p[1] - p[0] for p in min_max_per_column]

    print("min_max_per_column")
    print(min_max_per_column)
    print()

    print("spans_per_column")
    print(spans_per_column)
    print()

    num_T_rows = T.shape[0]
    num_X_rows = X.shape[0]
    num_X_test_rows = X_test.shape[0]
    num_features = int(T.shape[1] / 2)

    # We actually map values to the [0, max_state - 1] range
    # to avoid mismatches caused by X_test values being rounded
    # to max_state increasing the number of mismatches.
    # As a result, the effective number of bits being used to
    # encode values is log2(max_state - 1).

    for i in range(num_T_rows):
        for j in range(num_features * 2):
            #T_quant[i][j] = (T_quant[i][j] - min_max_per_column[int(j / 2)][0]) * (max_state - 1) / spans_per_column[int(j / 2)]
            if (not np.isnan(T_quant[i][j])):
                T_quant[i][j] = (bisect.bisect_right(sorted_values_per_column[int(j / 2)], T_quant[i][j])) * (max_state - 1) / num_X_rows

    np.nan_to_num(T_quant.transpose()[0::2], copy=False, nan=0.0)
    np.nan_to_num(T_quant.transpose()[1::2], copy=False, nan=max_state)

    if verbose:
        print(f"[quantize_thresholds]: (min_ts_val, max_ts_val, num_rows, num_features) = ({min_ts_val}, {max_ts_val}, {num_rows}, {num_features})")
        print()
        print("[quantize_thresholds]: T sample")
        print(T[0])
        print(T[1])
        print(T[2])
        print(T[3])
        print()
        print("[quantize_thresholds]: T_quant sample")
        print(T_quant[0])
        print(T_quant[1])
        print(T_quant[2])
        print(T_quant[3])
        print()

    simpleRound = True
    if simpleRound:
        T_quant = T_quant.round().astype(int)
    else:
        for i in range(num_T_rows):
            for j in range(num_features * 2):
                if j % 2 == 0:
                    T_quant[i][j] = int(round(T_quant[i][j]))
                else:
                    T_quant[i][j] = int(ceil(T_quant[i][j]))

    if verbose:
        print("[quantize_thresholds]: T_quant.round() sample")
        print(T_quant[0])
        print(T_quant[1])
        print(T_quant[2])
        print(T_quant[3])

    #pdb.set_trace()

    for i in range(num_X_rows):
        for j in range(num_features):
            #X[i][j] = min(max_state - 1, (X[i][j] - min_max_per_column[j][0]) * (max_state - 1) / spans_per_column[j])
            X[i][j] = bisect.bisect_right(sorted_values_per_column[j], X[i][j]) * (max_state - 1) / num_X_rows

    for i in range(num_X_test_rows):
        for j in range(num_features):
            #X_test[i][j] = min(max_state - 1, (X_test[i][j] - min_max_per_column[j][0]) * (max_state - 1) / spans_per_column[j])
            X_test[i][j] = bisect.bisect_right(sorted_values_per_column[j], X_test[i][j]) * (max_state - 1) / num_X_rows

    X_quant = [[round(el) for el in l] for l in X]
    X_test_quant = [[round(el) for el in l] for l in X_test]

    #X_quant_std_devs = [np.std(c, ddof=1) for c in X_quant.transpose()]
    #X_test_quant_std_devs = [np.std(c, ddof=1) for c in X_test_quant.transpose()]
    X_quant_std_devs = []
    X_test_quant_std_devs = []

    for j in range(num_features):
        print_histo(X[:][j], f"X_column_{j}")
        print_histo(X_quant[:][j], f"X__quant_column_{j}")
        X_quant_std_devs.append(np.std(X[:][j], ddof=1))
        X_test_quant_std_devs.append(np.std(X[:][j], ddof=1))

    print(X_quant_std_devs)
    print(X_test_quant_std_devs)

    return (T_quant, X, X_test)
    

def test00(max_state = 15):
    # create a random dataset
    #X, y = make_classification(n_samples=1000, n_informative=100, n_redundant=0, n_classes=2, n_features=120, random_state=0)
    X, y = make_classification(n_samples=1000, n_informative=3, n_redundant=0, n_classes=2, n_features=3, random_state=0)

    # train a model
    #model = RandomForestClassifier()
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model.fit(X_train, y_train)

    # represent the model encoded for an analog CAM
    # alternatively `from_xgboost`, `from_catboost` etc.
    #xmodel = XTimeModel.from_sklearn(model)
    xmodel = XTimeModel.from_xgboost(model.get_booster())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("## Training set")
    print(f"X matrix shape: {(X_train).shape}")
    print()

    print("### First row")
    print(X_train[0])
    print()

    print("### Other facts")
    print(f"(min_value, max_value) = ({min(X_train.flatten())}, {max(X_train.flatten())})")
    print()

    print("## Original model accuracy")
    print(accuracy)
    print()

    print("## CAM thresholds")
    print(f"Threshold matrix shape: {(xmodel.cam).shape}")
    print()

    print("### First row")
    print(xmodel.cam[0])
    print()

    print("### All rows")
    print(xmodel.cam)
    print()

    print("### Quantized rows")
    quantize_thresholds(X_train, xmodel.cam, max_state)
    print()

    print("## Leaves")

    print("### All rows")
    print(xmodel.leaves.flatten())
    print()

    print("### Quantized rows")
    print(quantize_leaves(xmodel.leaves.flatten()))
    print()

    print("### Other facts")
    print(f"(min_value, max_value) = ({min(xmodel.leaves.flatten())}, {max(xmodel.leaves.flatten())})")

def test01(max_state = 15):
    # create a random dataset
    #X, y = make_classification(n_samples=1000, n_informative=3, n_redundant=0, n_classes=2, n_features=3, random_state=0)
    #X, y = make_classification(n_samples=1000, n_informative=3, n_redundant=0, n_classes=2, n_features=3, random_state=0, class_sep=10)
    #X, y = make_classification(n_samples=40000, n_informative=120, n_redundant=0, n_classes=2, n_features=120, random_state=0, class_sep=400)
    X, y = make_classification(n_samples=4000, n_informative=29, n_redundant=0, n_classes=2, n_features=29, random_state=0, class_sep=400)

    # train a model
    #model = RandomForestClassifier()
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=8, n_estimators=2)
    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=8, n_estimators=6)
    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=8, n_estimators=100)
    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=2, n_estimators=256)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model.fit(X_train, y_train)

    # represent the model encoded for an analog CAM
    # alternatively `from_xgboost`, `from_catboost` etc.
    #xmodel = XTimeModel.from_sklearn(model)
    xmodel = XTimeModel.from_xgboost(model.get_booster())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("## Original model accuracy")
    print(accuracy)
    print()

    T_quant, X_quant, X_test_quant = quantize_thresholds(X_train, X_test, xmodel.cam, max_state, verbose=False)
    L_quant = quantize_leaves(xmodel.leaves.flatten())

    tree_ids = xmodel.tree_ids.flatten().astype(int)
    class_ids = xmodel.class_ids.flatten().astype(int)

    np.savetxt("T_quant.csv", T_quant, delimiter=",", fmt='%d')
    np.savetxt("L_quant.csv", L_quant, delimiter=",", fmt='%d')
    np.savetxt("t_ids.csv", tree_ids, delimiter=",", fmt='%d')
    np.savetxt("c_ids.csv", class_ids, delimiter=",", fmt='%d')
    np.savetxt("X_test_quant.csv", X_test_quant, delimiter=",", fmt='%d')
    np.savetxt("y_test.csv", y_test, delimiter=",", fmt='%d')

def test02(ms, nb, tm, ml, nt):
    filePath = "/home/lucas/Repos/Summer-CAMp/quant_training/raw_datasets/Churn_Modelling.csv"
    datasetName = "churnModelling"
    data = pd.read_csv(filePath)

    #print(data.head())
    print(data.info())

    X = data.drop(columns=["Exited"])
    y = data["Exited"]

    X = X.iloc[:,3:13]

    lableencoder_X_1 = LabelEncoder()
    X.iloc[:,1] = lableencoder_X_1.fit_transform(X.iloc[:,1])
    lableencoder_X_2 = LabelEncoder()
    X.iloc[:,2] = lableencoder_X_2.fit_transform(X.iloc[:,2])

    ct = ColumnTransformer(
        [("one_hot_encoder", OneHotEncoder(), [1])],
        remainder="passthrough"
    )

    X = ct.fit_transform(X)

    print(X)
    print(X.shape)

    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_leaves=4, n_estimators=512, max_bin=256)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_leaves=ml, n_estimators=nt, max_bin=nb, tree_method=tm)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model.fit(X_train, y_train)

    # represent the model encoded for an analog CAM
    # alternatively `from_xgboost`, `from_catboost` etc.
    #xmodel = XTimeModel.from_sklearn(model)
    xmodel = XTimeModel.from_xgboost(model.get_booster())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("## Original model accuracy")
    print(accuracy)
    print()

    T_quant, X_quant, X_test_quant = quantize_thresholds(X_train, X_test, xmodel.cam, ms, verbose=False)
    L_quant = quantize_leaves(xmodel.leaves.flatten())

    tree_ids = xmodel.tree_ids.flatten().astype(int)
    class_ids = xmodel.class_ids.flatten().astype(int)

    np.savetxt(f"T_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", T_quant, delimiter=",", fmt='%d')
    np.savetxt(f"L_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", L_quant, delimiter=",", fmt='%d')
    np.savetxt(f"t_ids_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", tree_ids, delimiter=",", fmt='%d')
    np.savetxt(f"c_ids_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", class_ids, delimiter=",", fmt='%d')
    np.savetxt(f"X_test_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", X_test_quant, delimiter=",", fmt='%d')
    np.savetxt(f"y_test_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", y_test, delimiter=",", fmt='%d')
    np.savetxt(f"original_accuracy_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", [accuracy], delimiter=",", fmt='%f')

def test03(ms, nb, tm, ml, nt):
    filePath = "/home/lucas/Repos/Summer-CAMp/quant_training/raw_datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    datasetName = "telcoChurn"
    data = pd.read_csv(filePath)

    #print(data.head())
    print(data.info())

    X = data.drop(columns=["Churn", "customerID"])
    y = data["Churn"]

    #categorical_columns = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    categorical_columns = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges"]

    print(X.dtypes)

    for k in categorical_columns:
        labelEncoder = LabelEncoder()
        X[k] = labelEncoder.fit_transform(X[k])

    print(X.dtypes)

    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(y)

    #print(X.to_string())
    print(X)
    print(X.iloc[0])

    hotOneEncode = False
    if hotOneEncode:
        ct = ColumnTransformer(
            [("one_hot_encoder", OneHotEncoder(), categorical_columns)],
            remainder="passthrough"
        )

        X = ct.fit_transform(X)

    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_leaves=4, n_estimators=256, max_bin=2, tree_method="approx")
    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_leaves=4, n_estimators=256)
    #model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_leaves=4, n_estimators=256, max_bin=256, tree_method="approx")

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_leaves=ml, n_estimators=nt, max_bin=nb, tree_method=tm)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model.fit(X_train, y_train)

    # represent the model encoded for an analog CAM
    # alternatively `from_xgboost`, `from_catboost` etc.
    #xmodel = XTimeModel.from_sklearn(model)
    xmodel = XTimeModel.from_xgboost(model.get_booster())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("## Original model accuracy")
    print(accuracy)
    print()

    T_quant, X_quant, X_test_quant = quantize_thresholds(X_train, X_test, xmodel.cam, ms, verbose=False)
    L_quant = quantize_leaves(xmodel.leaves.flatten())

    tree_ids = xmodel.tree_ids.flatten().astype(int)
    class_ids = xmodel.class_ids.flatten().astype(int)

    np.savetxt(f"T_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", T_quant, delimiter=",", fmt='%d')
    np.savetxt(f"L_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", L_quant, delimiter=",", fmt='%d')
    np.savetxt(f"t_ids_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", tree_ids, delimiter=",", fmt='%d')
    np.savetxt(f"c_ids_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", class_ids, delimiter=",", fmt='%d')
    np.savetxt(f"X_test_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", X_test_quant, delimiter=",", fmt='%d')
    np.savetxt(f"y_test_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", y_test, delimiter=",", fmt='%d')
    np.savetxt(f"original_accuracy_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv", [accuracy], delimiter=",", fmt='%f')

def test04():
    max_states = [7, 15, 127, 255]
    num_bins = list(np.logspace(np.log10(2), np.log10(255), 4, endpoint=True).round().astype(int))
    tree_method = ["hist", "approx"]
    max_leaves = ["4"]
    num_trees = list(np.logspace(np.log10(2), np.log10(512), 4, endpoint=True).round().astype(int))

    for ms in max_states:
        for nb in num_bins:
            for tm in tree_method:
                for ml in max_leaves:
                    for nt in num_trees:
                        print(f"(max_states, num_bins, tree_method, max_leaves, num_trees) = ({ms}, {nb}, {tm}, {ml}, {nt})")
                        test03(ms, nb, tm, ml, nt)
                        test02(ms, nb, tm, ml, nt)

#test00(15)
#test01(15)
#test02(15)
#test03(15)
test04()
