#!/usr/bin/python3

import numpy as np
import re
import os

datasetNames = ["churnModelling", "telcoChurn"]
max_states = [7, 15, 127, 255]
num_bins = list(np.logspace(np.log10(2), np.log10(255), 4, endpoint=True).round().astype(int))
tree_method = ["hist", "approx"]
max_leaves = ["4"]
#num_trees = list(np.logspace(np.log10(2), np.log10(512), 4, endpoint=True).round().astype(int))
num_trees = [128 * x for x in list(range(1,8))]

for datasetName in datasetNames:
    for ms in max_states:
        for nb in num_bins:
            for tm in tree_method:
                for ml in max_leaves:
                    for nt in num_trees:
                        #print(f"(max_states, num_bins, tree_method, max_leaves, num_trees) = ({ms}, {nb}, {tm}, {ml}, {nt})")
                        folderPath = f"{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}"
                        if not os.path.exists(folderPath):
                            os.makedirs(folderPath)
                        fileNames = []
                        fileNames.append(f"T_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv")
                        fileNames.append(f"L_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv")
                        fileNames.append(f"t_ids_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv")
                        fileNames.append(f"c_ids_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv")
                        fileNames.append(f"X_test_quant_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv")
                        fileNames.append(f"y_test_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv")
                        fileNames.append(f"original_accuracy_{datasetName}_{ms}_{nb}_{tm}_{ml}_{nt}.csv")

                        for fn in fileNames:
                            cleanFn = re.sub(r'(_[^_]*){6}[.]csv', '.csv', fn)
                            os.system(f"cp {fn} {folderPath}/{cleanFn}")
