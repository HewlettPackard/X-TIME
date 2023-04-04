###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import numpy as np
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tests")))
from utils import *

arg = readArgPost()

dirPath = "./log/{}/{}_nTree{:d}/{}/".format(arg['dataset'], arg['model'], arg['numTreePerClass'], arg['simulationID'])

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "log/{}/{}_nTree{:d}".format(arg['dataset'], arg['model'], arg['numTreePerClass']))))
componentConfig = __import__( arg['simulationID'], globals(), locals(), [], 0)

numSample = componentConfig.arg['numSample']

noc = componentConfig.arg['noc']
numPortControl = componentConfig.noc[noc]['numPortControl']
numPort = componentConfig.noc[noc]['numPort']
numLevel = componentConfig.noc[noc]['numLevel']
numTotalCore = numPortControl*pow(numPort, numLevel)

powerLabel = ['DL', 'ML', 'SL', 'DAC', 'SA', 'PC', 'REG', 'DRIVER', 'MMR', 'MEMORY', 'ADDER', 'DEMUX', 'ACCUM']
powerRaw = np.zeros((numTotalCore, len(powerLabel)))

fileName = os.scandir(dirPath)

for file in fileName:
    if re.match(r'Output_\d+.csv', file.name):
        f = open(dirPath+file.name)
        f.readline()
        if (f.readline().strip()):
            temp = np.loadtxt(dirPath+file.name, delimiter=", ", skiprows = 1, \
                dtype = {'names' : ('ComponentName', 'StatisticName', 'StatisticSubId', 'StatisticType', 'SimTime', 'Rank', 'Sum.f64', 'SumSQ.f64', 'Count.u64', 'Min.f64', 'Max.f64'), \
                        'formats' : ('U32', 'U32', 'U32', 'U32', 'i8', 'i8', 'f8', 'f8', 'i8', 'f8', 'f8')})
            for j in range(np.shape(temp)[0]):
                coreID = int(re.findall(r'\d+', temp[j][0])[0])
                powerName = re.findall(r'power(\w+)', temp[j][1])[0]
                powerID = powerLabel.index(powerName)
                powerRaw[coreID, powerID] += temp[j][6]
        f.close()
powerAvg = np.sum(powerRaw, axis=0)/numSample
powerFile = dirPath+"power.csv"
with open(powerFile, 'w', newline='') as f:
    writer = csv.writer(f)
    for i, name in enumerate(powerLabel):
        writer.writerow([name, powerAvg[i]])
    writer.writerow(['Total', np.sum(powerAvg)])
f.close()
