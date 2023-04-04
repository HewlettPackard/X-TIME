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

"""
    
    @brief Utility module for reading raw data and converting them to the proper format for SST
"""   

import numpy as np
import sys
import argparse
import os
import time
import csv
import json
import re
import subprocess

def readArg():
    """! 
    @brief      Read the arguments.
    @details    Read the arguments from the command prompt. Send the arguments to 'readComponent' and 'readModel' functions.

    @param      dataset:    Name of dataset             {churn, eye, forest, gas, gesture, telco, rossmann}
    @param      task:       Type of task                {classification, regression, SHAP}
    @param      model:      Type of model               {catboost, xgboost, rf}
    @param      numTree:    Number of tree per class    {202}
    @param      numSample:  Number of test samples      {2000}
    @param      numBatch:   Batch size                  {1}
    @param      config:     Name of configuration file  {default}
    @param      noc:        Type of NoC                 {tree, bus}
    @param      verbose:    Output verbosity            {1, 0 ~ 5}

    @return     arg
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="churn", help="Name of dataset")
    ap.add_argument("--task", type=str, default="classification", help="Type of task", choices=["classification", "regression", "SHAP"])
    ap.add_argument("--model", type=str, default="catboost", help="Type of model", choices=["catboost", "xgboost", "rf"])
    ap.add_argument("--numTree", type=int, default=202, help="Number of tree per class")
    ap.add_argument("--numSample", type=int, default=2000, help="Number of test samples")
    ap.add_argument("--numBatch", type=int, default=1, help="Batch size")
    ap.add_argument("--config", type=str, default="default", help="Name of configuration file")
    ap.add_argument("--noc", type=str, default="tree", help="Type of NoC", choices=["tree", "bus", "booster"])
    ap.add_argument("--verbose", type=int, default=1, help="Output verbosity", choices=range(6))
    ap.add_argument("--printFlag", type=str, default="yes", help="Print on prompt", choices=["yes", "no"])
    args = ap.parse_args()

    arg = {}
    arg['dataset'] = args.dataset
    arg['task'] = args.task
    arg['model'] = args.model
    arg['numTreePerClass'] = args.numTree
    arg['numSample'] = args.numSample
    arg['numBatch'] = args.numBatch
    arg['config'] = args.config
    arg['noc'] = args.noc
    arg['verbose'] = args.verbose
    arg['printFlag'] = args.printFlag

    return arg

def readComponent(arg):
    """!
    @brief      Read the component parameters.
    @details    Read the component parameters from the 'config' file. Send the parameters to 'configureNoC' function.\n
    List of components: {control, demux, driver, acam, mmr, memory, adder, accumulator}

    @param[in]  arg:                Arguments(dict) from 'readArg'
    
    @return     componentConfig
    """
    
    try:
        with open(arg['config'] + ".json", 'r') as f:
            componentConfig = json.load(f)
    except Exception as err:
        sys.exit('FATAL: could not import `{}` - {}'.format(arg['config'], err) )
    
    # NoC
    noc = arg['noc']
    mappingCore = componentConfig[noc]['mappingCore']
    mappingNoc = componentConfig[noc]['mappingNoc']
    numCoreClass = componentConfig[noc]['numCoreClass']
    numCoreBatch = componentConfig[noc]['numCoreBatch']
    numTreePerCore = componentConfig[noc]['numTreePerCore']
    numPortControl = componentConfig[noc]['numPortControl']
    numPort = componentConfig[noc]['numPort']
    numLevel = componentConfig[noc]['numLevel']
    numTotalCore = numPortControl*pow(numPort, numLevel)
    
    if arg['printFlag'] == "yes":
        print("\nReading components...")
        print("NoC : {}".format(noc))
        print("Number of control ports: {}".format(numPortControl))
        print("Number of ports, levels: {}, {}".format(numPort, numLevel))
        print("Number of total available cores: {}".format(numTotalCore))
        print("\nMapping option for core and NoC: {}, {}".format(mappingCore, mappingNoc))
        if mappingCore != 'continuous':
            print("Number of trees per core: {}".format(numTreePerCore))      
        if mappingNoc != 'continuous':
            print("Number of cores per batch: {}".format(numCoreBatch))
        if mappingNoc == 'numCoreClass':
            print("Number of cores per class: {}".format(numCoreClass))
    return componentConfig

def readModel(arg):
    """! 
    @brief      Read the model.
    @details    Read the model from '/data/{dataset}/{model}_nTree{numTree}/model_8bit.npy. Send the model to 'configureNoC' function.\n
    model_8bit.npy format (size: 2*#Feature + 3)\n
    Th_Low (Feature#0), Th_High (Feature#0), ..., Th_Low (Feature#N), Th_High (Feature#N), logit, Class ID, Tree ID

    @param[in]  arg:                Arguments(dict) from 'readArg'
    @param      numFeature:         Number of features
    @param      numClass:           Number of class
    @param      numLeafPerClass:    Number of leaves (branches) per class
    @param      numTreePerClass:    Number of tree per class. Check it is same as the argument, numTree. Assume each class has same number of trees.
    
    @return     modelConfig
    """

    
    modelRaw = np.load("./data/{0}/{1}_nTree{2}/model_8bit.npy".format(arg['dataset'], arg['model'], arg['numTreePerClass']))
    # model.npy format
    # Th_Low (Feature#0), Th_High (Feature#0), Th_Low (Feature#1), Th_High (Feature#1), ..., logit, Class ID, Tree ID
    numFeature = int((np.shape(modelRaw)[1]-3)/2)
    numClass = int(max(modelRaw[:, -2])+1)
    numLeafPerClass = np.zeros(numClass)
    numTreePerClass = max(modelRaw[:, -1])+1
    assert (numTreePerClass == arg['numTreePerClass'])
    
    model = []
    for i in range(numClass):
        modelClass = modelRaw[(modelRaw[:, -2] == i),:]
        model.append(modelClass)
        numLeafPerClass[i] = np.shape(modelClass)[0]

    modelConfig = {}
    modelConfig['model'] = model

    modelConfig['info'] = {}
    modelConfig['info']['numFeature'] = numFeature
    modelConfig['info']['numClass'] = numClass
    modelConfig['info']['numLeafPerClass'] = numLeafPerClass
    modelConfig['info']['numTreePerClass'] = numTreePerClass

    if arg['printFlag'] == "yes":
        print("\nReading dataset and model...")
        print("Dataset: {} (#Feature: {:d}, #Class: {}), Task: {}".format(arg['dataset'], numFeature, numClass, arg['task']))
        print("Model: {} (#Tree: {})".format(arg['model'], int(numTreePerClass)))
    return modelConfig



def genPacket(arg, inputConfig):
    """! 
    @brief      Generate input packet.
    @details    Generate input packet by adding sample Id, routing info to raw input data. Write the input packets in data format(.dat).

    @param[in]  arg:                Arguments(dict) from 'readArg'
    @param[in]  inputConfig:        Input configuraiton (dict) from 'configureNoc'
    
    @return     None
    """
    if arg['printFlag'] == "yes":
        print("\nGenerating input packet...")
    numBatch = arg['numBatch']
    x_test = np.load("./data/{0}/{1}_nTree{2}/x_test_8bit.npy".format(arg['dataset'], arg['model'], arg['numTreePerClass']))
    y_test = np.load("./data/{0}/y_test.npy".format(arg['dataset']))
    x_test_packet = []
    if (arg['task'] == "classification" or arg['task'] == "regression"):
        for i in range(np.shape(x_test)[0]):
            b = i%numBatch
            for j, index in enumerate(inputConfig[b]):
                x_packet = list(x_test[i, :])
                x_packet.extend([i])
                x_packet.extend(index)
                x_test_packet.append(x_packet)
        with open("./data/{0}/{1}_nTree{2}/x_test_8bit.dat".format(arg['dataset'], arg['model'], arg['numTreePerClass']), 'w') as file:
            for row in x_test_packet:
                file.write(' '.join([str(item) for item in row]))
                file.write('\n')
        with open("./data/{0}/y_test.dat".format(arg['dataset']), 'w') as file:
            for row in y_test:
                file.write(str(row))
                file.write('\n')

    # TODO: SHAP
    elif (arg['task'] == 'SHAP'):
        for i in range(arg['numSample']):
            b = i%numBatch
            for j in range(arg['numSample']):
                for index in inputConfig[b]:
                    x_packet = list(x_test[i, :])
                    x_packet.extend(x_test[j, :])
                    x_packet.extend([i])
                    x_packet.extend(index)
                    x_test_packet.append(x_packet)
        with open("./data/{0}/{1}_nTree{2}/x_test_8bit.dat".format(arg['dataset'], arg['model'], arg['numTreePerClass']), 'w') as file:
            for row in x_test_packet:
                file.write(' '.join([str(item) for item in row]))
                file.write('\n')
        with open("./data/{0}/y_test.dat".format(arg['dataset']), 'w') as file:
            for row in y_test:
                file.write(str(row))
                file.write('\n')

def readArgPost():
    """! 
    @brief      Read the post-precess arguments.
    @details    

    @param      dataset:    Name of dataset             {churn, eye, forest, gas, gesture, telco, rossmann}
    @param      model:      Type of model               {catboost, xgboost, rf}
    @param      numTree:    Number of tree per class    {202}

    @return     arg
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="churn", help="Name of dataset")
    ap.add_argument("--model", type=str, default="catboost", help="Type of model", choices=["catboost", "xgboost", "rf"])
    ap.add_argument("--numTree", type=int, default=202, help="Number of tree per class")
    ap.add_argument("--simulationID", type=str, help=" " )
    args = ap.parse_args()

    arg = {}
    arg['dataset'] = args.dataset
    arg['model'] = args.model
    arg['numTreePerClass'] = args.numTree
    arg['simulationID'] = args.simulationID

    return arg

def configureLog(arg):
    """! 
    @brief      Make a folder for saving log files.
    @details    The name of folder is the time the simulation starts. (format: %y%m%d%H%M%S) The configuration file also is copied with the arguments.
    @return     logDirPath
    """
    dirPath = "./log/{0}/{1}_nTree{2}/".format(arg['dataset'], arg['model'], arg['numTreePerClass'])
    logDir = time.strftime("%y%m%d%H%M%S", time.localtime())
    logDirPath = os.path.join(dirPath, logDir) + "/"
    os.makedirs(logDirPath+"/statistics/")

    logFile = dirPath+logDir+".json"
    temp = readJson(arg['config'])
    temp['arg'] = arg
    json_object = json.dumps(temp, indent=4)
    with open(logFile, 'w') as f:
        f.write(json_object)
    f.close()
 
    return logDirPath

def calculateArea(arg, componentConfig, modelConfig, outputDir):
    """! 
    @brief      Calculate area.
    @details    
    """
    noc = arg['noc']
    numPortControl = componentConfig[noc]['numPortControl']
    numPort = componentConfig[noc]['numPort']
    numLevel = componentConfig[noc]['numLevel']
    numTotalCore = numPortControl*pow(numPort, numLevel)
    numTotalRouter = numPortControl*(pow(numPort, numLevel)-1)/(numPort-1)

    acamQueue = componentConfig['acam']['acamQueue']
    acamStack = componentConfig['acam']['acamStack']
    acamCol = componentConfig['acam']['acamCol']
    acamRow = componentConfig['acam']['acamRow']

    numFeature = modelConfig['numFeature']
    numClass = modelConfig['numClass']
    numTreePerClass = modelConfig['numTreePerClass']

    sizeData = numFeature*2*componentConfig['acam']['gBit'] + numLevel*np.ceil(np.log2(numPort)) + np.ceil(np.log2(numPortControl)) + np.ceil(np.log2(arg['numSample']))
    sizeResult = 32 + np.ceil(np.log2(arg['numSample'])) + np.ceil(np.log2(numClass)) + np.ceil(np.log2(numTreePerClass))

    area = {}
    area['acam.array'] = numTotalCore*acamQueue*acamStack*acamCol*(acamRow*componentConfig['area']['array']) 
    area['acam.dac'] = numTotalCore*acamQueue*acamCol*2*(componentConfig['acam']['gBit']*(componentConfig['area']['finger'] + 1.5*componentConfig['area']['mirror']))
    area['acam.sa'] = numTotalCore*acamQueue*acamStack*acamRow*(9*componentConfig['area']['finger'])
    area['acam.pc'] = numTotalCore*acamQueue*acamStack*acamRow*(2*componentConfig['area']['finger'] + componentConfig['area']['gate'])
    area['acam.reg'] = numTotalCore*acamStack*acamRow*(componentConfig['area']['FF'])
    area['acam.peri'] = area['acam.dac'] + area['acam.sa'] + area['acam.pc'] + area['acam.reg']
    area['acam'] = area['acam.array'] + area['acam.peri']
    
    area['driver'] = numTotalCore*acamQueue*acamCol*(componentConfig['area']['FF'])
    area['mmr'] = numTotalCore*acamStack*acamRow*(2*componentConfig['area']['finger'] + componentConfig['area']['gate'])
    area['memory'] = numTotalCore*acamStack*acamRow*32*(componentConfig['area']['sram'])
    area['adder'] = numTotalCore*32*(componentConfig['area']['adder'])
    area['core'] = area['acam'] + area['driver'] + area['mmr'] + area['memory'] + area['adder']

    area['demux'] = numTotalRouter*sizeData*(componentConfig['area']['demux'] + componentConfig['area']['FF']*(numPort+1))
    area['accumulator'] = numTotalRouter*sizeResult*(componentConfig['area']['adder'] + componentConfig['area']['FF']*(numPort+1))
    area['router'] = area['demux'] + area['accumulator']

    area['total'] = area['core'] + area['router']

    areaFile = outputDir+"area.csv"
    with open(areaFile, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in area.items():
            writer.writerow(row)
    f.close()

def calculatePower(arg):
    numSample = arg['numSample']

    dirPath = "./log/{dataset}/{model}_nTree{numTree}/".format(dataset = arg['dataset'], model = arg['model'], numTree = arg['numTree'])
    configName = os.scandir(dirPath)
    noc = arg['noc']

    powerLabel = ['DL', 'ML', 'SL', 'DAC', 'SA', 'PC', 'REG', 'DRIVER', 'MMR', 'MEMORY', 'ADDER', 'DEMUX', 'ACCUM']
    for conf in configName:
        if conf.is_dir():
            confPath = dirPath+conf.name
            parameter = readJson(confPath)
            numPortControl = parameter[noc]['numPortControl']
            numPort = parameter[noc]['numPort']
            numLevel = parameter[noc]['numLevel']
            numTotalCore = numPortControl*pow(numPort,numLevel)

            powerRaw = np.zeros((numTotalCore, len(powerLabel)))

            fileName = os.scandir(confPath+"/statistics")

            for file in fileName:
                if re.match(r'Output_\d+.csv', file.name):
                    f = open(confPath+"/statistics/"+file.name)
                    f.readline()
                    if (f.readline().strip()):
                        temp = np.loadtxt(confPath+"/statistics/"+file.name, delimiter=", ", skiprows = 1, \
                            dtype = {'names' : ('ComponentName', 'StatisticName', 'StatisticSubId', 'StatisticType', 'SimTime', 'Rank', 'Sum.f64', 'SumSQ.f64', 'Count.u64', 'Min.f64', 'Max.f64'), \
                                    'formats' : ('U32', 'U32', 'U32', 'U32', 'i8', 'i8', 'f8', 'f8', 'i8', 'f8', 'f8')})
                        for j in range(np.shape(temp)[0]):
                            coreID = int(re.findall(r'\d+', temp[j][0])[0])
                            powerName = re.findall(r'power(\w+)', temp[j][1])[0]
                            powerID = powerLabel.index(powerName)
                            powerRaw[coreID, powerID] += temp[j][6]
                    f.close()
            powerAvg = np.sum(powerRaw, axis=0)/numSample
            powerFile = confPath+"/"+"power.csv"
            with open(powerFile, 'w', newline='') as f:
                writer = csv.writer(f)
                for i, name in enumerate(powerLabel):
                    writer.writerow([name, powerAvg[i]])
                writer.writerow(['Total', np.sum(powerAvg)])
            f.close()

def readAutoArg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numRank", type=int, default=1, help="Number of openmpi ranks")
    ap.add_argument("--numThread", type=int, default=1, help="Number of sst threads")

    ap.add_argument("--dataset", type=str, default="churn", help="Name of dataset")
    ap.add_argument("--task", type=str, default="classification", help="Type of task", choices=["classification", "regression", "SHAP"])
    ap.add_argument("--model", type=str, default="catboost", help="Type of model", choices=["catboost", "xgboost", "rf"])
    ap.add_argument("--numTree", type=int, default=202, help="Number of tree per class")
    ap.add_argument("--numSample", type=int, default=2000, help="Number of test samples")
    ap.add_argument("--numBatch", type=int, default=1, help="Batch size")
    
    ap.add_argument("--noc", type=str, default="tree", help="Type of NoC", choices=["tree", "bus", "booster"])
    ap.add_argument("--config", type=str, default="default", help="Name of default configuration file")
    ap.add_argument("--hyperparam", type=str, default="hyperparam", help="Name of hyperparameter file" )
    ap.add_argument("--verbose", type=int, default=1, help="Output verbosity", choices=range(6))
    ap.add_argument("--printFlag", type=str, default="yes", help="Print on prompt", choices=["yes", "no"])
    args = ap.parse_args()

    arg = {}
    arg['numRank'] = args.numRank
    arg['numThread'] = args.numThread

    arg['dataset'] = args.dataset
    arg['task'] = args.task
    arg['model'] = args.model
    arg['numTree'] = args.numTree
    arg['numSample'] = args.numSample
    arg['numBatch'] = args.numBatch

    arg['noc'] = args.noc
    arg['config'] = args.config
    arg['hyperparam'] = args.hyperparam
    arg['verbose'] = args.verbose
    arg['printFlag'] = args.printFlag
    return arg

def readJson(fileName):
    fileName = fileName + ".json"
    try:
        with open(fileName, 'r') as f:
            param = json.load(f)
    except Exception as err:
        sys.exit('FATAL: Not found `{}` - {}'.format(fileName, err) )
    return param

def parseParameter(parameters):
    parsedParameters = {}
    for comp in parameters.keys():
        parameter = parameters[comp]
        for k in parameter.keys():
            v = parameter[k]
            if v["type"] == "range":
                parsedParameters[comp+"."+k] = [i for i in range(v["min"], v["max"]+1, v["step"])]
            elif v["type"] == "constant":
                parsedParameters[comp+"."+k] = [v["value"]]
            elif v["type"] == "choice":
                parsedParameters[comp+"."+k] = v["value"]
    return parsedParameters

def updateConfig(fileName, parsedParameters, listConfig):
    config = readJson(fileName)
    for i, k in enumerate(parsedParameters.keys()):
        strID = re.findall(r'(\w+).(\w+)', k)[0]
        compID = strID[0]
        paraID = strID[1]
        config[compID][paraID] = listConfig[i]
    json_object = json.dumps(config, indent = 4)
    with open ("auto.json", 'w') as f:
        f.write(json_object)
    f.close()

def runSST(arg):
    command = "mpirun -n {numRank:d} sst --num_threads={numThread:d} ./tests/test_general.py -- --dataset={dataset} --model={model} --numTree={numTree:d} --numSample={numSample:d} --numBatch={numBatch:d} --config={config} --noc={noc} --task={task} --verbose={verbose} --printFlag={printFlag}".format(numRank = arg['numRank'], numThread = arg['numThread'], dataset = arg['dataset'], model = arg['model'], numTree = arg['numTree'], numSample = arg['numSample'], numBatch = arg['numBatch'], config = "auto", noc = arg['noc'], task = arg['task'], verbose = arg['verbose'], printFlag = arg['printFlag'])
    args = command.split(" ")
    subprocess.run(args)

