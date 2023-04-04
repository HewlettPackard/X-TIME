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

def configureNoc(arg, componentConfig, modelConfig, outputDir):
    """! 
    @brief      Configure NoC.
    @details    According to the compiling options, convert the model to the data for programming the components.

    @param[in]  arg:                Arguments(dict) from 'readArg'
    @param[in]  componentConfig:    Component configuration (dict) from 'readComponent'
    @param[in]  modelConfig:        Model configuration (dict) from 'readModel'
    @param      mappingCore:        Compiling option for core   {continuous, numTreePerCore}
    @param      mappingNoc:         Compiling option for NoC    {continuous, numCorePerBatch, numCorePerClass}
    
    @return     nocConfig, inputConfig
    """
    if arg['printFlag'] == "yes":
        print("\nConfiguring NoC...")
    
    # Model and componenent parameters
    # Model
    numFeature = modelConfig['info']['numFeature']
    numClass = modelConfig['info']['numClass']
    numTreePerClass = modelConfig['info']['numTreePerClass']

    # NoC
    noc = arg['noc']
    numPortControl = componentConfig[noc]['numPortControl']
    numPort = componentConfig[noc]['numPort']
    numLevel = componentConfig[noc]['numLevel']
    sizeBus = componentConfig[noc]['sizeBus']
    numBatch = arg['numBatch']

    # Link Latency
    # Update link latency
    sizeData, sizeResult, dataLatency, resultLatency, linkLatency= updateLinkLatency(arg, componentConfig, modelConfig['info'])
    componentConfig['linkLatency'].update(linkLatency)

    # Mapping core
    modelCore, numCore = mapCore(arg, componentConfig, modelConfig)

    # Mapping NoC
    acamThL, acamThH, acamXL, acamXH, acamValue, acamClassID, indexCore = mapNoc(arg, componentConfig, modelConfig['info'], modelCore, numCore)

    # Configure router
    routerConfig, inputConfig, numInputConfig = configureRouter(arg, componentConfig, indexCore)

    # aCAM conductance map
    gBit = componentConfig['acam']['gBit']
    gMax = componentConfig['acam']['gMax']
    gMin = componentConfig['acam']['gMin']

    gDelta = (gMax-gMin)/(2**gBit-1)
    gLow = gDelta*np.arange(2**gBit-1) + gMin
    gHigh = gDelta*np.arange(2**gBit-1) + gMin

    gHRS = componentConfig['acam']['gHRS']
    gLRS = gMax + gDelta

    nocConfig = {}

    nocConfig['model'] = {}
    nocConfig['model']['numFeature'] = modelConfig['info']['numFeature']
    nocConfig['model']['numClass'] = modelConfig['info']['numClass']
    nocConfig['model']['numLeafPerClass'] = modelConfig['info']['numLeafPerClass']
    nocConfig['model']['numTreePerClass'] = modelConfig['info']['numTreePerClass']
    
    nocConfig['noc'] = componentConfig[noc]

    nocConfig['core'] = {}
    nocConfig['core']['linkLatency'] = componentConfig['linkLatency']
    nocConfig['core']['verbose'] = arg['verbose']
    nocConfig['core']['numPort'] = numPort
    nocConfig['core']['numPortControl'] = numPortControl
    nocConfig['core']['numLevel'] = numLevel
    nocConfig['core']['outputDir'] = outputDir
    nocConfig['core']['task'] = arg['task']
    
    nocConfig['core']['driver'] = componentConfig['driver']
    nocConfig['core']['driver']['task'] = arg['task']
    nocConfig['core']['driver']['numFeature'] = modelConfig['info']['numFeature']
    nocConfig['core']['driver']['acamQueue'] = componentConfig['acam']['acamQueue']
    nocConfig['core']['driver']['acamStack'] = componentConfig['acam']['acamStack']
    nocConfig['core']['driver']['acamCol'] = componentConfig['acam']['acamCol']
    nocConfig['core']['driver']['acamRow'] = componentConfig['acam']['acamRow']

    nocConfig['core']['acam'] = componentConfig['acam']
    nocConfig['core']['acam']['task'] = arg['task']

    nocConfig['core']['acam']['power.gHRS'] = gHRS
    nocConfig['core']['acam']['power.gLRS'] = gLRS
    nocConfig['core']['acam']['power.GLow'] = gLow.flatten().tolist()
    nocConfig['core']['acam']['power.GHigh'] = gHigh.flatten().tolist()
    
    nocConfig['core']['acamThLow'] = acamThL
    nocConfig['core']['acamThHigh'] = acamThH
    nocConfig['core']['acamXLow'] = acamXL.astype(np.uintc)
    nocConfig['core']['acamXHigh'] = acamXH.astype(np.uintc)

    nocConfig['core']['mmr'] = componentConfig['mmr']
    nocConfig['core']['mmr']['numPort'] = componentConfig['acam']['acamStack']
    nocConfig['core']['mmr']['numMatch'] = componentConfig[noc]['numTreePerCore'] if componentConfig[noc]['mappingCore'] != 'continuous' else -1

    nocConfig['core']['memory'] = componentConfig['memory']
    
    nocConfig['core']['logit'] = acamValue
    nocConfig['core']['classID'] = acamClassID

    nocConfig['core']['adder'] = componentConfig['adder']
    nocConfig['core']['adder']['numMatch'] = componentConfig[noc]['numTreePerCore']*componentConfig[noc]['numTreePerCoreCol'] if componentConfig[noc]['mappingCore'] != 'continuous' else 1

    nocConfig['core']['mpe'] = componentConfig['mpe']
    nocConfig['core']['mpe']['numFeature'] = numFeature
    nocConfig['core']['mpe']['numPort'] = componentConfig['acam']['acamStack']
    nocConfig['core']['mpe']['acamRow'] = componentConfig['acam']['acamRow']
    nocConfig['core']['mpe']['weightFile'] = "./data/{0}/weight_".format(arg['dataset'])

    nocConfig['demux'] = componentConfig['demux']
    nocConfig['demux'].update(routerConfig)
    nocConfig['demux']['verbose'] = arg['verbose']
    nocConfig['demux']['outputQueueLatency'] = dataLatency
    nocConfig['demux']['numPort'] = numPort
    nocConfig['demux']['numPortControl'] = numPortControl
    nocConfig['demux']['numLevel'] = numLevel
    nocConfig['demux']['linkLatency'] = componentConfig['linkLatency']['Demux-Demux']
    nocConfig['demux']['outputDir'] = outputDir
    nocConfig['demux']['power.sizeData'] = int(sizeData)

    nocConfig['accumulator'] = componentConfig['accumulator']
    nocConfig['accumulator']['verbose'] = arg['verbose']
    nocConfig['accumulator']['outputQueueLatency'] = resultLatency
    nocConfig['accumulator']['numPort'] = numPort
    nocConfig['accumulator']['numPortControl'] = numPortControl
    nocConfig['accumulator']['numLevel'] = numLevel
    nocConfig['accumulator']['linkLatency'] = componentConfig['linkLatency']['Accumulator-Accumulator']
    nocConfig['accumulator']['outputDir'] = outputDir
    nocConfig['accumulator']['mode'] = 1 if noc == 'bus' else 0
    nocConfig['accumulator']['power.sizeData'] = int(sizeResult)

    nocConfig['control'] = componentConfig['control']
    nocConfig['control']['verbose'] = arg['verbose']
    nocConfig['control']['outputQueueLatency'] = dataLatency
    nocConfig['control']['task'] = arg['task']
    nocConfig['control']['mode'] = routerConfig[0][0]['mode']
    nocConfig['control']['numPort'] = numPortControl
    nocConfig['control']['numFeature'] = numFeature #if arg['task'] == 'classification' or arg['task'] == 'regression' else 2*numFeature
    nocConfig['control']['numSample'] = arg['numSample']
    nocConfig['control']['numLevel'] = numLevel
    nocConfig['control']['numClass'] = numClass
    nocConfig['control']['numTreePerClass'] = arg['numTreePerClass']
    nocConfig['control']['numBatch'] = numBatch
    nocConfig['control']['numInputConfig'] = numInputConfig.flatten().tolist()
    nocConfig['control']['xFile'] = "./data/{0}/{1}_nTree{2}/x_test_8bit.dat".format(arg['dataset'], arg['model'], arg['numTreePerClass'])
    nocConfig['control']['yFile'] = "./data/{0}/y_test.dat".format(arg['dataset'])
    nocConfig['control']['outputDir'] = outputDir

    if arg['printFlag'] == "yes":
        print("Number of cores per class: {}".format(numCore))
        print("Input configuration (#Batch: [Control, Demux at level 1, Demux at level 2, ..., Demux at level numLevel]):")
        print(inputConfig)

    return nocConfig, inputConfig

def updateLinkLatency(arg, componentConfig, modelConfig):
    noc = arg['noc']
    bitQuant = 8
    sizeBus = componentConfig[noc]['sizeBus']
    sizeData = modelConfig['numFeature']*bitQuant + componentConfig[noc]['numLevel']*np.ceil(np.log2(componentConfig[noc]['numPort'])) + np.ceil(np.log2(componentConfig[noc]['numPortControl'])) + np.ceil(np.log2(arg['numSample']))
    sizeResult = 32 + np.ceil(np.log2(arg['numSample'])) + np.ceil(np.log2(modelConfig['numClass'])) + np.ceil(np.log2(modelConfig['numTreePerClass']))

    dataLatency = np.ceil(float(sizeData)/sizeBus).astype(np.uintc)
    resultLatency = np.ceil(float(sizeResult)/sizeBus).astype(np.uintc)
    print(dataLatency, resultLatency)
    linkLatency = componentConfig['linkLatency']
    for k in linkLatency.keys():
        if linkLatency[k] == 0:
            if k in ['Control-Demux', 'Demux-Demux', 'Demux-Core']:
                linkLatency[k] = dataLatency-1
            elif k in ['Core-Accumulator', 'Accumulator-Accumulator', 'Accumulator-Control']:
                linkLatency[k] = resultLatency-1
        else:
            linkLatency[k] += -1
        
        if linkLatency[k] == 0:
            linkLatency[k] = 0.01
        linkLatency[k] = "{}ns".format(linkLatency[k])
    return sizeData, sizeResult, dataLatency, resultLatency, linkLatency

def mapCore(arg, componentConfig, modelConfig):
    model = modelConfig['model']
    numFeature = modelConfig['info']['numFeature']
    numClass = modelConfig['info']['numClass']
    numLeafPerClass = modelConfig['info']['numLeafPerClass']
    numTreePerClass = int(modelConfig['info']['numTreePerClass'])

    acamQueue = componentConfig['acam']['acamQueue']
    acamStack = componentConfig['acam']['acamStack']
    acamCol = componentConfig['acam']['acamCol']
    acamRow = componentConfig['acam']['acamRow']
    coreCol = acamQueue*acamCol
    coreRow = acamStack*acamRow

    noc = arg['noc']
    mappingCore = componentConfig[noc]['mappingCore']
    numTreePerCore = componentConfig[noc]['numTreePerCore']
    numTreePerCoreCol = int(componentConfig[noc]['numTreePerCoreCol'])

    if (coreCol < numTreePerCoreCol*numFeature):
        raise TypeError("Number of columns in a core is smaller than it needed.")
    
    modelCore = {}
    numCore = np.zeros(numClass).astype(np.uintc)
    if mappingCore == 'continuous':
        for i in range(numClass):
            modelCore[i] = {}
            numCore[i] = (np.ceil(numLeafPerClass[i]/coreRow)).astype(np.uintc)
            raw = model[i]
            mod = int(numCore[i]*coreRow - numLeafPerClass[i])
            temp = np.pad(raw, ((0, mod), (0,0)), 'constant', constant_values=0)
            tempL = temp[:, 0:-3:2]
            tempH = temp[:, 1:-3:2]
            tempL = np.pad(tempL, ((0,0), (0, coreCol-numFeature)), 'constant', constant_values=np.nan)
            tempH = np.pad(tempH, ((0,0), (0, coreCol-numFeature)), 'constant', constant_values=np.nan)
            
            thL = np.zeros((numCore[i], acamStack, acamQueue, acamRow, acamCol))
            thH = np.zeros((numCore[i], acamStack, acamQueue, acamRow, acamCol))
            value = np.zeros((numCore[i], coreRow))
            thL[:,:,:,:,:] = np.nan
            thH[:,:,:,:,:] = np.nan

            for j in range(numCore[i]):
                for s in range(acamStack):
                    begin = j*coreRow + s*acamRow
                    for q in range(acamQueue):
                        thL[j, s, q, :, :] = tempL[begin:begin+acamRow, q*acamCol:(q+1)*acamCol]
                        thH[j, s, q, :, :] = tempH[begin:begin+acamRow, q*acamCol:(q+1)*acamCol]
                value[j, :] = temp[j*coreRow:(j+1)*coreRow, -3]
            modelCore[i]['thL'] = thL
            modelCore[i]['thH'] = thH
            modelCore[i]['value'] = value
    elif mappingCore == 'numTreePerCore':
        for c in range(numClass):
            raw = model[c]
            tempModelCore = {}
            tempModelCore['th_map'] = {}
            tempModelCore['numTree'] = {}
            for i in range(numTreePerClass):
                tempModelCore['th_map'][i] = []
                tempModelCore['numTree'][i] = 0
            
            for i in range(numTreePerClass):
                temp = raw[raw[:, -1] == i].tolist()
                numLeaf = len(temp)
                for j in range(numTreePerClass):
                    if (tempModelCore['numTree'][j] < numTreePerCore) and (len(tempModelCore['th_map'][j]) + numLeaf <= coreRow):
                        tempModelCore['th_map'][j].extend(temp)
                        tempModelCore['numTree'][j] += 1
                        break
            
            for i in range(numTreePerClass):
                if tempModelCore['numTree'][i] == 0:
                    _ = tempModelCore['numTree'].pop(i)
                    _ = tempModelCore['th_map'].pop(i)

            numCore[c] = len(tempModelCore['numTree'])
            thL = np.zeros((numCore[c], acamStack, acamQueue, acamRow, acamCol))
            thH = np.zeros((numCore[c], acamStack, acamQueue, acamRow, acamCol))
            value = np.zeros((numCore[c], coreRow))
            thL[:,:,:,:,:] = np.nan
            thH[:,:,:,:,:] = np.nan    

            for i in range(numCore[c]):
                mod = int(coreRow - len(tempModelCore['th_map'][i]))
                tempLH = np.pad(tempModelCore['th_map'][i], ((0, mod), (0, 0)), 'constant', constant_values=0)
                tempL = tempLH[:, 0:-3:2]
                tempH = tempLH[:, 1:-3:2]
                tempL = np.pad(tempL, ((0, 0), (0, int(coreCol-numFeature))), 'constant', constant_values=np.nan)
                tempH = np.pad(tempH, ((0, 0), (0, int(coreCol-numFeature))), 'constant', constant_values=np.nan)
                for s in range(acamStack):
                    begin = s*acamRow
                    for q in range(acamQueue):
                        thL[i, s, q, :, :] = tempL[begin:begin+acamRow, q*acamCol:(q+1)*acamCol]
                        thH[i, s, q, :, :] = tempH[begin:begin+acamRow, q*acamCol:(q+1)*acamCol]
                value[i, :] = tempLH[:, -3]

            modelCore[c] = {}
            modelCore[c]['thL'] = thL
            modelCore[c]['thH'] = thH
            modelCore[c]['value'] = value

    # FIXME:
    elif mappingCore == 'numTreePerCoreCol':
        for c in range(numClass):
            raw = model[c]
            tempModelCore = {}
            tempModelCore['th_map'] = {}
            tempModelCore['numTree'] = {}
            for i in range(numTreePerClass):
                tempModelCore['th_map'][i] = []
                tempModelCore['numTree'][i] = 0
            
            for i in range(numTreePerClass):
                temp = raw[raw[:, -1] == i].tolist()
                numLeaf = len(temp)
                for j in range(numTreePerClass):
                    if (tempModelCore['numTree'][j] < numTreePerCore) and (len(tempModelCore['th_map'][j]) + numLeaf <= coreRow):
                        tempModelCore['th_map'][j].extend(temp)
                        tempModelCore['numTree'][j] += 1
                        break
            
            for i in range(numTreePerClass):
                if tempModelCore['numTree'][i] == 0:
                    _ = tempModelCore['numTree'].pop(i)
                    _ = tempModelCore['th_map'].pop(i)

            numCore[c] = np.ceil(len(tempModelCore['numTree'])/numTreePerCoreCol).astype(int)
            thL = np.zeros((numCore[c], acamStack, acamQueue, acamRow, acamCol))
            thH = np.zeros((numCore[c], acamStack, acamQueue, acamRow, acamCol))
            value = np.zeros((numCore[c], acamStack*acamRow*numTreePerCoreCol))
            thL[:,:,:,:,:] = np.nan
            thH[:,:,:,:,:] = np.nan    

            for i in range(numCore[c]):
                i_tree = int(i*numTreePerCoreCol)
                mod = int(coreRow - len(tempModelCore['th_map'][i_tree]))
                tempLH = np.pad(tempModelCore['th_map'][i_tree], ((0, mod), (0, 0)), 'constant', constant_values=0)
                tempL = tempLH[:, 0:-3:2]
                tempH = tempLH[:, 1:-3:2]
                tempV = tempLH[:, -3]
                
                for j in range(1, numTreePerCoreCol):
                    i_tree = int(i*numTreePerCoreCol + j)
                    if i_tree in tempModelCore['th_map'].keys():
                        mod = int(coreRow - len(tempModelCore['th_map'][i_tree]))
                        tempLH = np.pad(tempModelCore['th_map'][i_tree], ((0, mod), (0, 0)), 'constant', constant_values=0)
                        tempL = np.hstack((tempL, tempLH[:, 0:-3:2]))
                        tempH = np.hstack((tempH, tempLH[:, 1:-3:2]))
                        tempV = np.hstack((tempV, tempLH[:, -3]))
                
                tempL = np.pad(tempL, ((0, 0), (0, int(coreCol-numTreePerCoreCol*numFeature))), 'constant', constant_values=np.nan)
                tempH = np.pad(tempH, ((0, 0), (0, int(coreCol-numTreePerCoreCol*numFeature))), 'constant', constant_values=np.nan)
                
                for s in range(acamStack):
                    begin = s*acamRow
                    for q in range(acamQueue):
                        thL[i, s, q, :, :] = tempL[begin:begin+acamRow, q*acamCol:(q+1)*acamCol]
                        thH[i, s, q, :, :] = tempH[begin:begin+acamRow, q*acamCol:(q+1)*acamCol]
                    value[i, :] = tempV[:]

            modelCore[c] = {}
            modelCore[c]['thL'] = thL
            modelCore[c]['thH'] = thH
            modelCore[c]['value'] = value
    else:
        raise TypeError("Unknown compile option for mapping core.")
    
    return modelCore, numCore

def mapNoc(arg, componentConfig, modelConfig, modelCore, numCore):
    # Compiling option for NoC    {continuous, numCorePerBatch, numCorePerClass}
    
    numClass = modelConfig['numClass']
    
    # Core
    acamQueue = componentConfig['acam']['acamQueue']
    acamStack = componentConfig['acam']['acamStack']
    acamCol = componentConfig['acam']['acamCol']
    acamRow = componentConfig['acam']['acamRow']
    
    noc = arg['noc']
    mappingNoc = componentConfig[noc]['mappingNoc']
    numCoreClass = componentConfig[noc]['numCoreClass']
    numCoreBatch = componentConfig[noc]['numCoreBatch']
    numPortControl = componentConfig[noc]['numPortControl']
    numPort = componentConfig[noc]['numPort']
    numLevel = componentConfig[noc]['numLevel']
    numTotalCore = numPortControl*pow(numPort, numLevel)
    numBatch = arg['numBatch']

    acamThL = np.zeros((numTotalCore, acamStack, acamQueue, acamRow, acamCol))
    acamThH = np.zeros((numTotalCore, acamStack, acamQueue, acamRow, acamCol))
    acamXL = np.zeros((numTotalCore, acamStack, acamQueue, acamRow, acamCol))
    acamXH = np.zeros((numTotalCore, acamStack, acamQueue, acamRow, acamCol))
    acamValue = np.zeros((numTotalCore, acamStack*acamRow))
    acamClassID = -1*np.ones(numTotalCore)
    indexCore = -1*np.ones((numTotalCore, 2))   # Batch ID, Class ID

    if mappingNoc == 'continuous':
        for i in range(numBatch):
            for j in range(numClass):
                begin = i*int(np.sum(numCore))+int(np.sum(numCore[:j]))
                acamThL[begin:begin+numCore[j],:,:,:,:] = np.nan_to_num(modelCore[j]['thL'])
                acamThH[begin:begin+numCore[j],:,:,:,:] = np.nan_to_num(modelCore[j]['thH'])
                acamXL[begin:begin+numCore[j],:,:,:,:] = np.isnan(modelCore[j]['thL'])
                acamXH[begin:begin+numCore[j],:,:,:,:] = np.isnan(modelCore[j]['thH'])
                acamValue[begin:begin+numCore[j],:] = modelCore[j]['value']
                acamClassID[begin:begin+numCore[j]] = j

                indexCore[begin:begin+numCore[j], 0] = i
                indexCore[begin:begin+numCore[j], 1] = j
        
    elif mappingNoc == 'numCoreClass':
        for i in range(numBatch):
            numMin = 0
            for j in range(numClass):
                begin = numMin*numCoreClass + i*numCoreBatch
                if np.any(indexCore[begin:begin+numCore[j], 0] != -1):
                    raise TypeError("Overlapped core mapping")
                acamThL[begin:begin+numCore[j],:,:,:,:] = np.nan_to_num(modelCore[j]['thL'])
                acamThH[begin:begin+numCore[j],:,:,:,:] = np.nan_to_num(modelCore[j]['thH'])
                acamXL[begin:begin+numCore[j],:,:,:,:] = np.isnan(modelCore[j]['thL'])
                acamXH[begin:begin+numCore[j],:,:,:,:] = np.isnan(modelCore[j]['thH'])
                acamValue[begin:begin+numCore[j],:] = modelCore[j]['value']
                acamClassID[begin:begin+numCore[j]] = j

                indexCore[begin:begin+numCore[j], 0] = i
                indexCore[begin:begin+numCore[j], 1] = j
                numMin += (np.ceil(float(numCore[j])/numCoreClass)).astype(np.uintc)
    elif mappingNoc == 'numCoreBatch':
        numCoreBatch = int(max(numCoreBatch, np.sum(numCore)))
        if numCoreBatch*numBatch > numTotalCore:
            raise TypeError("Unable to map {0:d} cores to {1:d} physical cores".format(numCoreBatch*numBatch, numTotalCore))
        else:
            for i in range(numBatch):
                for j in range(numClass):
                    begin = i*numCoreBatch + int(np.sum(numCore[:j]))
                    if np.any(indexCore[begin:begin+numCore[j], 0] != -1):
                        raise TypeError("Overlapped core mapping")
                    acamThL[begin:begin+numCore[j],:,:,:,:] = np.nan_to_num(modelCore[j]['thL'])
                    acamThH[begin:begin+numCore[j],:,:,:,:] = np.nan_to_num(modelCore[j]['thH'])
                    acamXL[begin:begin+numCore[j],:,:,:,:] = np.isnan(modelCore[j]['thL'])
                    acamXH[begin:begin+numCore[j],:,:,:,:] = np.isnan(modelCore[j]['thH'])
                    acamValue[begin:begin+numCore[j],:] = modelCore[j]['value']
                    acamClassID[begin:begin+numCore[j]] = j

                    indexCore[begin:begin+numCore[j], 0] = i
                    indexCore[begin:begin+numCore[j], 1] = j
    else:
        raise TypeError("Unknown compile option for mapping NoC.")

    return acamThL, acamThH, acamXL, acamXH, acamValue, acamClassID, indexCore

def configureRouter(arg, componentConfig, indexCore):
    # Configure router
    noc = arg['noc']
    numPortControl = componentConfig[noc]['numPortControl']
    numPort = componentConfig[noc]['numPort']
    numLevel = componentConfig[noc]['numLevel']
    numBatch = arg['numBatch']

    if noc == 'tree':
        routerConfig = {}
        routerConfig[numLevel] = {}
        for i in range(numPortControl*pow(numPort, numLevel-1)):
            temp = indexCore[i*numPort : (i+1)*numPort, 0]
            index, count = np.unique(temp[temp>=0], return_counts=True)
            mode = []
            indexBatch = []
            indexInputs = []
            offset = -1
            for m, n in enumerate(index):
                d = np.where(temp == n)[0][-1]
                mode.append(d-offset)
                indexBatch.append(n)
                offset = d
                indexInput = [-1 for x in range(numLevel+2)]
                indexInput[0] = n
                indexInput[-1] = m
                indexInputs.append(indexInput)
            routerConfig[numLevel][i] = {}
            routerConfig[numLevel][i]['mode'] = mode
            routerConfig[numLevel][i]['indexBatch'] = indexBatch  
            routerConfig[numLevel][i]['indexInputs'] = indexInputs
            if len(indexBatch) == 0:
                routerConfig[numLevel][i]['indexFlag'] = [-1]
            elif len(indexBatch) == 1:
                routerConfig[numLevel][i]['indexFlag'] = [indexBatch[0]*indexBatch[0]]
            else:
                routerConfig[numLevel][i]['indexFlag'] = [indexBatch[0]*indexBatch[1]]

        for l in range(numLevel-1, 0, -1):
            routerConfig[l] = {}
            for i in range(numPortControl*pow(numPort, l-1)):
                temp = []
                temp_index = []
                for j in range(numPort):    
                    temp.extend(routerConfig[l+1][i*numPort+j]['indexFlag'])
                temp = np.array(temp)
                
                index, count = np.unique(temp[temp>=0], return_counts=True)
                mode = []
                indexBatch = []
                indexInputs = []
                indexFlag = []
                offset = -1
                for m, n in enumerate(index):
                    d = np.where(temp == n)[0][-1]
                    mode.append(d-offset)
                    indexBatch.append(routerConfig[l+1][i*numPort+d]['indexBatch'])
                    indexFlag.append(n)
                    offset = d
                    for k in range(len(routerConfig[l+1][i*numPort+d]['indexInputs'])):
                        indexInput = routerConfig[l+1][i*numPort+d]['indexInputs'][k]
                        indexInput[l+1] = m
                        if (indexInput not in indexInputs):
                            indexInputs.append(indexInput)

                routerConfig[l][i] = {}
                routerConfig[l][i]['mode'] = mode
                routerConfig[l][i]['indexBatch'] = indexBatch
                routerConfig[l][i]['indexInputs'] = indexInputs
                if len(indexBatch) == 0:
                    routerConfig[l][i]['indexFlag'] = [-1]
                elif len(indexBatch) == 1:
                    routerConfig[l][i]['indexFlag'] = [indexFlag[0]*indexFlag[0]]
                else:
                    routerConfig[l][i]['indexFlag'] = [indexFlag[0]*indexFlag[1]]

        routerConfig[0] = {}
        temp = []
        for j in range(numPortControl):
            temp.extend(routerConfig[1][j]['indexFlag'])
        temp = np.array(temp)

        index, count = np.unique(temp[temp>=0], return_counts=True)
        mode = []
        indexBatch = []
        indexInputs = []
        indexFlag = []
        offset = -1
        for m, n in enumerate(index):
            d = np.where(temp == n)[0][-1]
            mode.append(d-offset)
            indexBatch.append(routerConfig[1][d]['indexBatch'])
            indexFlag.append(n)
            offset = d
            for k in range(len(routerConfig[1][d]['indexInputs'])):
                indexInput = routerConfig[1][d]['indexInputs'][k]
                indexInput[1] = m
                if (indexInput not in indexInputs):
                    indexInputs.append(indexInput)
        routerConfig[0][0] = {}
        routerConfig[0][0]['mode'] = mode
        routerConfig[0][0]['indexBatch'] = indexBatch
        routerConfig[0][0]['indexInputs'] = indexInputs
        if len(indexBatch) == 0:
            routerConfig[0][0]['indexFlag'] = [-1]
        elif len(indexBatch) == 1:
            routerConfig[0][0]['indexFlag'] = [indexFlag[0]*indexFlag[0]]
        else:
            routerConfig[0][0]['indexFlag'] = [indexFlag[0]*indexFlag[1]]

    elif noc == 'bus':
        routerConfig = {}
        routerConfig[numLevel] = {}
        for i in range(numPortControl*pow(numPort, numLevel-1)):
            temp = indexCore[i*numPort : (i+1)*numPort, 0]
            index, count = np.unique(temp[temp>=0], return_counts=True)
            mode = []
            indexBatch = []
            indexInputs = []
            offset = -1
            for m, n in enumerate(index):
                d = np.where(temp == n)[0][-1]
                mode.append(d-offset)
                indexBatch.append(n)
                offset = d
                indexInput = [-1 for x in range(numLevel+2)]
                indexInput[0] = n
                indexInput[-1] = m
                indexInputs.append(indexInput)
            routerConfig[numLevel][i] = {}
            routerConfig[numLevel][i]['mode'] = mode
            routerConfig[numLevel][i]['indexBatch'] = indexBatch  
            routerConfig[numLevel][i]['indexInputs'] = indexInputs
            if len(indexBatch) == 0:
                routerConfig[numLevel][i]['indexFlag'] = [-1]
            elif len(indexBatch) == 1:
                routerConfig[numLevel][i]['indexFlag'] = [indexBatch[0]*indexBatch[0]]
            else:
                routerConfig[numLevel][i]['indexFlag'] = [indexBatch[0]*indexBatch[1]]

        for l in range(numLevel-1, 1, -1):
            routerConfig[l] = {}
            for i in range(numPortControl*pow(numPort, l-1)):
                temp = []
                temp_index = []
                for j in range(numPort):    
                    temp.extend(routerConfig[l+1][i*numPort+j]['indexFlag'])
                temp = np.array(temp)
                
                index, count = np.unique(temp[temp>=0], return_counts=True)
                mode = []
                indexBatch = []
                indexInputs = []
                indexFlag = []
                offset = -1
                for m, n in enumerate(index):
                    d = np.where(temp == n)[0][-1]
                    mode.append(d-offset)
                    indexBatch.append(routerConfig[l+1][i*numPort+d]['indexBatch'])
                    indexFlag.append(n)
                    offset = d
                    for k in range(len(routerConfig[l+1][i*numPort+d]['indexInputs'])):
                        indexInput = routerConfig[l+1][i*numPort+d]['indexInputs'][k]
                        indexInput[l+1] = m
                        if (indexInput not in indexInputs):
                            indexInputs.append(indexInput)

                routerConfig[l][i] = {}
                routerConfig[l][i]['mode'] = mode
                routerConfig[l][i]['indexBatch'] = indexBatch
                routerConfig[l][i]['indexInputs'] = indexInputs
                if len(indexBatch) == 0:
                    routerConfig[l][i]['indexFlag'] = [-1]
                elif len(indexBatch) == 1:
                    routerConfig[l][i]['indexFlag'] = [indexFlag[0]*indexFlag[0]]
                else:
                    routerConfig[l][i]['indexFlag'] = [indexFlag[0]*indexFlag[1]]

        l = 1
        routerConfig[l] = {}
        for i in range(numPortControl*pow(numPort, l-1)):
            indexInputs = []
            for j in range(numPort):
                if len(routerConfig[l+1][i*numPort+j]['indexInputs']) != 0:
                    for k in range(len(routerConfig[l+1][i*numPort+j]['indexInputs'])):
                        indexInput = routerConfig[l+1][i*numPort+j]['indexInputs'][k]
                        indexInput[l+1] = j
                        indexInputs.append(indexInput)

            routerConfig[l][i] = {}
            routerConfig[l][i]['mode'] = [1 for x in range(numPort)]
            routerConfig[l][i]['indexInputs'] = indexInputs

        routerConfig[0] = {}
        indexInputs = []
        for j in range(numPortControl):
            if len(routerConfig[1][j]['indexInputs']) != 0:
                for k in range(len(routerConfig[1][j]['indexInputs'])):
                    indexInput = routerConfig[1][j]['indexInputs'][k]
                    indexInput[1] = j
                    indexInputs.append(indexInput)

        routerConfig[0][0] = {}
        routerConfig[0][0]['mode'] = [1 for x in range(numPortControl)]
        routerConfig[0][0]['indexInputs'] = indexInputs
    elif noc == 'booster':
        numLevel = 1
        routerConfig = {}
        routerConfig[1] = {}
        indexInputs = []
        for i in range(numPortControl):
            routerConfig[1][i] = {}
            routerConfig[1][i]['mode'] = [numPort]
            routerConfig[1][i]['indexInputs'] = [indexCore[i*numPort, 0], i, 0]
            indexInputs.append(routerConfig[1][i]['indexInputs'])

        routerConfig[0] = {}
        routerConfig[0][0] = {}
        routerConfig[0][0]['mode'] = [1 for x in range(numPortControl)]
        routerConfig[0][0]['indexInputs'] = indexInputs

    else:
        raise TypeError("Unknown NoC.")

    # Configure input packet
    numInputConfig = np.zeros(numBatch).astype(np.uintc)
    inputConfig = {}
    for i in range(numBatch):
        inputConfig[i] = []
    for i, index in enumerate(routerConfig[0][0]['indexInputs']):
        if index[0]!=-1:
            inputConfig[int(index[0])].append(index[1:])
            numInputConfig[int(index[0])]+=1

    if noc == 'booster':
        numInputConfig = np.ones(1).astype(np.uintc)

    return routerConfig, inputConfig, numInputConfig