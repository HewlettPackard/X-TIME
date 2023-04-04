// 
// Copyright (2023) Hewlett Packard Enterprise Development LP
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 


#include "control.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for control 
* @details Read parameters, open input/truth files, initialize result/resultSumTree, configure output, register clock handler, and configure links.
*/
control::control(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    selfLatency             = params.find<uint32_t>("latency", 1);
    
    task                    = params.find<std::string>("task", "classification");
    numPort                 = params.find<uint32_t>("numPort", 1);
    numFeature              = params.find<uint32_t>("numFeature", 10);
    numSample               = params.find<uint32_t>("numSample", 100);
    numLevel                = params.find<uint32_t>("numLevel", 6);
    numClass                = params.find<uint32_t>("numClass", 1);
    numTreePerClass         = params.find<uint32_t>("numTreePerClass", 1);
    numBatch                = params.find<uint32_t>("numBatch", 1);
    params.find_array("mode", mode);
    params.find_array("numInputConfig", numInputConfig);
    
    /* Open input file(xFile) and ground truth file(yFile) */
    std::string xFile       = params.find<std::string>("xFile");
    std::string yFile       = params.find<std::string>("yFile");
    xStream.open(xFile);
    yStream.open(yFile);

    /* Initialize result (predict, truth, #1~#numClass logit) and resultSumTree (#1~#numClass sumTree) */
    if ((task == "classification") || (task == "regression")){
        result.resize(numSample);
        resultSumTree.resize(numSample);
        latencySample.resize(numSample);
        for (uint32_t i = 0; i < numSample; ++i){
            // result[i] : predict, truth, class#1 result, class#2 result, ..., class#numClass result
            result[i].resize(numClass+2, 0); 
            result[i][0] = -1;
            yStream >> result[i][1];
            // resultSumTree[i] : indicate how many tree results are added to get the result
            resultSumTree[i].resize(numClass, 0);
            latencySample[i] = 0;
        }
    }
    /* Initialize result (sampleID, classID, #1~#numFeature SHAP values) */
    else if (task == "SHAP"){
        result.resize(numSample*numClass);
        for (uint32_t i = 0; i < numSample*numClass; ++i){
            result[i].resize(numFeature+2, 0);
            result[i][0] = i/numClass;
            result[i][1] = i%numClass;
        }
    }
    
    /* Configure output (outSTd: command prompt, outFile: txt file, outFileResult: txt file with 'result') */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Control" + std::to_string(compId) + "]:          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"control"+std::to_string(compId)+".txt");
    outResult.init(prefix, verbose, 0, Output::FILE, outputDir+"result.txt");

    /* Register clock handler */ 
    clockHandler = new Clock::Handler<control>(this, &control::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    auto inputStat = registerStatistic<uint64_t>("iq_size");
    auto outputStat = registerStatistic<uint64_t>("oq_size");
    for (uint32_t i = 0; i < numPort; ++i){
        inputPort.push_back(new Port(i, clockPeriod, outStd, this));
        inputPort[i]->getQueue().setLatency(iqLat);
        inputPort[i]->getQueue().setQueueSizeStat(inputStat);
        inputLink.push_back(configureLink("input_port"+std::to_string(i), new Event::Handler<Port>(inputPort.back(), &Port::handleNewResult)));

        outputPort.push_back(new Port(i, clockPeriod, outStd, this));
        outputPort[i]->getQueue().setLatency(oqLat);
        outputPort[i]->getQueue().setQueueSizeStat(outputStat);
        outputLink.push_back(configureLink("output_port"+std::to_string(i)));
    }
    selfLink = configureSelfLink("self", freq, new Event::Handler<control>(this, &control::handleSelf));
    
    /* Load the first input packet */
    handleLoad();
    
    timeStart = time(NULL);
    
    /* Register it as primary component so that it should be completed to finish the simulation */
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();

}

/**
 * @brief Load the input packet to the outputPort (outputQueue).
 * @details Read the input packet (data + info) from the input file. Load DataEvent (data, info) to the outputPort.
 */
void
control::handleLoad( ){
    if ((task == "classification") || (task == "regression")){
        /* Read the input packet (data + info) from the input file */
        double temp;
        uint32_t sumInputConfig = std::accumulate(numInputConfig.begin(), numInputConfig.end(), 0);
        xMatrix.resize(sumInputConfig);
        xInfo.resize(sumInputConfig);
        for (uint32_t i = 0; i < sumInputConfig; ++i){
            xMatrix[i].resize(numFeature, 0);
            xInfo[i].resize(numLevel+2, -1);
            if (xStream.peek() != EOF){
                for (uint32_t j = 0; j < numFeature; ++j){
                    xStream >> xMatrix[i][j];
                }
                for (uint32_t j = 0; j < numLevel+2; ++j){
                    xStream >> temp;
                    xInfo[i][j] = static_cast<int32_t>(temp);
                }
            }
        }

        /* Load DataEvent (data, info) to the outputPort */ 
        int32_t p;
        uint32_t begin = 0;
        for (uint32_t i = 0; i < sumInputConfig; ++i){
            begin = 0;
            p = xInfo[i][1];
            for (uint32_t j = 0; j < p; ++j){
                begin += mode[j];
            }
            for (uint32_t j = 0; j < mode[p]; ++j){
                DataEvent *foreEv = new DataEvent(xMatrix[i], xInfo[i], xInfo[i][0], -1, 0);
                outputPort[begin+j]->getQueue().push_relative(getNextClockCycle(clockPeriod) - 1, foreEv);
                
            }
        }
        // if (mode.size() == numPort){
        //     numSent = xInfo[0][0];
        // }
        // else{
            // numSent += numBatch;
            numSent += numInputConfig.size();
        // }
    }

    else if (task == "SHAP"){
         /* Read the input packet (data + info) from the input file */
        double temp;
        uint32_t sumInputConfig = std::accumulate(numInputConfig.begin(), numInputConfig.end(), 0);
        xMatrix.resize(sumInputConfig*numSample);
        xInfo.resize(sumInputConfig*numSample);
        for (uint32_t i = 0; i < sumInputConfig*numSample; ++i){
            xMatrix[i].resize(2*numFeature, 0);
            xInfo[i].resize(numLevel+2, -1);
            if (xStream.peek() != EOF){
                for (uint32_t j = 0; j < 2*numFeature; ++j){
                    xStream >> xMatrix[i][j];
                }
                for (uint32_t j = 0; j < numLevel+2; ++j){
                    xStream >> temp;
                    xInfo[i][j] = static_cast<int32_t>(temp);
                }
            }
        }

        /* Load DataEvent (data, info) to the outputPort */ 
        int32_t p;
        uint32_t begin = 0;
        for (uint32_t i = 0; i < sumInputConfig*numSample; ++i){
            begin = 0;
            p = xInfo[i][1];
            for (uint32_t j = 0; j < p; ++j){
                begin += mode[j];
            }
            for (uint32_t j = 0; j < mode[p]; ++j){
                DataEvent *foreEv = new DataEvent(xMatrix[i], xInfo[i], xInfo[i][0], -1, 0);
                outputPort[begin+j]->getQueue().push_relative(getNextClockCycle(clockPeriod) - 1, foreEv);
            }
        }
        numSent += numBatch;
        if (numSent >= numSample){
            // Null event to end the simulation
            DataEvent *foreEv = new DataEvent(std::vector<double>(1, 0), std::vector<int32_t>(1, -1), -1, -1, 0);
            outputPort[0]->getQueue().push_relative(getNextClockCycle(clockPeriod) - 1, foreEv);
        }
        
    }
    load = true;
}

/**
 * @brief Perform classification/regression/SHAP.
 * @details Classification/Regression: Update result and resutSumTree for each DataEvent from the ports. 
 * Predict the class label/final value based on the result when all entry of resultSumTree reach numTreePerClass.\n
 * shap: Update result for each DataEvent from the ports until an event with -1 of sampleID (end signal) is received.
 */
void
control::handleSelf(Event *ev){
    DataMatrixEvent *inEv = static_cast<DataMatrixEvent*>(ev);

    if (task == "classification"){
        uint32_t dataSize = inEv->getDataDouble().size();
        int32_t sampleID, classID;
        /* Update result and resutSumTree for each DataEvent from the ports */
        for (uint32_t i = 0; i < dataSize; ++i){
            sampleID = inEv->getDataInt()[i][0];
            classID = inEv->getDataInt()[i][1];
            if (sampleID < numSample){
                result[sampleID][classID+2] += inEv->getDataDouble()[i][0];
                resultSumTree[sampleID][classID] += inEv->getDataInt()[i][2];
            }
        }
        /* Predict the class label based on the result when all entry of resultSumTree reach numTreePerClass */
        for (uint32_t i = 0; i < dataSize; ++i){
            sampleID = inEv->getDataInt()[i][0];
            classID = inEv->getDataInt()[i][1];
            if (sampleID < numSample){
                uint32_t x = numTreePerClass;
                if (std::all_of(resultSumTree[sampleID].cbegin(), resultSumTree[sampleID].cend(), [x] (uint32_t j){ return j >= x;})){
                    uint32_t predict = 0;
                    double predictLogit = 0.0;
                    if (result[sampleID][0] == -1){
                        if (numClass != 1){
                            predict = std::distance(result[sampleID].begin()+2, std::max_element(result[sampleID].begin()+2, result[sampleID].end()));
                            predictLogit = result[sampleID][predict+2];
                            result[sampleID][0] = static_cast<double>(predict);
                        }
                        else{
                            predictLogit = result[sampleID][2];
                            if ( predictLogit >= 0 ){
                                result[sampleID][0] = 1;
                            }
                            else{
                                result[sampleID][0] = 0;
                            }
                        }
                        if (result[sampleID][0] == result[sampleID][1]){
                            ++numCorrect;
                        }
                        ++numTested;
                        outStd.verbose(CALL_INFO, 1, 0, "ID#%d Truth %d, Predict %d (%.3lf) -> %d/%d = %.3lf\n", sampleID, static_cast<uint32_t>(result[sampleID][1]), static_cast<uint32_t>(result[sampleID][0]), predictLogit, numCorrect, numTested, static_cast<double>(numCorrect)/numTested);
                        outFile.verbose(CALL_INFO, 1, 0, "ID#%d Truth %d, Predict %d (%.3lf) -> %d/%d = %.3lf\n", sampleID, static_cast<uint32_t>(result[sampleID][1]), static_cast<uint32_t>(result[sampleID][0]), predictLogit, numCorrect, numTested, static_cast<double>(numCorrect)/numTested);
                        if (latencySimTime == 0){
                            latencySimTime = getCurrentSimTime();
                        }
                        latencySample[sampleID] = getCurrentSimTime() - latencySample[sampleID];
                    }
                }
            }
        }
    }
    else if (task == "regression"){
        uint32_t dataSize = inEv->getDataDouble().size();
        int32_t sampleID, classID = 0;
        /* Update result and resutSumTree for each DataEvent from the ports */
        for (uint32_t i = 0; i < dataSize; ++i){
            sampleID = inEv->getDataInt()[i][0];
            if (sampleID < numSample){
                // predict, truth, logit
                result[sampleID][classID+2] += inEv->getDataDouble()[i][0];
                resultSumTree[sampleID][classID] += inEv->getDataInt()[i][2];
            }
        }
        /* Predict the final value based on the result when resultSumTree reaches numTreePerClass */
        for (uint32_t i = 0; i < dataSize; ++i){
            sampleID = inEv->getDataInt()[i][0];
            uint32_t x = numTreePerClass;
            if (sampleID < numSample){
                if (std::all_of(resultSumTree[sampleID].cbegin(), resultSumTree[sampleID].cend(), [x] (uint32_t j){ return j >= x;})){
                    result[sampleID][0] = result[sampleID][classID+2];
                    mse += std::pow(result[sampleID][0]-result[sampleID][1], 2);
                    ++numTested;
                    outStd.verbose(CALL_INFO, 1, 0, "ID#%d Truth %.3lf, Predict %.3lf -> RMSE %.3lf\n", sampleID, result[sampleID][1], result[sampleID][0], std::sqrt(mse/numTested));
                    outFile.verbose(CALL_INFO, 1, 0, "ID#%d Truth %.3lf, Predict %.3lf -> RMSE %.3lf\n", sampleID, result[sampleID][1], result[sampleID][0], std::sqrt(mse/numTested));
                    if (latencySimTime == 0){
                        latencySimTime = getCurrentSimTime();
                    }
                    latencySample[sampleID] = getCurrentSimTime() - latencySample[sampleID];
                }
            }
        }
    }
    else if (task == "SHAP"){
        /*  Update result for each DataEvent from the ports until an event with -1 of sampleID (end signal) is received. */
        uint32_t dataSize = inEv->getDataDouble().size();
        int32_t sampleID, classID = 0;
        for (uint32_t i = 0; i < dataSize; ++i){
            sampleID = inEv->getDataInt()[i][0];
            classID = inEv->getDataInt()[i][1];
            if (sampleID != -1){
                std::transform(result[sampleID*numClass+classID].begin()+2, result[sampleID*numClass+classID].end(), inEv->getDataDouble()[i].begin(), result[sampleID*numClass+classID].begin()+2, std::plus<double>());
            }
            else{
                numTested = -1;
            }
            outStd.verbose(CALL_INFO, 1, 0, "ID#%d ClassID#%d SumCore#%d\n", sampleID, classID, inEv->getDataInt()[i][2]);
            outFile.verbose(CALL_INFO, 1, 0, "ID#%d ClassID#%d SumCore#%d\n", sampleID, classID, inEv->getDataInt()[i][2]);
        }
    }
    busy = false;
    delete inEv;
}

/**
 * @brief For each port, load the incoming event to own inputQueue. 
 */
void
control::Port::handleNewResult(Event *ev) {
    portQueue.push(m_control->getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. If all test samples are tested, terminate the simulation and print the simulation results
 * 2. Load the input packet to outputPort (outputQueue) (handleLoad)
 * 3. Pop the output event from the outputPort (outputQueue) and send it through the outputlink
 * 4. Pop the input event from the inputPort (inputQueue) and perfrom the classification or regression (handleSelf)
 */
bool
control::clockTick(Cycle_t cycle) {

    /* Terminate the simulation when all samples are tested */
    if ((task == "classification") && (numTested == numSample)){
        /* Measure the simulation time in the scale of SST global clock cycle */
        time_t timeEnd = time(NULL);
        double seconds = difftime(timeEnd, timeStart);
        SimTime_t finalSimTime = getCurrentSimTime();

        /* Print the summarized simulation results (accuracy, throughput, latency, simulation time) */
        outStd.output("\tAccuracy : %u/%d = %.3lf\n", numCorrect, numTested, static_cast<double>(numCorrect)/numTested);
        outStd.output("\tThroughput : %d/(%ld ns) --> %.3f M samples/s, Latency : %ld ns\n", numTested, finalSimTime, static_cast<float>(1e3*(numTested))/(finalSimTime), latencySimTime);
        outStd.output("\tTime : %.1f\n", seconds);
        outFile.output("\nAccuracy : %u/%d = %.3lf\n", numCorrect, numTested, static_cast<double>(numCorrect)/numTested);
        outFile.output("Throughput : %d/(%ld ns) --> %.3f M samples/s, Latency : %ld ns\n", numTested, finalSimTime, static_cast<float>(1e3*(numTested))/(finalSimTime), latencySimTime);
        outFile.output("Time : %.1f\n", seconds);

        /* Print the result matrix (predict, truth, #1~#numClass logits)*/
        for (uint32_t i = 0; i < numSample; ++i){
            for (uint32_t j = 0; j < numClass+2; ++j){
                outResult.output("%.3lf ", result[i][j]);
            }
            // outResult.output("%ld", latencySample[i]);
            outResult.output("\n");
        }
        /* Okay to end the simulation */
        primaryComponentOKToEndSim();
        return true;
    }
    else if ((task == "regression") && (numTested == numSample)){
        /* Measure the simulation time in the scale of SST global clock cycle */
        time_t timeEnd = time(NULL);
        double seconds = difftime(timeEnd, timeStart);
        SimTime_t finalSimTime = getCurrentSimTime();

        /* Print the summarized simulation results (RMSE, throughput, latency, simulation time) */
        outStd.output("\nRMSE : %.3lf\n", std::sqrt(mse/numTested));
        outStd.output("Throughput : %d/(%ld ns) --> %.3f M samples/s, Latency : %ld ns\n", numTested, finalSimTime, static_cast<float>(1e3*(numTested))/(finalSimTime), latencySimTime);
        outStd.output("Time : %.3f\n", seconds);

        outFile.output("\nRMSE : %.3lf\n", std::sqrt(mse/numTested));
        outFile.output("Throughput : %d/(%ld ns) --> %.3f M samples/s, Latency : %ld ns\n", numTested, finalSimTime, static_cast<float>(1e3*(numTested))/(finalSimTime), latencySimTime);
        outFile.output("Time : %.3f\n", seconds);

        /* Print the result matrix (predict, truth, logit)*/
        for (uint32_t i = 0; i < numSample; ++i){
            for (uint32_t j = 0; j < numClass+2; ++j){
                outResult.output("%.3lf ", result[i][j]);
            }
            outResult.output("\n");
        }
        /* Okay to end the simulation */
        primaryComponentOKToEndSim();
        return true;
    }
    else if ((task == "SHAP") && (numTested == -1)){
        /* Measure the simulation time in the scale of SST global clock cycle */
        time_t timeEnd = time(NULL);
        double seconds = difftime(timeEnd, timeStart);
        SimTime_t finalSimTime = getCurrentSimTime();

        /* Print the summarized simulation results (throughput, simulation time) */
        outStd.output("\nThroughput : %d/(%ld ns) --> %.3f M samples/s\n", numSample, finalSimTime, static_cast<float>(1e3*(numSample))/(finalSimTime));
        outStd.output("Time : %.3f\n", seconds);

        outFile.output("\nThroughput : %d/(%ld ns) --> %.3f M samples/s\n", numSample, finalSimTime, static_cast<float>(1e3*(numSample))/(finalSimTime));
        outFile.output("Time : %.3f\n", seconds);

        /* Print the result matrix (sampleID, classID, SHAP#1, SHAP#2, ..., SHAP#numFeature)*/
        for (uint32_t i = 0; i < numSample*numClass; ++i){
            outResult.output("%.3lf %.3lf ", result[i][0], result[i][1]);
            for (uint32_t j = 2; j < numFeature+2; ++j){
                outResult.output("%.3lf ", result[i][j]/numSample);
            }
            outResult.output("\n");
        }
        /* Okay to end the simulation */
        primaryComponentOKToEndSim();
        return true;
    }
    else{

        /* Load the input packet to outputPort (outputQueue) (handleLoad) */
        if (numSent < numSample){
            handleLoad();
        }

        /* Pop the output event from the outputPort (outputQueue) and send it through the outputlink */
        for (uint32_t p = 0; p < numPort; ++p){
            auto outEv = outputPort[p]->getEvent(cycle);
            if (outEv){
                outputLink[p]->send(outEv);
                outStd.verbose(CALL_INFO, 2, 0, "Out (Port#%u) -> ID#%d (size %ld)[0] = %.3lf\n", p, outEv->getIndex1(), outEv->getDataDouble().size(), outEv->getDataDouble()[0]);
                outFile.verbose(CALL_INFO, 2, 0, "Out (Port#%u) -> ID#%d (size %ld)[0] = %.3lf\n", p, outEv->getIndex1(), outEv->getDataDouble().size(), outEv->getDataDouble()[0]);
                // latencySample[outEv->getIndex1()] = getCurrentSimTime();
            }
        }

        /* Pop the input event from the inputPort (inputQueue) and perfrom the classification/regression/SHAP (handleSelf) */
        if (!busy){
            bool anyIn = false;
            std::vector<std::vector<double>> dataPorts;
            std::vector<std::vector<int32_t>> indexPorts;
            for (uint32_t i = 0; i < numPort; ++i){
                auto inEv = inputPort[i]->getEvent(cycle);
                if (inEv){
                    busy = true;
                    anyIn = true;
                    dataPorts.push_back(inEv->getDataDouble());

                    std::vector<int32_t> tempIndex;
                    tempIndex.push_back(inEv->getIndex1());
                    tempIndex.push_back(inEv->getIndex2());
                    tempIndex.push_back(inEv->getIndex3());
                    indexPorts.push_back(tempIndex);
                    outStd.verbose(CALL_INFO, 2, 0, "In (Port#%u) -> ID#%d, Class#%d, SumTree#%d [0] = %.3lf\n", i, inEv->getIndex1(), inEv->getIndex2(), inEv->getIndex3(), inEv->getDataDouble()[0]);
                    outFile.verbose(CALL_INFO, 2, 0, "In (Port#%u) -> ID#%d, Class#%d, SumTree#%d [0] = %.3lf\n", i, inEv->getIndex1(), inEv->getIndex2(), inEv->getIndex3(), inEv->getDataDouble()[0]);
                }
            }
            if (anyIn){
                DataMatrixEvent *inEvs = new DataMatrixEvent(dataPorts, indexPorts, 0, 0, 0);
                // selfLink->send(selfLatency*(dataPorts.size())-1, inEvs);
                selfLink->send(0, inEvs);
            }
        }
    }  
    return false; 
}

