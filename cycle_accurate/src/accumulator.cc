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


#include "accumulator.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for accumulator 
* @details Read parameters, configure output, register clock handler, and configure links.
*/
accumulator::accumulator(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);

    numPort                 = params.find<uint32_t>("numPort", 4);
    mode                    = params.find<uint32_t>("mode", 0);

    /* Power parameters */
    uint32_t sizeData       = params.find<uint32_t>("power.sizeData", 32);
    powerReg                = params.find<double>("power.reg", -1);
    powerAccum              = params.find<double>("power.accumulator", -1);
    powerAccum = (powerAccum+powerReg*(numPort+1))*sizeData;
    powerACCUM = registerStatistic<double>("powerACCUM");

    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Accumul" + std::to_string(compId) + "]:          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"accumulator"+std::to_string(compId)+".txt");

    /* Register clock handler */ 
    clockHandler = new Clock::Handler<accumulator>(this, &accumulator::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    auto inputStat = registerStatistic<uint64_t>("iq_size");
    for (uint32_t i = 0; i < numPort; ++i){
        inputPort.push_back(new Port(i, clockPeriod, outStd, this));
        inputPort[i]->getQueue().setLatency(iqLat);
        inputPort[i]->getQueue().setQueueSizeStat(inputStat);
        inputLink.push_back(configureLink("input_port"+std::to_string(i), new Event::Handler<Port>(inputPort.back(), &Port::handleNewData)));
    }
    auto outputStat = registerStatistic<uint64_t>("oq_size");
    outputQueue.setLatency(oqLat);
    outputQueue.setQueueSizeStat(outputStat);
    outputLink = configureLink("output_port");
    selfLink = configureSelfLink("self", freq, new Event::Handler<accumulator>(this, &accumulator::handleSelf));

}

/**
 * @brief Accumulate the results when they have same smapleID and classID.
 */
void
accumulator::handleSelf(Event *ev){
    DataMatrixEvent *inEv = static_cast<DataMatrixEvent*>(ev);
    
    uint32_t inEvSize = inEv->getDataDouble().size();
    /* Accumulate the results when they have same sampleID and classID. */ 
    if (mode == 0){
        std::vector<std::vector<double>> tempData;
        std::vector<std::vector<int32_t>> tempIndex;
        tempData.resize(inEvSize);
        tempIndex.resize(inEvSize);

        tempData[0] = inEv->getDataDouble()[0];
        tempIndex[0] = inEv->getDataInt()[0];

        for (uint32_t i = 1; i < inEvSize; ++i){
            for (uint32_t j = 0; j < inEvSize; ++j){
                if (tempData[j].size() != 0){
                    // Check SampleID, ClassID 
                    if ((tempIndex[j][0] == inEv->getDataInt()[i][0]) && (tempIndex[j][1] == inEv->getDataInt()[i][1])){
                        for (uint32_t k = 0; k < inEv->getDataDouble()[i].size(); ++k){
                            tempData[j][k] += inEv->getDataDouble()[i][k];
                        }
                        tempIndex[j][2] += inEv->getDataInt()[i][2];
                        break;
                    }
                }
                else{
                    tempData[j] = inEv->getDataDouble()[i];
                    tempIndex[j] = inEv->getDataInt()[i];
                    break;
                }
            }
        }

        for (uint32_t i = 0; i < inEvSize; ++i){
            if (tempData[i].size() != 0){
                DataEvent *outEv = new DataEvent(tempData[i], tempIndex[i][0], tempIndex[i][1], tempIndex[i][2]);
                outputQueue.push_relative(getNextClockCycle(clockPeriod) - 1, outEv);
            }
        }
    }
    /* No accumulate, but serialize the parallel input packets. */ 
    else if (mode == 1){
        for (uint32_t i = 0; i < inEvSize; ++i){
            DataEvent *outEv = new DataEvent(inEv->getDataDouble()[i], inEv->getDataInt()[i][0], inEv->getDataInt()[i][1], inEv->getDataInt()[i][2]);
            outputQueue.push_relative(getNextClockCycle(clockPeriod) - 1, outEv);
        }
    }
    powerACCUM->addData(powerAccum);
    busy = false;
    delete inEv;
}

/**
 * @brief Load the incoming event to its portQueue.
 */
void
accumulator::Port::handleNewData(Event *ev) {
    portQueue.push(m_accumulator->getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the output event from the outputPort (outputQueue) and send it through the outputlink
 * 2. Pop the input event from the inputPort (portQueue) and load the event to the outputPort according to mode (handleSelf)
 */
bool
accumulator::clockTick(Cycle_t cycle) {
    /* Pop the output event from the outputPort (outputQueue) and send it through the outputlink */
    auto outEv = outputQueue.pop(cycle);
    if (outEv){
        outputLink->send(outEv);
        outStd.verbose(CALL_INFO, 3, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex3(), outEv->getDataDouble()[0]);
        outFile.verbose(CALL_INFO, 5, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex3(), outEv->getDataDouble()[0]);
    }

    /* Pop the input event from the inputPort (portQueue) and load the event to the outputPort according to mode (handleSelf) */
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
                outStd.verbose(CALL_INFO, 3, 0, "In (Port#%u) -> ID#%d,%d [0] = %.3lf\n", i, inEv->getIndex1(), inEv->getIndex3(), inEv->getDataDouble()[0]);
                outFile.verbose(CALL_INFO, 5, 0, "In (Port#%u) -> ID#%d,%d [0] = %.3lf\n", i, inEv->getIndex1(), inEv->getIndex3(), inEv->getDataDouble()[0]);
            }
        }
        if (anyIn){
            DataMatrixEvent *inEvs = new DataMatrixEvent(dataPorts, indexPorts, 0, 0, 0);
            selfLink->send(latency*(numPort)-1, inEvs);
        }
    }
    return false; 
}
