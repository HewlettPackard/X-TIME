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


#include "memory.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for memory 
* @details Read parameters, program memory, configure output, register clock handler, and configure links.
*/
memory::memory(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);

    classID                 = params.find<int32_t>("classID", -1);
    
    /* Program memory */
    params.find_array("logit", logit);

    /* Power parameters */
    powerMemory                = params.find<double>("power.memory", -1);
    powerMemory = powerMemory*logit.size()*32;
    powerMEMORY = registerStatistic<double>("powerMEMORY");

    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Memory" + std::to_string(compId) + " ] :          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"memory"+std::to_string(compId)+".txt");

    /* Register clock handler */
    clockHandler = new Clock::Handler<memory>(this, &memory::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    inputLink = configureLink("input_port", new Event::Handler<memory>(this, &memory::handleNewData));
    outputLink = configureLink("output_port");
    selfLink = configureSelfLink("self", freq, new Event::Handler<memory>(this, &memory::handleSelf));

    auto inputStat = registerStatistic<uint64_t>("iq_size");
    inputQueue.setLatency(iqLat);
    inputQueue.setQueueSizeStat(inputStat);
    auto outputStat = registerStatistic<uint64_t>("oq_size");
    outputQueue.setLatency(oqLat);
    outputQueue.setQueueSizeStat(outputStat);

}

/**
 * @brief Read the memory data according to the input address
 */
void
memory::handleSelf(Event *ev){
    DataEvent *inEv = static_cast<DataEvent*>(ev);
    int32_t index = inEv->getDataInt()[0];
    /* -1 of classID means the memory is not programmed. So, no need to generate the output event. */
    if (classID != -1){
        if (index < logit.size()){
            std::vector<double> data(1, logit[inEv->getDataInt()[0]]);
            DataEvent *dataEv = new DataEvent(data, inEv->getIndex1(), classID, 1);
            outputQueue.push(getNextClockCycle(clockPeriod) - 1, dataEv);
        }
        else{
            std::vector<double> data(1, 0);
            DataEvent *dataEv = new DataEvent(data, inEv->getIndex1(), classID, 1);
            outputQueue.push(getNextClockCycle(clockPeriod) - 1, dataEv);
        }
        powerMEMORY->addData(powerMemory);
    }
    busy = false;
    delete inEv;
}

/**
 * @brief Load the incoming input event to inputQueue.
 */
void
memory::handleNewData(Event *ev) {
    inputQueue.push(getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the output event from the outputPort (outputQueue) and send it through the outputlink
 * 2. Pop the input event from the inputPort (inputQueue) and load the event to the outputQueue (handleSelf)
 */
bool
memory::clockTick(Cycle_t cycle) {
    /* Pop the output event from the outputPort (outputQueue) and send it through the outputlink */
    auto outEv = outputQueue.pop(cycle);
    if (outEv){
        outputLink->send(outEv);
        outStd.verbose(CALL_INFO, 4, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataDouble()[0]);
        outFile.verbose(CALL_INFO, 5, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataDouble()[0]);
    }

    /* Pop the input event from the inputPort (inputQueue) and load the event to the outputQueue (handleSelf) */
    if (!busy){
        auto inEv = inputQueue.pop(cycle);
        if (inEv){
            busy = true;
            selfLink->send(latency - 1, inEv);
            outStd.verbose(CALL_INFO, 4, 0, "In -> ID#%d,%d [0] = %u\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataInt()[0]);
            outFile.verbose(CALL_INFO, 5, 0, "In -> ID#%d,%d [0] = %u\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataInt()[0]);
        }
    }
    return false; 
}

