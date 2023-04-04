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


#include "adder.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for adder 
* @details Read parameters, configure output, register clock handler, and configure links.
*/
adder::adder(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);

    numMatch                = params.find<uint32_t>("numMatch", 1);

    /* Power parameters */
    powerAdder              = params.find<double>("power.adder", -1);
    powerAdder = powerAdder*32;
    powerADDER = registerStatistic<double>("powerADDER");

    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Adder" + std::to_string(compId) + "  ] :          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"adder"+std::to_string(compId)+".txt");

    /* Register clock handler */ 
    clockHandler = new Clock::Handler<adder>(this, &adder::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    inputLink = configureLink("input_port", new Event::Handler<adder>(this, &adder::handleNewData));
    outputLink = configureLink("output_port");
    selfLink = configureSelfLink("self", freq, new Event::Handler<adder>(this, &adder::handleSelf));

    auto inputStat = registerStatistic<uint64_t>("iq_size");
    inputQueue.setLatency(iqLat);
    inputQueue.setQueueSizeStat(inputStat);
    auto outputStat = registerStatistic<uint64_t>("oq_size");
    outputQueue.setLatency(oqLat);
    outputQueue.setQueueSizeStat(outputStat);

}

/**
 * @brief Accumulate #numMatch results sequentially and send the accumulated results at once.
 */
void
adder::handleSelf(Event *ev){
    DataEvent *inEv = static_cast<DataEvent*>(ev);
    if (numAdded == 0) {
        tempData = inEv->getDataDouble();
        numAdded = 1;
    }
    else{
        tempData[0] += inEv->getDataDouble()[0];
        ++numAdded;
    }
    if (numAdded == numMatch){
        DataEvent *outEv = new DataEvent(tempData, inEv->getIndex1(), inEv->getIndex2(), numMatch);
        outputQueue.push_relative(getNextClockCycle(clockPeriod) - 1, outEv);
        numAdded = 0;
    }
    busy = false;
    delete inEv;
}

/**
 * @brief Load the incoming event to its inputQueue.
 */
void
adder::handleNewData(Event *ev) {
    inputQueue.push(getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the output event from the outputPort (outputQueue) and send it through the outputlink
 * 2. Pop the input event from the inputPort (inputQueue) and load the event to the outputPort according to mode (handleSelf)
 */
bool
adder::clockTick(Cycle_t cycle) {
    /* Pop the output event from the outputPort (outputQueue) and send it through the outputlink */
    auto outEv = outputQueue.pop(cycle);
    if (outEv){
        outputLink->send(outEv);
        outStd.verbose(CALL_INFO, 4, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex3(), outEv->getDataDouble()[0]);
        outFile.verbose(CALL_INFO, 5, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex3(), outEv->getDataDouble()[0]);
    }

    /* Pop the input event from the inputPort (inputQueue) and load the event to the outputPort according to mode (handleSelf) */
    if (!busy){
        auto inEv = inputQueue.pop(cycle);
        if (inEv){
            busy = true;
            selfLink->send(latency - 1, inEv);
            outStd.verbose(CALL_INFO, 4, 0, "In -> ID#%d,%d [0] = %.3lf\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataDouble()[0]);
            outFile.verbose(CALL_INFO, 5, 0, "In -> ID#%d,%d [0] = %.3lf\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataDouble()[0]);
        }
    }
    return false; 

}