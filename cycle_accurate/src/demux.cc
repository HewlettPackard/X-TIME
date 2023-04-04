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


#include "demux.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for demux 
* @details Read parameters, configure output, register clock handler, and configure links.
*/
demux::demux(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);
    
    level                   = params.find<uint32_t>("level", 0);
    numPort                 = params.find<uint32_t>("numPort", 4);
    params.find_array("mode", mode);

    /* Power parameters */
    uint32_t sizeData       = params.find<uint32_t>("power.sizeData", 32);
    powerReg                = params.find<double>("power.reg", -1);
    powerDemux              = params.find<double>("power.demux", -1);
    powerDemux = (powerDemux+powerReg*(numPort+1))*sizeData;
    powerDEMUX = registerStatistic<double>("powerDEMUX");

    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Demux" + std::to_string(compId) + "  ]:          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"demux"+std::to_string(compId)+".txt");

    /* Register clock handler */ 
    clockHandler = new Clock::Handler<demux>(this, &demux::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    auto outputStat = registerStatistic<uint64_t>("oq_size");
    for (uint32_t i = 0; i < numPort; ++i){
        outputPort.push_back(new Port(i, clockPeriod, outStd));
        outputPort[i]->getQueue().setLatency(oqLat);
        outputPort[i]->getQueue().setQueueSizeStat(outputStat);
        outputLink.push_back(configureLink("output_port"+std::to_string(i)));
    }
    auto inputStat = registerStatistic<uint64_t>("iq_size");
    inputQueue.setLatency(iqLat);
    inputQueue.setQueueSizeStat(inputStat);
    inputLink = configureLink("input_port", new Event::Handler<demux>(this, &demux::handleNewData));
    selfLink = configureSelfLink("self", freq, new Event::Handler<demux>(this, &demux::handleSelf));
    
}

/**
 * @brief According to mode, load the input event to the outputPort (outputQueue).
 */
void
demux::handleSelf(Event *ev){
    DataEvent *inEv = static_cast<DataEvent*>(ev);
    uint32_t portID = 0;
    for (uint32_t i = 0; i < mode.size(); ++i){
        if (inEv->getDataInt()[level+1] == i){
            for (uint32_t j = 0; j < mode[i]; ++j){
                DataEvent *ev = new DataEvent(inEv->getDataDouble(), inEv->getDataInt(), inEv->getIndex1(), inEv->getIndex2(), inEv->getIndex3());
                outputPort[portID+j]->getQueue().push_relative(getNextClockCycle(clockPeriod) - 1, ev);               
            }
        }
        portID += mode[i];
    }
    powerDEMUX->addData(powerDemux);
    busy = false;
    delete inEv;
}

/**
 * @brief Load the incoming event to its inputQueue.
 */
void
demux::handleNewData(Event *ev) {
    inputQueue.push(getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the output event from the outputPort (outputQueue) and send it through the outputlink
 * 2. Pop the input event from the inputPort (inputQueue) and load the event to the outputPort according to mode (handleSelf)
 */
bool
demux::clockTick(Cycle_t cycle) {
    /* Pop the output event from the outputPort (outputQueue) and send it through the outputlink */
    for (uint32_t i =0; i < numPort; ++i){
        auto outEv = outputPort[i]->getEvent(cycle);
        if (outEv){
            outputLink[i]->send(outEv);
            outStd.verbose(CALL_INFO, 3, 0, "Out (Port#%u) -> ID#%d,%d [0] = %.3lf\n", i, outEv->getIndex1(), outEv->getIndex2(), outEv->getDataDouble()[0]);
            outFile.verbose(CALL_INFO, 3, 0, "Out (Port#%u) -> ID#%d,%d [0] = %.3lf\n", i, outEv->getIndex1(), outEv->getIndex2(), outEv->getDataDouble()[0]);
        }
    }

    /* Pop the input event from the inputPort (inputQueue) and load the event to the outputPort according to mode (handleSelf) */
    if (!busy){
        auto inEv = inputQueue.pop(cycle);
        if (inEv){
            busy = true;
            selfLink->send(latency - 1, inEv);
            outStd.verbose(CALL_INFO, 3, 0, "In -> ID#%d,%d [0] = %.3lf\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataDouble()[0]);
            outFile.verbose(CALL_INFO, 3, 0, "In -> ID#%d,%d [0] = %.3lf\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataDouble()[0]);
        }
    }
    return false; 
}
