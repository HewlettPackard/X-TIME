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


#include "mmr.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for mmr 
* @details Read parameters, configure output, register clock handler, and configure links.
*/
mmr::mmr(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);

    numPort                 = params.find<uint32_t>("numPort", 1);
    numMatch                = params.find<int32_t>("numMatch", -1);

    /* Power parameters */
    powerMmr                 = params.find<double>("power.mmr", -1);
    powerMMR = registerStatistic<double>("powerMMR");

    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Mmr" + std::to_string(compId) + "    ]:          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"mmr"+std::to_string(compId)+".txt");
    
    /* Register clock handler */ 
    clockHandler = new Clock::Handler<mmr>(this, &mmr::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    outputLink = configureLink("output_port");
    selfLink = configureSelfLink("self", freq, new Event::Handler<mmr>(this, &mmr::handleSelf));
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

}

/**
 * @brief Convert the match vector to memory address.
 */
void
mmr::handleSelf(Event *ev){
    DataEvent *inEv = static_cast<DataEvent*>(ev);
    std::vector<int32_t> data = inEv->getDataInt();
    std::vector<int32_t>::iterator it = std::find(data.begin(), data.end(), 1);
    
    /* Unknown number of matches */
    if (numMatch == -1){
        while (it != data.end()){
            int32_t idx = it-data.begin();
            std::vector<int32_t> address(1, idx);
            DataEvent *addressEv = new DataEvent(address, inEv->getIndex1(), inEv->getIndex2(), inEv->getIndex3());
            outputQueue.push(getNextClockCycle(clockPeriod) - 1, addressEv);
            data[idx] = false;
            it = std::find(data.begin(), data.end(), 1);
            powerMMR->addData(data.size()*powerMmr);
        }
    }
    /* Predefined number of matches */
    else{
        for (uint32_t i = 0; i < numMatch; ++i){
            int32_t idx = it-data.begin();
            std::vector<int32_t> address(1, idx);
            DataEvent *addressEv = new DataEvent(address, inEv->getIndex1(), inEv->getIndex2(), inEv->getIndex3());
            outputQueue.push(getNextClockCycle(clockPeriod) - 1, addressEv);
            data[idx] = false;
            it = std::find(data.begin(), data.end(), 1);
            powerMMR->addData(data.size()*powerMmr);
        }
    }
    busy = false;
    delete inEv;
}

/**
 * @brief Load the incoming input event to its portQueue.
 */
void
mmr::Port::handleNewData(Event *ev) {
    portQueue.push(m_mmr->getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the output event from the outputPort (outputQueue) and send it through the outputlink
 * 2. Pop the input event from the inputPort (portQueue) and load the event to the outputQueue (handleSelf)
 */
bool
mmr::clockTick(Cycle_t cycle) {
    /* Pop the output event from the outputPort (outputQueue) and send it through the outputlink */
    auto outEv = outputQueue.pop(cycle);
    if (outEv){
        outputLink->send(outEv);
        outStd.verbose(CALL_INFO, 4, 0, "Out -> ID#%d,%d [0] = %u\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataInt()[0]);
        outFile.verbose(CALL_INFO, 5, 0, "Out -> ID#%d,%d [0] = %u\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataInt()[0]);
    }

    /* Pop the input event from the inputPort (portQueue) and load the event to the outputQueue (handleSelf) */
    if (!busy){
        bool anyIn = false;
        std::vector<int32_t> dataPorts;
        std::vector<int32_t> indexPorts;
        
        for (uint32_t i = 0; i < numPort; ++i){
            auto inEv = inputPort[i]->getEvent(cycle);
            if (inEv){
                busy = true;
                anyIn = true;
                auto tempData = inEv->getDataInt();
                dataPorts.insert(dataPorts.end(), tempData.begin(), tempData.end());

                indexPorts.push_back(inEv->getIndex1());
                indexPorts.push_back(inEv->getIndex2());
                indexPorts.push_back(inEv->getIndex3());
                outStd.verbose(CALL_INFO, 4, 0, "In -> ID#%d,%d [0] = %d\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataInt()[0]);
                outFile.verbose(CALL_INFO, 5, 0, "In -> ID#%d,%d [0] = %d\n", inEv->getIndex1(), inEv->getIndex2(), inEv->getDataInt()[0]);
            }
        }
        if (anyIn){
            DataEvent *inEvs = new DataEvent(dataPorts, indexPorts[0], indexPorts[1], indexPorts[2]);
            selfLink->send(0, inEvs);
        }
    }
    return false; 
}

