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


#include "driver.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for driver 
* @details Read parameters, configure output, register clock handler, and configure links.
*/
driver::driver(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);
    
    task                    = params.find<std::string>("task", "classification");
    acamQueue               = params.find<uint32_t>("acamQueue", 1);
    acamStack               = params.find<uint32_t>("acamStack", 1);
    acamCol                 = params.find<uint32_t>("acamCol", 32);
    acamRow                 = params.find<uint32_t>("acamRow", 256);

    /* Power parameters */
    powerReg                = params.find<double>("power.reg", -1);
    powerDriver = powerReg*(acamQueue*acamCol);
    powerDRIVER = registerStatistic<double>("powerDRIVER");
    
    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Driver" + std::to_string(compId) + " ]:          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"driver"+std::to_string(compId)+".txt");
    
    /* Register clock handler */ 
    clockHandler = new Clock::Handler<driver>(this, &driver::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    std::string portName;
    for (uint32_t s = 0; s < acamStack; ++s){
        portName = "en_port"+std::to_string(s);
        enLinks.push_back(configureLink(portName));
    }
    for (uint32_t q = 0; q < acamQueue; ++q){
        for (uint32_t s = 0; s < acamStack; ++s){
            portName = "dl_port"+std::to_string(q*acamStack+s);
            dlLinks.push_back(configureLink(portName));
        }
    }
    inputLink = configureLink("input_port", new Event::Handler<driver>(this, &driver::handleNewData));
    selfLink = configureSelfLink("self", freq, new Event::Handler<driver>(this, &driver::handleSelf));

    auto inputStat = registerStatistic<uint64_t>("iq_size");
    inputQueue.setLatency(iqLat);
    inputQueue.setQueueSizeStat(inputStat);
    auto dlStat = registerStatistic<uint64_t>("dlq_size");
    dlQueue.setLatency(oqLat);
    dlQueue.setQueueSizeStat(dlStat);
    auto enStat = registerStatistic<uint64_t>("enq_size");
    enQueue.setLatency(oqLat);
    enQueue.setQueueSizeStat(enStat);

}

/**
 * @brief Load the input event to the outputPort (outputQueue).
 */
void
driver::handleSelf(Event *ev){
    /* Input event: std::vector<double> data, std::vector<int32_t> routingInfo, int32_t sampleID, int32_t -1, int32_t 0 */
    DataEvent *inEv = static_cast<DataEvent*>(ev);
    if (task == "classification" || task == "regression"){
        /* Data and sampleID */
        std::vector<double> foreground = inEv->getDataDouble();
        uint32_t foreIndex = inEv->getIndex1();

        // TODO: Map trees columnwise - time-multiplexing
        /* Zero-padding data */
        uint32_t pad = acamQueue*acamCol - foreground.size();
        std::vector<int32_t> foreX(foreground.size(), 0);
        for (uint32_t i = 0; i < pad ; ++i){
            foreground.push_back(0.0);
            foreX.push_back(1);
        }

        /* Load the bunch of DataEvent (DataMatrixEvent) to dlQueue/enQueue */
        std::vector<std::vector<double>> dlFore;
        std::vector<std::vector<int32_t>> dlX;
        for (uint32_t q = 0; q < acamQueue; ++q){
            dlFore.push_back(std::vector<double>(foreground.begin() + q*acamCol, foreground.begin() + (q+1)*acamCol));
            dlX.push_back(std::vector<int32_t>(foreX.begin() + q*acamCol, foreX.begin() + (q+1)*acamCol));
        }
        DataMatrixEvent *foreEv = new DataMatrixEvent(dlFore, dlX, foreIndex, -1, -1);
        dlQueue.push(getNextClockCycle(clockPeriod) - 1, foreEv);
        std::vector<std::vector<int32_t>> en(acamStack, std::vector<int32_t> (acamRow, 1));
        DataMatrixEvent *enFEv = new DataMatrixEvent(en, foreIndex, -1, -1);
        enQueue.push(getNextClockCycle(clockPeriod) - 1, enFEv);

        powerDRIVER->addData(powerDriver + powerReg*foreground.size());
    }
    else if (task == "SHAP"){
        /* Generate 2*#numFeature events per dl/enQueue.
        2: foreground sample, background sample
        #numFeature: extract a single feature value from sample at a time, and others are 'don't care'. 
        */
        if (inEv->getIndex1() != -1){
            uint32_t numFeature = inEv->getDataDouble().size()/2;
            uint32_t foreIndex = inEv->getIndex1();
            std::vector<double> foreground, background;
            for (uint32_t i = 0; i < numFeature; ++i){
                foreground.push_back(inEv->getDataDouble()[i]);
                background.push_back(inEv->getDataDouble()[i+numFeature]);
            }
            
            uint32_t pad = acamQueue*acamCol - foreground.size();
            for (uint32_t i = 0; i < pad ; ++i){
                foreground.push_back(0.0);
                background.push_back(0.0);
            }

            for (uint32_t i = 0; i < numFeature; ++i){
                std::vector<std::vector<double>> dlFore;
                std::vector<std::vector<double>> dlBack;
                std::vector<std::vector<int32_t>> dlX;
                std::vector<int32_t> foreX(acamQueue*acamCol, 1);
                foreX[i] = 0;
                for (uint32_t q = 0; q < acamQueue; ++q){
                    dlFore.push_back(std::vector<double>(foreground.begin() + q*acamCol, foreground.begin() + (q+1)*acamCol));
                    dlBack.push_back(std::vector<double>(background.begin() + q*acamCol, background.begin() + (q+1)*acamCol));
                    dlX.push_back(std::vector<int32_t>(foreX.begin() + q*acamCol, foreX.begin() + (q+1)*acamCol));
                }
                DataMatrixEvent *foreEv = new DataMatrixEvent(dlFore, dlX, foreIndex, -1, 1);
                DataMatrixEvent *backEv = new DataMatrixEvent(dlBack, dlX, foreIndex, -1, -1);
                dlQueue.push_relative(getNextClockCycle(clockPeriod) - 1, foreEv);
                dlQueue.push_relative(getNextClockCycle(clockPeriod) - 1, backEv);

                std::vector<std::vector<int32_t>> en(acamStack, std::vector<int32_t> (acamRow, 1));
                DataMatrixEvent *enFEv = new DataMatrixEvent(en, foreIndex, -1, 1);
                DataMatrixEvent *enBEv = new DataMatrixEvent(en, foreIndex, -1, -1);
                enQueue.push_relative(getNextClockCycle(clockPeriod) - 1, enFEv);
                enQueue.push_relative(getNextClockCycle(clockPeriod) - 1, enBEv);

                powerDRIVER->addData(powerDriver + powerReg*foreground.size());
            }
        }
        /* An event with -1 of sampleID is the end signal. So, empty event flows.*/
        else{
            std::vector<std::vector<double>> dlFore;
            std::vector<std::vector<int32_t>> dlX;
            for (uint32_t q = 0; q < acamQueue; ++q){
                dlFore.push_back(std::vector<double>(acamCol, 0));
                dlX.push_back(std::vector<int32_t>(acamCol, 1));
            }
            DataMatrixEvent *foreEv = new DataMatrixEvent(dlFore, dlX, -1, -1, -1);
            dlQueue.push(getNextClockCycle(clockPeriod) - 1, foreEv);
            std::vector<std::vector<int32_t>> en(acamStack, std::vector<int32_t> (acamRow, 1));
            DataMatrixEvent *enFEv = new DataMatrixEvent(en, -1, -1, -1);
            enQueue.push(getNextClockCycle(clockPeriod) - 1, enFEv);
        }
    }
    busy = false;
    delete inEv;
}

/**
 * @brief Load the incoming event to its inputQueue.
 */
void
driver::handleNewData(Event *ev) {
    inputQueue.push(getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the dl events from the dlQueue and send them to all aCAMs through the dlLinks
 * 2. Pop the en events from the enQueue and send them to the first column of aCAMs through the enLinks
 * 3. Pop the input event from the inputPort (inputQueue) and load the event to the dlQueue/enQueue (handleSelf)
 */
bool
driver::clockTick(Cycle_t cycle) {
    /* Pop the dl events from the dlQueue and send them to all aCAMs through the dlLinks */
    auto dlEv = dlQueue.pop(cycle);
    if (dlEv){
        std::vector<std::vector<double>> dl = dlEv->getDataDouble();
        std::vector<std::vector<int32_t>> dlX = dlEv->getDataInt();
        for (uint32_t q = 0; q < acamQueue; ++q){
            for (uint32_t s = 0; s < acamStack; ++s){
                DataEvent *dataEv = new DataEvent(dl[q], dlX[q], dlEv->getIndex1(), dlEv->getIndex2(), dlEv->getIndex3());
                dlLinks[q*acamStack+s]->send(dataEv);
                outStd.verbose(CALL_INFO, 4, 0, "DL (s%u, q%u) -> ID#%d,%d [0] = %.3lf\n", s, q, dataEv->getIndex1(), dataEv->getIndex2(), dataEv->getDataDouble()[0]);
                outFile.verbose(CALL_INFO, 5, 0, "DL (s%u, q%u) -> ID#%d,%d [0] = %.3lf\n", s, q, dataEv->getIndex1(), dataEv->getIndex2(), dataEv->getDataDouble()[0]);
            }
        }
    }

    /* Pop the en events from the enQueue and send them to the first column of aCAMs through the enLinks */
    auto enEv = enQueue.pop(cycle);
    if (enEv){
        std::vector<std::vector<int32_t>> en = enEv->getDataInt();
        for (uint32_t s = 0; s < acamStack; ++s){
            DataEvent *dataEv = new DataEvent(en[s], enEv->getIndex1(), enEv->getIndex2(), enEv->getIndex3());
            enLinks[s]->send(dataEv);
            outStd.verbose(CALL_INFO, 4, 0, "EN (s%u, q%u) -> ID#%d,%d [0] = %u\n", s, 0, dataEv->getIndex1(),  dataEv->getIndex2(), dataEv->getDataInt()[0]);
            outFile.verbose(CALL_INFO, 5, 0, "EN (s%u, q%u) -> ID#%d,%d [0] = %u\n", s, 0, dataEv->getIndex1(),  dataEv->getIndex2(), dataEv->getDataInt()[0]);
        }
    }

    /* Pop the input event from the inputPort (inputQueue) and load the event to the dlQueue/enQueue (handleSelf) */
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
