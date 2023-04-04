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


#include "mpe.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for mpe 
* @details Read parameters, initialize variables, configure output, register clock handler, and configure links.
*/
mpe::mpe(ComponentId_t id, Params &params) : Component(id) {

    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);

    numPort                 = params.find<uint32_t>("numPort", 1);
    numFeature              = params.find<uint32_t>("numFeature", 10);
    acamRow                 = params.find<uint32_t>("acamRow", 256);
    classID                 = params.find<int32_t>("classID", -1);
    uint32_t maxDepth       = params.find<uint32_t>("maxDepth", 10);

    params.find_array("logit", v);

    std::string weightFile   = params.find<std::string>("weightFile");
    std::ifstream weightStream (weightFile+std::to_string(maxDepth)+"x"+std::to_string(maxDepth)+".txt");
    weightMatrix.resize(maxDepth);
    for (uint32_t i = 0; i < maxDepth; ++i){
        weightMatrix[i].resize(maxDepth);
        for (uint32_t j = 0; j < maxDepth; ++j){
            weightStream >> weightMatrix[i][j];
        }
    }

    /* Initialize variables */
    n.resize(numPort*acamRow, 0);
    s.resize(numPort*acamRow, 0);
    p.resize(numPort*acamRow, 1);
    u.assign(numPort*acamRow, std::vector<int32_t> (numFeature, 0));
    c.assign(numPort*acamRow, std::vector<int32_t> (numFeature, 0));
    shapley.resize(numFeature, 0.0);

    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Mpe" + std::to_string(compId) + "    ]:          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"mpe"+std::to_string(compId)+".txt");
    
    /* Register clock handler */ 
    clockHandler = new Clock::Handler<mpe>(this, &mpe::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    outputLink = configureLink("output_port");
    selfLink = configureSelfLink("self", freq, new Event::Handler<mpe>(this, &mpe::handleSelf));
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
 * @brief Compute SHAP value given foreground/background sample pair.
 */
void
mpe::handleSelf(Event *ev){
    DataEvent *inEv = static_cast<DataEvent*>(ev);

    /* Get a pair of match vectors (foreground, background). Update n, s, p, u, c #numFeature times. Compute SHAP value. */
    if (inEv->getIndex1() != -1){
        if (inEv->getIndex3() == 1){
            foreground = inEv->getDataInt();
        }
        else if (inEv->getIndex3() == -1){
            background = inEv->getDataInt();
            for (uint32_t l = 0; l < numPort*acamRow; ++l){
                n[l] += (foreground[l] ^ background[l]);
                s[l] += (foreground[l] & ~background[l]);
                p[l] *= (foreground[l] | background[l]);
                u[l][numPair] = foreground[l] & ~background[l];
                c[l][numPair] = (foreground[l] & ~background[l]) - (~foreground[l] & background[l]);
            }
            ++numPair;
            outStd.verbose(CALL_INFO, 4, 0, "Fore#%u, Feature#%u \n", inEv->getIndex1(), numPair);
        }
        
        if (numPair == numFeature){
            for (uint32_t i = 0; i < numPort*acamRow; ++i){
                if (p[i]!=0){
                    for (uint32_t j = 0; j < numFeature; ++j){
                        shapley[j] += (weightMatrix[n[i]][s[i]-u[i][j]])*v[i]*c[i][j];
                    }
                }
            }
            DataEvent *outEv = new DataEvent(shapley, inEv->getIndex1(), classID, 1);
            outputQueue.push(getNextClockCycle(clockPeriod) - 1, outEv);
            n.assign(numPort*acamRow, 0);
            s.assign(numPort*acamRow, 0);
            p.assign(numPort*acamRow, 1);
            u.assign(numPort*acamRow, std::vector<int32_t> (numFeature, 0));
            c.assign(numPort*acamRow, std::vector<int32_t> (numFeature, 0));
            shapley.assign(numFeature, 0.0);
            foreground.clear();
            background.clear();
            numPair = 0;
        }
    }
    /* An event with -1 of sampleID is the end signal. So, empty event flows.*/
    else{
        DataEvent *outEv = new DataEvent(std::vector<double> (1, 0), -1, classID, 1);
        outputQueue.push(getNextClockCycle(clockPeriod) - 1, outEv);
    }
    busy = false;
    delete inEv;
}

/**
 * @brief Load the incoming event to its portQueue.
 */
void
mpe::Port::handleNewData(Event *ev) {
    portQueue.push(m_mpe->getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the output event (SHAP value) from the outputQueue and send it to accumulator through the outputLink
 * 2. Pop the input event from the inputPort (portQueue) and load the event to the outputQueue (handleSelf)
 */
bool
mpe::clockTick(Cycle_t cycle) {
    /* Pop the output event (SHAP value) from the outputQueue and send it to accumulator through the outputLink */
    auto outEv = outputQueue.pop(cycle);
    if (outEv){
        outputLink->send(outEv);
        outStd.verbose(CALL_INFO, 4, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataDouble()[0]);
        outFile.verbose(CALL_INFO, 5, 0, "Out -> ID#%d,%d [0] = %.3lf\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataDouble()[0]);
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

