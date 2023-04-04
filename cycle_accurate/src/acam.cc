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

#include "acam.h"

using namespace SST;
using namespace SST::XTIME;

/**
* @brief Main constructor for acam 
* @details Read parameters, program aCAM, configure output, register clock handler, and configure links.
*/
acam::acam(ComponentId_t id, Params &params) : Component(id) {
    
    /* Read parameters */
    uint32_t compId         = params.find<uint32_t>("id", 0);
    uint32_t verbose        = params.find<uint32_t>("verbose", 0);
    uint32_t iqLat          = params.find<uint32_t>("inputQueueLatency", 1);
    uint32_t oqLat          = params.find<uint32_t>("outputQueueLatency", 1);
    std::string freq        = params.find<std::string>("freq", "1GHz");
    latency                 = params.find<uint32_t>("latency", 1);

    acamCol                 = params.find<uint32_t>("acamCol", 32);
    acamRow                 = params.find<uint32_t>("acamRow", 256);

    /* Power parameters*/
    double Cml              = params.find<double>("power.Cml", -1);
    double Cpre             = params.find<double>("power.Cpre", -1);
    double Cmlso            = params.find<double>("power.Cmlso", -1);
    double Cdl              = params.find<double>("power.Cdl", -1);
    double Rw               = params.find<double>("power.Rw", -1);
    double K1               = params.find<double>("power.K1", -1);
    double Vth              = params.find<double>("power.Vth", -1);
    double Vml              = params.find<double>("power.Vml", -1);
    double Vns              = params.find<double>("power.Vns", -1);
    double Vsl              = params.find<double>("power.Vsl", -1);
    double Vdd              = params.find<double>("power.Vdd", -1);
    double Tclk             = params.find<double>("power.Tclk", -1);
    gHRS                    = params.find<double>("power.gHRS", -1);
    gLRS                    = params.find<double>("power.gLRS", -1);
    powerDac                = params.find<double>("power.dac", -1);
    powerSa                 = params.find<double>("power.sa", -1);
    powerPc                 = params.find<double>("power.pc", -1);
    powerRegStatic          = params.find<double>("power.reg.static", -1);
    powerRegDynamic         = params.find<double>("power.reg.dynamic", -1);

    double Rout = Tclk/(acamRow*Cdl) - 0.5*Rw*(acamRow-1);
    powerDL_row = Vdd*Vdd/Rout*acamCol;
    powerML_row = 0.5 * (Cml*acamCol + Cpre + Cmlso) / Tclk * std::pow(Vml-Vns, 2);
    powerSL_imax = 0.5 * K1 * std::pow(Vml - Vth, 2);
    
    powerDL = registerStatistic<double>("powerDL");
    powerML = registerStatistic<double>("powerML");
    powerSL = registerStatistic<double>("powerSL");
    powerDAC = registerStatistic<double>("powerDAC");
    powerSA = registerStatistic<double>("powerSA");
    powerPC = registerStatistic<double>("powerPC");
    powerREG = registerStatistic<double>("powerREG");
    
    params.find_array("power.GLow", GLow);
    params.find_array("power.GHigh", GHigh);

    /* Program aCAM */
    std::vector<double> acamThLow;
    std::vector<double> acamThHigh;
    std::vector<int32_t> acamThXLow;
    std::vector<int32_t> acamThXHigh;
    params.find_array("acamThLow", acamThLow);
    params.find_array("acamThHigh", acamThHigh);
    params.find_array("acamThXLow", acamThXLow);
    params.find_array("acamThXHigh", acamThXHigh);
    
    matchRows.resize(acamRow);
    for (uint32_t row = 0; row < acamRow; ++row){
        std::vector<double> acamThLowRow = {acamThLow.begin() + row*acamCol, acamThLow.begin() + (row+1)*acamCol};
        std::vector<double> acamThHighRow = {acamThHigh.begin() + row*acamCol, acamThHigh.begin() + (row+1)*acamCol};
        std::vector<int32_t> acamThXLowRow = {acamThXLow.begin() + row*acamCol, acamThXLow.begin() + (row+1)*acamCol};
        std::vector<int32_t> acamThXHighRow = {acamThXHigh.begin() + row*acamCol, acamThXHigh.begin() + (row+1)*acamCol};
        matchRows[row].init(acamCol, GLow, GHigh, powerSL_imax, Vsl, gHRS, gLRS, powerDL, powerML, powerSL);
        matchRows[row].program(acamThLowRow, acamThHighRow, acamThXLowRow, acamThXHighRow);
    }

    /* Configure output (outSTd: command prompt, outFile: txt file) */
    std::string outputDir   = params.find<std::string>("outputDir");
    std::string prefix = "@t @X [Acam" + std::to_string(compId) + "   ]:          ";
    outStd.init(prefix, verbose, 0, Output::STDOUT);
    outFile.init(prefix, verbose, 0, Output::FILE, outputDir+"acam"+std::to_string(compId)+".txt");

    /* Register clock handler */ 
    clockHandler = new Clock::Handler<acam>(this, &acam::clockTick);
    clockPeriod = registerClock(freq, clockHandler);

    /* Configure links */
    enLink      = configureLink("en_port", new Event::Handler<acam>(this, &acam::handleNewEN));
    dlLink      = configureLink("dl_port", new Event::Handler<acam>(this, &acam::handleNewDL));
    outLink     = configureLink("output_port");
    selfLink    = configureSelfLink("self", freq, new Event::Handler<acam>(this, &acam::handleSelf));

    auto dlStat = registerStatistic<uint64_t>("dlq_size");
    dlQueue.setLatency(iqLat);
    dlQueue.setQueueSizeStat(dlStat);
    auto enStat = registerStatistic<uint64_t>("enq_size");
    enQueue.setLatency(iqLat);
    enQueue.setQueueSizeStat(enStat);
    auto outputStat = registerStatistic<uint64_t>("oq_size");
    outputQueue.setLatency(oqLat);
    outputQueue.setQueueSizeStat(outputStat);
    
}   

/**
 * @brief Compare input data with cam contents and produce a match vector
 */
void
acam::handleSelf(Event *ev){
    DataEvent *enEv = static_cast<DataEvent*>(ev);
    std::vector<int32_t> match(acamRow, 0);
    std::vector<int32_t> en = enEv->getDataInt();

    /* Add power consumptions, which are constant and activated when en singal comes */
    powerDL->addData(powerDL_row);
    powerDAC->addData(powerDac*acamCol);
    powerREG->addData(powerRegStatic*acamRow);
    for (uint32_t row = 0; row < acamRow; ++row){
        if (en[row]){
            /* Add power consumptions, which are constant and activated when a row is on */
            powerML->addData(powerML_row);
            powerSA->addData(powerSa);
            powerPC->addData(powerPc);
            powerREG->addData(powerRegDynamic);

            /* Compare input data with cam contents */
            if (matchRows[row].isMatch(dl, dlX)){
                match[row] = 1;
            }
        }
    }
    DataEvent *matchEv = new DataEvent(match, enEv->getIndex1(),enEv->getIndex2(), enEv->getIndex3());
    outputQueue.push(getNextClockCycle(clockPeriod) - 1, matchEv);
    busy = false;
    delete enEv;
}

/**
 * @brief Load the incoming en event to enQueue.
 */
void
acam::handleNewEN(Event *ev) {
    enQueue.push(getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Load the incoming dl event to dlQueue.
 */
void
acam::handleNewDL(Event *ev) {
    dlQueue.push(getNextClockCycle(clockPeriod) - 1, static_cast<DataEvent*>(ev));
}

/**
 * @brief Component operations every clock cycle.
 * @details For every clock cycle, 
 * 1. Pop the output event (match vector) from the outputQueue and send it to MMR through the outputLink
 * 2. Pop the en events from the enQueue 
 * 3. If en event is valid, pop the dl event from the dlQueue and perform the match operation (handleSelf)
 */
bool
acam::clockTick(Cycle_t cycle) {
    /* Pop the output event (match vector) from the outputQueue and send it to MMR through the outputLink */
    auto outEv = outputQueue.pop(cycle);
    if (outEv){
        outLink->send(outEv);
        outStd.verbose(CALL_INFO, 4, 0, "Out -> ID#%d,%d [0] = %d\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataInt()[0]);
        outFile.verbose(CALL_INFO, 5, 0, "Out -> ID#%d,%d [0] = %d\n", outEv->getIndex1(), outEv->getIndex2(), outEv->getDataInt()[0]);
    }
    
    /* Pop the en events from the enQueue  */
    /* If en event is valid, pop the dl event from the dlQueue and perform the match operation (handleSelf) */
    if (!busy){
        auto enEv = enQueue.pop(cycle);
        if (enEv){
            auto dlEv = dlQueue.pop(cycle);
            dl = dlEv->getDataDouble();
            dlX = dlEv->getDataInt();
            outStd.verbose(CALL_INFO, 4, 0, "DL -> ID#%d,%d [0] = %.3lf (%d)\n", dlEv->getIndex1(), dlEv->getIndex2(), dlEv->getDataDouble()[0], dlEv->getDataInt()[0]);
            outFile.verbose(CALL_INFO, 5, 0, "DL -> ID#%d,%d [0] = %.3lf (%d)\n", dlEv->getIndex1(), dlEv->getIndex2(), dlEv->getDataDouble()[0], dlEv->getDataInt()[0]);
            
            selfLink->send(latency - 1, enEv);
            outStd.verbose(CALL_INFO, 4, 0, "EN -> ID#%d,%d [0] = %u\n", enEv->getIndex1(), enEv->getIndex2(), enEv->getDataInt()[0]);
            outFile.verbose(CALL_INFO, 5, 0, "EN -> ID#%d,%d [0] = %u\n", enEv->getIndex1(), enEv->getIndex2(), enEv->getDataInt()[0]);
            busy = true;
        }
    }
    return false; 
}

/**
 * @brief Initialize aCAM row.
 */
void
acam::MatchRow::init(uint32_t _size, std::vector<double> &_GLow, std::vector<double> &_GHigh, double &_powerSL_imax, double &_Vsl, double &_gHRS, double &_gLRS, Statistic<double> *_powerDL, Statistic<double> *_powerML, Statistic<double> *_powerSL){
    low.resize(_size);
    high.resize(_size);
    lowX.resize(_size);
    highX.resize(_size);
    GLow = _GLow;
    GHigh = _GHigh;
    powerSL_imax = _powerSL_imax;
    Vsl = _Vsl;
    gHRS = _gHRS;
    gLRS = _gLRS;
    powerDL = _powerDL;
    powerML = _powerML;
    powerSL = _powerSL;
}

/**
 * @brief Program aCAM row with threshold map (low, high, lowX, highX)
 */
void
acam::MatchRow::program(std::vector<double> _low, std::vector<double> _high, std::vector<int32_t> _lowX, std::vector<int32_t> _highX){
    low = _low;
    high = _high;
    lowX = _lowX;
    highX = _highX;
}

/**
 * @brief Compare the data with aCAM thresholds and produce the match result (0/1)
 */
bool
acam::MatchRow::isMatch(std::vector<double> _data, std::vector<int32_t> _dataX){
    bool match = true;
    uint32_t LSBLo, LSBHi, HSBLo, HSBHi;
    for (uint32_t col = 0; col < _data.size(); ++col){
        if (_dataX[col] == 0){
            if (lowX[col] && highX[col]){
                match = match && true;
            }
            else if (lowX[col]){
                if (!(_data[col] < high[col])){
                    match = match && false;
                }
            }
            else if (highX[col]){
                if (!(low[col] <= _data[col])){
                    match = match && false;
                }
            }
            else{
                if (!((low[col] <= _data[col]) && (high[col] > _data[col]))){
                    match = match && false;
                }
            }
        }

        LSBLo = static_cast<uint32_t>(low[col])%16;
        LSBHi = static_cast<uint32_t>(low[col]/16);
        HSBLo = static_cast<uint32_t>(high[col])%16;
        HSBHi = static_cast<uint32_t>(high[col]/16);
        calcPowerSL(lowX[col], highX[col], LSBLo, LSBHi);
        calcPowerSL(lowX[col], highX[col], HSBLo, HSBHi);
    }
    return match;
}

/**
 * @brief Calculate the SL power, which depends on CAM thresholds.
 */
void
acam::MatchRow::calcPowerSL(int32_t _lowX, int32_t _highX, uint32_t _indexLow, uint32_t _indexHigh){
    double iTotLo = Vsl * GLow[_indexLow];
    double iTotHi = Vsl * GHigh[_indexHigh];
    iTotLo = (_lowX)? Vsl * gHRS : iTotLo;
    iTotHi = (_highX)? Vsl * gLRS : iTotHi;
    iTotLo = (iTotLo > powerSL_imax)? powerSL_imax : iTotLo;
    iTotHi = (iTotHi > powerSL_imax)? powerSL_imax : iTotHi;
    powerSL->addData(Vsl * (iTotLo+iTotHi));
}
