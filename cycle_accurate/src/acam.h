#ifndef _ACAM_COMPONENT_H
#define _ACAM_COMPONENT_H

#include "event.h"
#include "data_queue.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/timeConverter.h>
#include <sst/core/timeLord.h>
#include <sst/core/output.h>

#include <iostream>
#include <vector>
#include <cmath>

namespace SST {
namespace XTIME {
/**
* @brief Analog Content-Addressable Memory (aCAM) in Core
* @details Evaluate a list of conditions (acamThLow <= dl < acamThHigh) per row when the enable singal vector (en) is arrived. If every condition is true, the match result is also true.\n
* Analog thresholds (acamThLow, acamThHigh, acamThXLow, acamThXHigh) are programmed in constructor. If 'don't care' threshold (acamThXLow, acamThXHigh) is true, the low/high-side condition is always true.\n
* Data line value (dl, size: acamCol) is applied from 'driver'. Enable signal vector (en, size: acamRow) is applied from 'driver' or previous 'acam'. Only the rows with true value of enable signal vector are active.
*/
class acam : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.acam" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        acam,
        "xtime",
        "acam",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME aCAM",
        COMPONENT_CATEGORY_UNCATEGORIZED
    );
    /**
    * @brief List of parameters
    * @details SST_ELI_DOCUMENT_PARAMS({ “name”, “description”, “default value” }).
    */
    SST_ELI_DOCUMENT_PARAMS(
        {"id",                  "(uint) ID of the component", "0"},
        {"verbose",             "(uint) Output verbosity. The higher verbosity, the more debug info", "0"},
        {"inputQueueLatency",   "(uint) Latency of input queue", "1"},
        {"outputQueueLatency",  "(uint) Latency of output queue", "1"},
        {"freq",                "(string) Clock frequency", "1GHz"},
        {"latency",             "(uint) Latency of component operation (handleSelf)", "0"},

        {"acamCol",             "(uint) Number of acam column", "32"},
        {"acamRow",             "(uint) Number of acam row", "256"},

        {"acamThLow",           "(vector<double>) Low threshold", " "},
        {"acamThHigh",          "(vector<double>) High threshold", " "},
        {"acamThXLow",          "(vector<int32_t>) Low 'don't care' threshold", " "},
        {"acamThXHigh",         "(vector<int32_t>) High 'dont' care' threshold", " "},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"en_port",             "Enable signal", {"XTIME.DataEvent"}},
        {"dl_port",             "Data", {"XTIME.DataEvent"}},
        {"output_port",         "Match port", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "dlq_size",           "Data line output queue size on insertion", "packet", 2},
        { "enq_size",           "Enable signal output queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
        { "powerDL",            "Power consumption of Data line", "W", 1},
        { "powerML",            "Power consumption of Match line", "W", 1},
        { "powerSL",            "Power consumption of Source line", "W", 1},
        { "powerDAC",           "Power consumption of DAC", "W", 1},
        { "powerSA",            "Power consumption of SA", "W", 1},
        { "powerPC",            "Power consumption of PC", "W", 1},
        { "powerREG",           "Power consumption of REG", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    acam(ComponentId_t id, Params& params);
    ~acam() { }

    void handleNewEN( SST::Event* ev );
    void handleNewDL( SST::Event* ev );
    void handleSelf( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( uint32_t phase ) {}
	void setup() { }
    void finish() { }

private:
    class MatchRow {
    public:
        MatchRow() {}
        void init(uint32_t _size, std::vector<double> &_GLow, std::vector<double> &_GHigh, double &_powerSL_imax, double &_Vsl, double &_gHRS, double &_LRS, Statistic<double> *_powerDL, Statistic<double> *_powerML, Statistic<double> *_powerSL);
        void program(std::vector<double> _low, std::vector<double> _high, std::vector<int32_t> _lowX, std::vector<int32_t> _highX);
        bool isMatch(std::vector<double> _data, std::vector<int32_t> _dataX);
        void calcPowerSL(int32_t _lowX, int32_t _highX, uint32_t _indexLow, uint32_t _indexHigh);
    private:
        std::vector<double> low;
        std::vector<double> high;
        std::vector<int32_t> lowX;
        std::vector<int32_t> highX;

        std::vector<double> GLow;
        std::vector<double> GHigh;
        double powerSL_imax;
        double Vsl;
        double gHRS;
        double gLRS;
        Statistic<double>* powerDL = nullptr;
        Statistic<double>* powerML = nullptr;
        Statistic<double>* powerSL = nullptr;
    };
    
    /** Clock *****************************************************************/
    Clock::Handler<acam>        *clockHandler;
    TimeConverter               *clockPeriod;
    
    /** IO ********************************************************************/
    Output                      outStd;
    Output                      outFile;

    /** Link/Port *************************************************************/
    Link*                       enLink;
    Link*                       dlLink;
    Link*                       outLink;
    Link*                       selfLink;

    /** Temporary data/result *************************************************/
    std::vector<MatchRow>       matchRows;
    Queue<DataEvent*>           outputQueue;
    Queue<DataEvent*>           dlQueue;
    Queue<DataEvent*>           enQueue;
    std::vector<double>         dl;
    std::vector<int32_t>        dlX;

    /** Parameters ************************************************************/
    uint32_t                    latency;
    uint32_t                    acamCol;
    uint32_t                    acamRow;

    /** Power parameters ******************************************************/
    double                      gHRS;
    double                      gLRS;
    std::vector<double>         GLow;
    std::vector<double>         GHigh;

    double                      powerDL_row;
    double                      powerML_row;
    double                      powerSL_imax;
    double                      powerDac;
    double                      powerSa;
    double                      powerPc;
    double                      powerRegStatic;
    double                      powerRegDynamic;

    Statistic<double>*          powerDL;
    Statistic<double>*          powerML;
    Statistic<double>*          powerSL;
    Statistic<double>*          powerDAC;
    Statistic<double>*          powerSA;
    Statistic<double>*          powerPC;
    Statistic<double>*          powerREG;

    /** Control signal ********************************************************/
    bool                        busy = false;
};

}
}

#endif
