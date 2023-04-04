#ifndef _ADDER_COMPONENT_H
#define _ADDER_COMPONENT_H

#include "event.h"
#include "data_queue.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/timeConverter.h>
#include <sst/core/timeLord.h>
#include <sst/core/output.h>

#include <fstream>
#include <cstdint>
#include <vector>

namespace SST {
namespace XTIME {
/**
* @brief Adder in Core
* @details Accumulate the input results.\n 
* Input results come from different rows of stacked aCAMs. They must have same classID but may have different sampleID. If they have different sampleID, send DataEvents serially. 
* Assume the computing time is constant, 'latency'*'numPort'.
*/
class adder : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.adder" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        adder,
        "xtime",
        "adder",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Adder in Core",
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

        {"numMatch",            "(int) Number of matches", "1"},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"input_port",          "Input port", {"XTIME.DataEvent"}},
        {"output_port",         "Output port", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
        { "powerADDER",         "Power consumption of Adder", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    adder(ComponentId_t id, Params& params);
    ~adder() { }

    void handleNewData( SST::Event* ev );
    void handleSelf( SST::Event* ev );
    bool clockTick( Cycle_t cycle );
    
    void init( uint32_t phase ) {}
	void setup() { }
    void finish() { }

private:
    /** Clock *****************************************************************/
    Clock::Handler<adder>       *clockHandler;
    TimeConverter               *clockPeriod;

    /** IO ********************************************************************/
    Output                      outStd;
    Output                      outFile;

    /** Link/Port *************************************************************/
    Link*                       inputLink;
    Link*                       outputLink;
    Link*                       selfLink;

    /** Temporary data/result *************************************************/
    Queue<DataEvent*>           outputQueue;
    Queue<DataEvent*>           inputQueue;

    /** Parameters ************************************************************/
    uint32_t                    latency;
    uint32_t                    numMatch;
    uint32_t                    numAdded = 0;
    std::vector<double>         tempData;
    std::vector<int32_t>        tempIndex;

    /** Power parameters ******************************************************/
    double                      powerAdder;
    Statistic<double>*          powerADDER;

    /** Control signal ********************************************************/
    bool                        busy = false;

};

}
}

#endif
