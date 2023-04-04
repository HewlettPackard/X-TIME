#ifndef _MEMORY_COMPONENT_H
#define _MEMORY_COMPONENT_H

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
* @brief Memory in Core
* @details Receive data packet (address) from 'mmr', and send result packet (logit) to 'adder'.\n
* Leaf values in the tree-based model (logit) are programmed in constructor. Assume the leaves mapped on this memory are coming from the trees with same classID.\n
* Data packet is consisted of address (vector<double>), and sampleID (int).\n
* Result packet is consisted of result (vector<double>), sampleID (int), classID (int), sumTree (int). Assign its classID to 'classID' of result packet. sumTree is 1.\n 
*/
class memory : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.memory" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        memory,
        "xtime",
        "memory",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Memory",
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

        {"classID",             "(uint) ID of class", "-1"},
        {"logit",               "(list) Logits", " "},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"input_port",          "Address from mmr", {"XTIME.DataEvent"}},
        {"output_port",         "Logits to adder", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
        { "powerMEMORY",        "Power consumption of Memory", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    memory(ComponentId_t id, Params& params);
    ~memory() { }

    void handleNewData( SST::Event* ev );
    void handleSelf( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( uint32_t phase ) {}
	void setup() { }
    void finish() { }

private:
    /** Clock *****************************************************************/
    Clock::Handler<memory>      *clockHandler;
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
    std::vector<double>         logit;
    
    /** Parameters ************************************************************/
    uint32_t                    latency;
    int32_t                     classID;

    /** Power parameters ******************************************************/
    double                      powerMemory;
    Statistic<double>*          powerMEMORY;

    /** Control signal ********************************************************/
    bool                        busy = false;
    
};

}
}

#endif
