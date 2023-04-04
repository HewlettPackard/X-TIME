#ifndef _DEMUX_COMPONENT_H
#define _DEMUX_COMPONENT_H

#include "event.h"
#include "data_queue.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/timeConverter.h>
#include <sst/core/timeLord.h>
#include <sst/core/output.h>

#include <iostream>
#include <vector>

namespace SST {
namespace XTIME {
/**
* @brief Demux in Router
* @details Receive the data packet, and distribute its copies to the ports according to 'mode' and routing info in the packet.\n 
* Data packet is consisted of data (vector<double>), routing info (vector<int>), and sampleID (int). ('level'+2)th element of routing info indicates the group index of output ports the data should go.
* (e.g. When 'mode' of demux at level 0 is [3,1], data packet with routing info [2,0,3,4] goes through port#0, #1, #2 of this demux. Data packet with routing info [2,1,3,4] goes through port#3 of this demux)\n
* 'mode' indicates how many ports are tied. (e.g. 'mode' [3,1] means this demux has 4 ports and port#0~#2 are tied)
*/
class demux : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.demux" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        demux,
        "xtime",
        "demux",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Demux in Router",
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

        {"level",               "(uint) Level in tree NoC", "1"},
        {"numPort",             "(uint) Number of output port", "4"},
        {"mode",                "(vector<uint>) Deumx mode", "[4]"},
        {"outputDir",           "(string) Path of output files", " "},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"input_port",          "Input port", {"XTIME.DataEvent"}},
        {"output_port%d",       "Output ports", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
        { "powerDEMUX",         "Power consumption of Demux", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    demux(ComponentId_t id, Params& params);
    ~demux() { }

    void handleSelf( SST::Event* ev );
    void handleNewData( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( unsigned int phase ) {}
	void setup() { }
    void finish() { }

private:
    class Port {
    public:
        Port(uint32_t portID, TimeConverter *clockPeriod, Output &outStd):
            portID(portID),
            clockPeriod(clockPeriod),
            outStd(outStd)
        {}

        DataEvent* getEvent(Cycle_t curCycle){
            return portQueue.pop(curCycle);
        }
        Queue<DataEvent*>& getQueue(){
            return portQueue;
        }
    private:
        uint32_t                    portID;
        TimeConverter               *clockPeriod;
        Queue<DataEvent*>           portQueue;
        Output                      outStd;
    };

    /** Clock *****************************************************************/
    Clock::Handler<demux>       *clockHandler;
    TimeConverter               *clockPeriod;

    /** IO ********************************************************************/
    Output                      outStd;
    Output                      outFile;

    /** Link/Port *************************************************************/
    std::vector<Link*>          outputLink;
    Link*                       inputLink;
    Link*                       selfLink;
    std::vector<Port*>          outputPort;

    /** Temporary data/result *************************************************/
    Queue<DataEvent*>           inputQueue;

    /** Parameters ************************************************************/
    uint32_t                    latency;
    uint32_t                    level;
    uint32_t                    numPort;
    std::vector<uint32_t>       mode;

    /** Power parameters ******************************************************/
    double                      powerReg;
    double                      powerDemux;
    Statistic<double>*          powerDEMUX;

    /** Control signal ********************************************************/
    bool                        busy = false;

};

}
}

#endif
