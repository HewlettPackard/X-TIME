#ifndef _MMR_COMPONENT_H
#define _MMR_COMPONENT_H

#include "event.h"
#include "data_queue.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/timeConverter.h>
#include <sst/core/timeLord.h>
#include <sst/core/output.h>

#include <cstdint>
#include <vector>
#include <algorithm>

namespace SST {
namespace XTIME {
/**
* @brief Multiple Match Resolver (MMR) in Core
* @details Receive data packet (match vector) from 'acam', convert the match vector to the address, and send data packet (address) to 'memory'.\n
* Input data packet is consisted of match vector (vector<int>), and sampleID (int). Convert a match vector (size:acamRow) to an address (index of first true) until the match vector has no true.\n
* Output data packet is consisted of address (vecotr<int>), and sampleID (int).
*/
class mmr : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.mmr" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        mmr,
        "xtime",
        "mmr",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Multiple Match Resolver",
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

        {"numPort",             "(uint) Number of output port", "4"},
        {"numMatch",            "(int) Number of matches", "-1"},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"input_port%d",        "Raw match results from aCAM", {"XTIME.DataEvent"}},
        {"output_port",         "Addresses to SRAM", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
        { "powerMMR",           "Power consumption of MMR", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    mmr(ComponentId_t id, Params& params);
    ~mmr() { }

    void handleNewData( SST::Event* ev );
    void handleSelf( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( uint32_t phase ) {}
	void setup() { }
    void finish() { }

private:
    class Port {
    public:
        Port(uint32_t portID, TimeConverter *clockPeriod, Output &outStd, mmr *mmr):
            portID(portID),
            clockPeriod(clockPeriod),
            outStd(outStd),
            m_mmr(mmr)
        {}

        void handleNewData( SST::Event* ev );
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
        mmr                       *m_mmr;
    };

    /* Clock *****************************************************************/
    Clock::Handler<mmr>         *clockHandler;
    TimeConverter               *clockPeriod;

    /* IO ********************************************************************/
    Output                      outStd;
    Output                      outFile;

    /* Link/Port *************************************************************/
    std::vector<Link*>          inputLink;
    Link*                       outputLink;
    Link*                       selfLink;
    std::vector<Port*>          inputPort;

    /* Temporary data/result *************************************************/
    Queue<DataEvent*>           outputQueue;

    /* Parameters ************************************************************/
    uint32_t                    latency;
    uint32_t                    numPort;
    int32_t                     numMatch;

    /** Power parameters ******************************************************/
    double                      powerMmr;
    Statistic<double>*          powerMMR;

    /* Control signal ********************************************************/
    bool                        busy = false;
};

}
}

#endif
