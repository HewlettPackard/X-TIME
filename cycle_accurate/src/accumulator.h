#ifndef _ACCUMULATOR_COMPONENT_H
#define _ACCUMULATOR_COMPONENT_H

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
* @brief Accumulator in Router
* @details Recieve the result packets, and accumulate them, which have same sampleID and classID.\n 
* Result packet is consisted of result (vector<double>), sampleID (int), classID (int), sumTree (int). sumTree indicates how many tree contributes its result.\n 
* According to 'mode', it can accumulate (0) or bypass (1) the results. Only the results with same sampleID and classID can be accumulated. \n 
* When the results are accumulated, both result and sumTree are updated. Assume the computing time is constant, 'latency'*'numPort'.
*/
class accumulator : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.accumulator" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        accumulator,
        "xtime",
        "accumulator",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Accumulator in Router",
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

        {"numPort",             "(uint) Number of ports", "4"},
        {"mode",                "(uint) Accumulate (0) or Bypass (1)", "0"},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"input_port%d",        "Input ports", {"XTIME.DataEvent"}},
        {"output_port",         "Output port ", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
        { "powerACCUM",         "Power consumption of Accumulator", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    accumulator(ComponentId_t id, Params& params);
    ~accumulator() { }

    void handleSelf( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( unsigned int phase ) {}
	void setup() { }
    void finish() { }

private:
    class Port {
    public:
        Port(uint32_t portID, TimeConverter *clockPeriod, Output &outStd, accumulator *accumulator):
            portID(portID),
            clockPeriod(clockPeriod),
            outStd(outStd),
            m_accumulator(accumulator)
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
        accumulator                 *m_accumulator;
    };
    /** Clock *****************************************************************/
    Clock::Handler<accumulator> *clockHandler;
    TimeConverter               *clockPeriod;

    /** IO ********************************************************************/
    Output                      outStd;
    Output                      outFile;

    /** Link/Port *************************************************************/
    std::vector<Link*>          inputLink;
    Link*                       outputLink;
    Link*                       selfLink;
    std::vector<Port*>          inputPort;

    /** Temporary data/result *************************************************/
    Queue<DataEvent*>           outputQueue;

    /** Parameters ************************************************************/
    uint32_t                    latency;
    uint32_t                    numPort;
    uint32_t                    mode;

    /** Power parameters ******************************************************/
    double                      powerReg;
    double                      powerAccum;
    Statistic<double>*          powerACCUM;

    /** Control signal ********************************************************/
    bool                        busy = false;
    
};

}
}

#endif
