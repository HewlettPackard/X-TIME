#ifndef _DRIVER_COMPONENT_H
#define _DRIVER_COMPONENT_H

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
* @brief Driver in Core
* @details Receive the data packet, and apply data and enable signal to acams.\n
* Data packet is consisted of data (vector<double>), routing info (vector<int>), and sampleID (int). Zero-pad the incoming data to fit it to size of 'acamQueue'*'acamCol'.\n
* Apply the same data (size:acamCol) to acams in the same queue. Enable signal (size:acamRow) is applied to acams in the first queue.
*/
class driver : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.driver" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        driver,
        "xtime",
        "driver",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Driver in Core",
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
        {"latency",             "(uint) Latency of component operation (handleSelf)", "1"},
        
        {"task",                "(string) Type of task", "classification"},
        {"acamQueue",           "(uint) Number of queued acams", "1"},
        {"acamStack",           "(uint) Number of stacked acams", "1"},
        {"acamCol",             "(uint) Number of acam column", "32"},
        {"acamRow",             "(uint) Number of acam row", "256"},
        {"outputDir",           "(string) Path of output files", " "},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"input_port",          "Input port", {"XTIME.DataEvent"}},
        {"dl_port%d",           "Data to aCAMs", {"XTIME.DataEvent"}},
        {"en_port%d",           "Enable signal to aCAMs", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "dlq_size",           "Data line output queue size on insertion", "packet", 2},
        { "enq_size",           "Enable signal output queue size on insertion", "packet", 2},
        { "powerDRIVER",        "Power consumption of Driver", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    driver(ComponentId_t id, Params& params);
    ~driver() { }

    void handleSelf( SST::Event* ev );
    void handleNewData( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( unsigned int phase ) {}
	void setup() { }
    void finish() { }

private:
    /** Clock *****************************************************************/
    Clock::Handler<driver>      *clockHandler;
    TimeConverter               *clockPeriod;

    /** IO ********************************************************************/
    Output                      outStd;
    Output                      outFile;

    /** Link/Port *************************************************************/
    std::vector<Link*>          dlLinks;
    std::vector<Link*>          enLinks;
    Link*                       inputLink;
    Link*                       selfLink;

    /** Temporary data/result *************************************************/
    Queue<DataEvent*>           inputQueue;
    Queue<DataMatrixEvent*>     dlQueue;
    Queue<DataMatrixEvent*>     enQueue;

    /** Parameters ************************************************************/
    uint32_t                    latency;
    uint32_t                    numFeature;
    uint32_t                    acamQueue;
    uint32_t                    acamStack;
    uint32_t                    acamCol;
    uint32_t                    acamRow;
    std::string                 task;

    /** Power parameters ******************************************************/
    double                      powerReg;
    double                      powerDriver;
    Statistic<double>*          powerDRIVER;

    /** Control signal ********************************************************/
    bool                        busy = false;
};

}
}

#endif
