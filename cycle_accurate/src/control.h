#ifndef _CONTROL_COMPONENT_H
#define _CONTROL_COMPONENT_H

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
#include <iostream>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

namespace SST {
namespace XTIME {
/**
* @brief Control
* @details Send the data packets, receive the result packets, and predict the class label/regression result.\n
* Data packet is consisted of data (vector<double>), routing info (vector<int>), and sampleID (int). First element of routing info indicates the index of output port the data should go.\n 
* Result packet is consisted of result (vector<double>), sampleID (int), classID (int), sumTree (int). sumTree indicates how many tree contributes its result.\n 
* When every class has the complete result, which is the sum of 'numTreePerClass' number of tree results, it predict the class label/regression result.
*/
class control : public SST::Component {
public:
    /**
    * @brief Register a component.
    * @details SST_ELI_REGISTER_COMPONENT(class, “library”, “name”, version, “description”, category).
    * The full name used to reference this is "xtime.control" ("library.name")
    */
    SST_ELI_REGISTER_COMPONENT(
        control,
        "xtime",
        "control",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Control",
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
        {"loadLatency",         "(uint) Latency of load operation (handleSelf)", "1"},

        {"task",                "(string) Type of task", "classification"},
        {"numPort",             "(uint) Number of ports", "1"},
        {"numFeature",          "(uint) Number of features", "10"},
        {"numSample",           "(uint) Number of test samples", "100"},
        {"numLevel",            "(uint) Number of sub-tree levels", "6"},
        {"numClass",            "(uint) Number of classes", "1"},
        {"numTreePerClass",     "(uint) Number of trees per class", "1"},
        {"numBatch",            "(uint) Number of batches", "1"},
        {"mode",                "(vector<int>) Port configuration"},
        {"numInputConfig",      "(vector<int>) Number of input packets for given numBatch"},
        {"xFile",               "(string) Name of input data file", " "},
        {"yFile",               "(string) Name of truth file", " "},
        {"outputDir",           "(string) Path of output files", " "},
    );
    /**
    * @brief List of ports
    * @details SST_ELI_DOCUMENT_PORTS({ “name”, “description”, vector of supported events }).
    */
    SST_ELI_DOCUMENT_PORTS(
        {"input_port%d",        "Result port", {"XTIME.DataEvent"}},
        {"output_port%d",       "Data port", {"XTIME.DataEvent"}},
    );
    /**
    * @brief List of statistics
    * @details SST_ELI_DOCUMENT_STATISTICS({ “name”, “description”, “units”, enable level }).
    */
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    control(ComponentId_t id, Params& params);
    ~control() { }
    
    void handleLoad( );
    void handleSelf( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( uint32_t phase ) {}
	void setup() { }
    void finish() { }
    
private:
    class Port {
    public:
        Port(uint32_t portID, TimeConverter *clockPeriod, Output &outStd, control *control):
            portID(portID),
            clockPeriod(clockPeriod),
            outStd(outStd),
            m_control(control)
        {}

        void handleNewResult( SST::Event* ev );
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
        control                     *m_control;
    };

    /** Clock *****************************************************************/
    Clock::Handler<control>         *clockHandler;
    TimeConverter                   *clockPeriod;

    /** IO ********************************************************************/
    Output                          outStd;
    Output                          outFile;
    Output                          outResult;
    std::ifstream                   xStream;
    std::ifstream                   yStream;

    /** Link/Port *************************************************************/
    std::vector<Link*>              inputLink;
    std::vector<Link*>              outputLink;
    Link*                           selfLink;
    std::vector<Port*>              inputPort;
    std::vector<Port*>              outputPort;

    /** Temporary data/result *************************************************/
    std::vector<std::vector<double>> xMatrix;
    std::vector<std::vector<int32_t>> xInfo;
    std::vector<std::vector<double>> result;
    std::vector<std::vector<uint32_t>> resultSumTree;
    int32_t                         numTested = 0;
    uint32_t                        numCorrect = 0;
    uint32_t                        numSent = 0;
    double                          mse = 0.0;
    time_t                          timeStart;
    SimTime_t                       latencySimTime = 0;
    std::vector<SimTime_t>          latencySample;

    /** Parameters ************************************************************/
    uint32_t                        selfLatency;
    std::string                     task;
    uint32_t                        numPort;
    uint32_t                        numFeature;
    uint32_t                        numSample;
    uint32_t                        numLevel;
    uint32_t                        numClass;
    uint32_t                        numTreePerClass;
    uint32_t                        numBatch;
    std::vector<uint32_t>           mode;
    std::vector<uint32_t>           numInputConfig;

    /** Control signal ********************************************************/
    bool                            busy = false;
    bool                            load = true;
    
};

}
}

#endif
