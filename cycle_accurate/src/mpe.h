
#ifndef _MPE_COMPONENT_H
#define _MPE_COMPONENT_H

#include "event.h"
#include "data_queue.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/timeConverter.h>
#include <sst/core/timeLord.h>
#include <sst/core/output.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace SST {
namespace XTIME {
/**
* @brief Match Processing Unit (MPE) in Core
* @details Receive a series of data packet (match vector) from 'acam', and compute SHAP value given foreground/background sample pair.\n
* Input data packet is consisted of match vector (vector<int>), sampleID (int), and flag bit (1: foreground, -1: background). The number of packets is 2*#numFeature in total.\n
* When 2*#numFeature packets are received and processed, the SHAP value can be calculated and sent to 'accumulator'.
* Output data packet is consisted of SHAP value (vecotr<int>), sampleID (int), classID (int), and 1 (int).
*/
class mpe : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        mpe,
        "xtime",
        "mpe",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "XTIME Match Processing Element",
        COMPONENT_CATEGORY_UNCATEGORIZED
    );
    SST_ELI_DOCUMENT_PARAMS(
        {"id",                  "(uint) ID of the component", "0"},
        {"verbose",             "(uint) Output verbosity. The higher verbosity, the more debug info", "0"},
        {"inputQueueLatency",   "(uint) Latency of input queue", "1"},
        {"outputQueueLatency",  "(uint) Latency of output queue", "1"},
        {"freq",                "(string) Clock frequency", "1GHz"},
        {"latency",             "(uint) Latency of component operation (handleSelf)", "0"},

        {"numPort",             "(uint) Number of input ports", "1"},
        {"numFeature",          "(uint) Number of features", "10"},
        {"acamRow",             "(uint) Number of acam rows", "256"},
        {"classID",             "(uint) ID of class", "-1"},
        {"logit",               "(list) Logits", " "},
        {"maxDepth",            "(uint) Max depth of tree-based model", "10"},
        {"weightFile",          "(string) Name of weight file", " "},
    );
    SST_ELI_DOCUMENT_PORTS(
        {"input_port%d",        "Match port", {"XTIME.DataEvent"}},
        {"output_port",         "Result port", {"XTIME.DataEvent"}},
    );
    SST_ELI_DOCUMENT_STATISTICS(
        { "iq_size",            "Input queue size on insertion", "packet", 2},
        { "oq_size",            "Output queue size on insertion", "packet", 2},
        // { "powerMMR",           "Power consumption of MMR", "W", 1},
    );
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
    );

    mpe(ComponentId_t id, Params& params);
    ~mpe() { }

    void handleNewData( SST::Event* ev );
    void handleSelf( SST::Event* ev );
    bool clockTick( Cycle_t cycle );

    void init( uint32_t phase ) {}
	void setup() { }
    void finish() { }

private:
    class Port {
    public:
        Port(uint32_t portID, TimeConverter *clockPeriod, Output &outStd, mpe *mpe):
            portID(portID),
            clockPeriod(clockPeriod),
            outStd(outStd),
            m_mpe(mpe)
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
        mpe                         *m_mpe;
    };

    /* Clock *****************************************************************/
    Clock::Handler<mpe>         *clockHandler;
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
    std::vector<int32_t>        n;
    std::vector<int32_t>        s;
    std::vector<int32_t>        p;
    std::vector<double>         v;
    std::vector<double>         shapley;
    std::vector<std::vector<int32_t>> u;
    std::vector<std::vector<int32_t>> c;
    std::vector<std::vector<double>> weightMatrix;
    std::vector<int32_t>        foreground;
    std::vector<int32_t>        background;
    uint32_t                    numPair = 0;

    /* Parameters ************************************************************/
    uint32_t                    latency;
    uint32_t                    numPort;
    uint32_t                    numFeature;
    uint32_t                    acamRow;
    uint32_t                    classID;

    /* Control signal ********************************************************/
    bool                        busy = false;
};

}
}

#endif
