#pragma once

#include <sst/core/simulation.h>
#include <sst/core/statapi/statbase.h>

#include <queue>
#include <cstdint>

namespace SST {
namespace XTIME {

using namespace Statistics;

template<class T>
class Queue {
public:
    using Entry = std::pair<Cycle_t, T>;
    Queue(Cycle_t latency = 1) :
        m_latency(latency)
    {
    }

    void setLatency(Cycle_t latency) { m_latency = latency; }
    void setQueueSizeStat(Statistic<uint64_t> *stat) { m_statQueueSize = stat; }

    void push(Cycle_t cycle, T obj) {
        m_delayQueue.push(std::make_pair(cycle + m_latency, obj));
        if(m_statQueueSize) {
            m_statQueueSize->addData(m_delayQueue.size());
        }
    }
    
    void push_relative(Cycle_t cycle, T obj) {
        if(m_delayQueue.empty()) {
            m_delayQueue.push(std::make_pair(cycle + 1, obj));
        }
        else{
            auto back = m_delayQueue.back();
            m_delayQueue.push(std::make_pair(back.first + m_latency, obj));
        }
        if(m_statQueueSize) {
            m_statQueueSize->addData(m_delayQueue.size());
        }
    }

    T pop(Cycle_t cycle) {
        T out = nullptr;
        if(!m_delayQueue.empty()) {
            auto entry = m_delayQueue.front();
            if(entry.first <= cycle) {
                m_delayQueue.pop();
                out = entry.second;
            }
        }
        return out;
    }

    Cycle_t              m_latency;
    std::queue<Entry>    m_delayQueue;
    Statistic<uint64_t>* m_statQueueSize = nullptr;
};

}
}
