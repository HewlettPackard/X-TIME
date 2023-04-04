
#pragma once

#include <sst/core/event.h>

#include <cstdint>

namespace SST {
namespace XTIME {    

/**
* @brief Event flowing between components in 'xtime' library
*/
class DataEvent : public Event {

public:
    void serialize_order(SST::Core::Serialization::serializer &ser)  override {
        Event::serialize_order(ser);
        ser & dataDouble;
        ser & dataInt;
        ser & index1;
        ser & index2;
        ser & index3;
    }    

    /**
    * @brief Constructor with vector<double>, vector<int>, int, int, int
    */
    DataEvent(std::vector<double> dataDouble, std::vector<int32_t> dataInt, int32_t index1, int32_t index2, int32_t index3) :
        Event(),
        dataDouble(dataDouble),
        dataInt(dataInt),
        index1(index1),
        index2(index2),
        index3(index3)
    {}

    /**
    * @brief Constructor with vector<double>, int, int, int
    */
    DataEvent(std::vector<double> dataDouble, int32_t index1, int32_t index2, int32_t index3) :
        Event(),
        dataDouble(dataDouble),
        index1(index1),
        index2(index2),
        index3(index3)
    {}

    /**
    * @brief Constructor with vector<int>, int, int, int
    */
    DataEvent(std::vector<int32_t> dataInt, int32_t index1, int32_t index2, int32_t index3) :
        Event(),
        dataInt(dataInt),
        index1(index1),
        index2(index2),
        index3(index3)
    {}

    std::vector<double> getDataDouble() { return dataDouble; }
    std::vector<int32_t> getDataInt() { return dataInt; }
    int32_t getIndex1() { return index1; }
    int32_t getIndex2() { return index2; }
    int32_t getIndex3() { return index3; }

private:
    DataEvent()  {} // For Serialization only

    std::vector<double> dataDouble;
    std::vector<int32_t> dataInt;
    int32_t index1;
    int32_t index2;
    int32_t index3;

    ImplementSerializable(SST::XTIME::DataEvent);
};

/**
* @brief Event for computing DataEvents from multiple ports at the same cycle
*/
class DataMatrixEvent : public Event {

public:
    void serialize_order(SST::Core::Serialization::serializer &ser)  override {
        Event::serialize_order(ser);
        ser & dataDouble;
        ser & dataInt;
        ser & index1;
        ser & index2;
        ser & index3;
    }    
    
    /**
    * @brief Constructor with vector<vector<double>>, vector<vector<int>>, int, int, int
    */
    DataMatrixEvent(std::vector<std::vector<double>> dataDouble, std::vector<std::vector<int32_t>> dataInt, int32_t index1, int32_t index2, int32_t index3) :
        Event(),
        dataDouble(dataDouble),
        dataInt(dataInt),
        index1(index1),
        index2(index2),
        index3(index3)
    {}
    
    /**
    * @brief Constructor with vector<vector<int>>, int, int, int
    */
    DataMatrixEvent(std::vector<std::vector<int32_t>> dataInt, int32_t index1, int32_t index2, int32_t index3) :
        Event(),
        dataInt(dataInt),
        index1(index1),
        index2(index2),
        index3(index3)
    {}

    std::vector<std::vector<double>> getDataDouble() { return dataDouble; }
    std::vector<std::vector<int32_t>> getDataInt() { return dataInt; }
    int32_t getIndex1() { return index1; }
    int32_t getIndex2() { return index2; }
    int32_t getIndex3() { return index3; }

private:
    DataMatrixEvent()  {} // For Serialization only

    std::vector<std::vector<double>> dataDouble;
    std::vector<std::vector<int32_t>> dataInt;
    int32_t index1;
    int32_t index2;
    int32_t index3;

    ImplementSerializable(SST::XTIME::DataMatrixEvent);
};

}
}
