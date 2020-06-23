//
// Created by maierj on 6/22/20.
//

#ifndef GENERATEVIRTUALSEQUENCE_POSE_PARTITIONING_H
#define GENERATEVIRTUALSEQUENCE_POSE_PARTITIONING_H

#include <glob_includes.h>
#include "tuple"
#include "getStereoCameraExtr.h"
#include "helper_funcs.h"

using namespace std;

template<typename T>
void shuffleVector(std::vector<T> &idxs, size_t si);
template<typename T, typename T1>
void reOrderVector(std::vector<T> &reOrderVec, std::vector<T1> &idxs);

struct actRange{
    std::pair<double,double> txRange;
    std::pair<double,double> tyRange;
    std::pair<double,double> tzRange;
    std::pair<double,double> rollRange;
    std::pair<double,double> pitchRange;
    std::pair<double,double> yawRange;

    explicit actRange():
            txRange(make_pair(0, 0)),
            tyRange(make_pair(0, 0)),
            tzRange(make_pair(0, 0)),
            rollRange(make_pair(0, 0)),
            pitchRange(make_pair(0, 0)),
            yawRange(make_pair(0, 0)){}

    bool empty() const{
        return nearZero(txRange.first) && nearZero(txRange.second);
    }
};

struct partitionPoses{
    vector<std::pair<double,double>> txRangei;
    vector<std::pair<double,double>> tyRangei;
    vector<std::pair<double,double>> tzRangei;
    vector<std::pair<double,double>> rollRangei;
    vector<std::pair<double,double>> pitchRangei;
    vector<std::pair<double,double>> yawRangei;
    std::pair<double,double> txRange_init;
    std::pair<double,double> tyRange_init;
    std::pair<double,double> tzRange_init;
    std::pair<double,double> rollRange_init;
    std::pair<double,double> pitchRange_init;
    std::pair<double,double> yawRange_init;
    size_t nr_stereoConfs = 0;
    size_t nr_var = 0;
    size_t nr_partitions = 0;
    bool txVariable = false;
    bool tyVariable = false;
    bool tzVariable = false;
    bool rollVariable = false;
    bool pitchVariable = false;
    bool yawVariable = false;
    double txSize = 0;
    double tySize = 0;
    double tzSize = 0;
    double rollSize = 0;
    double pitchSize = 0;
    double yawSize = 0;
    bool available = false;
    int vec_pointer = -3;
    actRange userng;
    bool valid_found = false;
    //Parameter ID, range, nr of partitions, is translation
    vector<tuple<variablePars, double, size_t, bool>> parts;

    partitionPoses(std::pair<double,double> txRange_init_,
                   std::pair<double,double> tyRange_init_,
                   std::pair<double,double> tzRange_init_,
                   std::pair<double,double> rollRange_init_,
                   std::pair<double,double> pitchRange_init_,
                   std::pair<double,double> yawRange_init_,
                   bool txVariable_,
                   bool tyVariable_,
                   bool tzVariable_,
                   bool rollVariable_,
                   bool pitchVariable_,
                   bool yawVariable_,
                   const size_t &nr_stereoConfs_);

    partitionPoses()= default;

    void partition();
    bool adaptRange(double &first, double &second, double &y1, double &y2,
                    const variablePars &type0, const variablePars &type1) const;
    bool adaptRangeFixed(double &first, double &second, double &y1, double &y2,
                         const variablePars &type0, const variablePars &type1) const;
    static bool adaptRangeZ(double &first, double &second, const double &y1, const double &y2);
    static bool adaptZ(double &first, double &second, double &y1, double &y2, double &z1, double &z2,
                const bool &variable_Y, const bool &variable_Z);
    void getNewPartitions_internal(double &first, double &second, double &y1, double &y2, double &z1, double &z2,
                                   const bool &variable1, const bool &variable2,
                                   const variablePars &type0, const variablePars &type1) const;
    void clear();
    actRange get(bool increase = false);
    bool moreAvailable() const;
};


#endif //GENERATEVIRTUALSEQUENCE_POSE_PARTITIONING_H
