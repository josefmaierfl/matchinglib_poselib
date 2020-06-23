//
// Created by maierj on 6/22/20.
//
#include "pose_partitioning.h"

using namespace std;

partitionPoses::partitionPoses(std::pair<double,double> txRange_init_,
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
                               const size_t &nr_stereoConfs_):
        txRange_init(move(txRange_init_)),
        tyRange_init(move(tyRange_init_)),
        tzRange_init(move(tzRange_init_)),
        rollRange_init(move(rollRange_init_)),
        pitchRange_init(move(pitchRange_init_)),
        yawRange_init(move(yawRange_init_)),
        nr_stereoConfs(nr_stereoConfs_),
        txVariable(txVariable_),
        tyVariable(tyVariable_),
        tzVariable(tzVariable_),
        rollVariable(rollVariable_),
        pitchVariable(pitchVariable_),
        yawVariable(yawVariable_){
    nr_var = 0;
    auto siXYZ = DBL_MAX, siRPY = DBL_MAX;
    if(txVariable) {
        nr_var++;
        txSize = txRange_init.second - txRange_init.first;
        siXYZ = txSize;
        if(txRange_init.first < 0 && txRange_init.second > 0){
            siXYZ /= 4.;
        }else{
            siXYZ /= 2.;
        }
        parts.emplace_back(variablePars::TXv, txSize, 1, true);
    }
    if(tyVariable){
        nr_var++;
        tySize = tyRange_init.second - tyRange_init.first;
        double tmp = tySize;
        if(tyRange_init.first < 0 && tyRange_init.second > 0){
            tmp /= 4.;
        }else{
            tmp /= 2.;
        }
        if(tmp < siXYZ) siXYZ = tmp;
        parts.emplace_back(variablePars::TYv, tySize, 1, true);
    }
    if(tzVariable){
        nr_var++;
        tzSize = tzRange_init.second - tzRange_init.first;
        double tmp = tzSize;
        if(tzRange_init.first < 0 && tzRange_init.second > 0){
            tmp /= 4.;
        }else{
            tmp /= 2.;
        }
        if(tmp < siXYZ) siXYZ = tmp;
        parts.emplace_back(variablePars::TZv, tzSize, 1, true);
    }
    if(rollVariable){
        nr_var++;
        rollSize = rollRange_init.second - rollRange_init.first;
        siRPY = rollSize;
        if(rollRange_init.first < 0 && rollRange_init.second > 0){
            siRPY /= 4.;
        }else{
            siRPY /= 2.;
        }
        parts.emplace_back(variablePars::ROLLv, rollSize, 1, false);
    }
    if(pitchVariable){
        nr_var++;
        pitchSize = pitchRange_init.second - pitchRange_init.first;
        double tmp = pitchSize;
        if(pitchRange_init.first < 0 && pitchRange_init.second > 0){
            tmp /= 4.;
        }else{
            tmp /= 2.;
        }
        if(tmp < siRPY) siRPY = tmp;
        parts.emplace_back(variablePars::PITCHv, pitchSize, 1, false);
    }
    if(yawVariable){
        nr_var++;
        yawSize = yawRange_init.second - yawRange_init.first;
        double tmp = yawSize;
        if(yawRange_init.first < 0 && yawRange_init.second > 0){
            tmp /= 4.;
        }else{
            tmp /= 2.;
        }
        if(tmp < siRPY) siRPY = tmp;
        parts.emplace_back(variablePars::YAWv, yawSize, 1, false);
    }
//    if(nr_stereoConfs > nr_var){
//        nr_partitions = nr_stereoConfs;
//    }else{
//        nr_partitions = nr_var;
//    }
    sort(parts.begin(), parts.end(),
         [](const tuple<variablePars, double, size_t, bool> &first,
            const tuple<variablePars, double, size_t, bool> &second){
             return std::get<1>(first) > std::get<1>(second);
         });
    size_t mul = 1, nr[2] = {0, 0};
    bool found[2];
    found[0] = !nearZero(siXYZ);
    found[1] = !nearZero(siRPY);
    for(auto &i: parts){
        if(!found[0] && !found[1]) break;
        if(found[0] && std::get<3>(i)){
            nr[0] = static_cast<size_t>(ceil(std::get<1>(i) / siXYZ));
            found[0] = false;
        }
        if(found[1] && !std::get<3>(i)){
            nr[1] = static_cast<size_t>(ceil(std::get<1>(i) / siRPY));
            found[1] = false;
        }
    }
    nr_partitions = nr[0] + nr[0] * nr[1];
    for(auto &i: parts){
        size_t &p0 = std::get<2>(i);
        size_t p1 = p0 + 1;
        size_t mul_new = p1 * mul / p0;
        if(mul_new <= nr_partitions){
            p0 = p1;
            mul = mul_new;
        }else{
            break;
        }
    }
}

void partitionPoses::clear(){
    txRangei.clear();
    tyRangei.clear();
    tzRangei.clear();
    rollRangei.clear();
    pitchRangei.clear();
    yawRangei.clear();
}

actRange partitionPoses::get(bool increase){
    if(increase){
        vec_pointer++;
        if(vec_pointer >= static_cast<int>(txRangei.size())) vec_pointer = 0;
        if(vec_pointer >= 0){
            userng.txRange = txRangei[vec_pointer];
            userng.tyRange = tyRangei[vec_pointer];
            userng.tzRange = tzRangei[vec_pointer];
            userng.rollRange = rollRangei[vec_pointer];
            userng.pitchRange = pitchRangei[vec_pointer];
            userng.yawRange = yawRangei[vec_pointer];
        }else{
            userng.txRange = txRange_init;
            userng.tyRange = tyRange_init;
            userng.tzRange = tzRange_init;
            userng.rollRange = rollRange_init;
            userng.pitchRange = pitchRange_init;
            userng.yawRange = yawRange_init;
        }
    }else if(userng.empty()){
        if(vec_pointer >= 0){
            userng.txRange = txRangei[vec_pointer];
            userng.tyRange = tyRangei[vec_pointer];
            userng.tzRange = tzRangei[vec_pointer];
            userng.rollRange = rollRangei[vec_pointer];
            userng.pitchRange = pitchRangei[vec_pointer];
            userng.yawRange = yawRangei[vec_pointer];
        }else{
            userng.txRange = txRange_init;
            userng.tyRange = tyRange_init;
            userng.tzRange = tzRange_init;
            userng.rollRange = rollRange_init;
            userng.pitchRange = pitchRange_init;
            userng.yawRange = yawRange_init;
        }
    }
    return userng;
}

bool partitionPoses::moreAvailable() const{
    return vec_pointer < static_cast<int>(txRangei.size()) - 1;
}

void partitionPoses::partition() {
    if(nr_var < 2 || nr_stereoConfs == 1){
        txRangei.push_back(txRange_init);
        tyRangei.push_back(tyRange_init);
        tzRangei.push_back(tzRange_init);
        rollRangei.push_back(rollRange_init);
        pitchRangei.push_back(pitchRange_init);
        yawRangei.push_back(yawRange_init);
        return;
    }

    vector<std::pair<double,double>> txRangei_tmp;
    vector<std::pair<double,double>> tyRangei_tmp;
    vector<std::pair<double,double>> tzRangei_tmp;
    vector<std::pair<double,double>> rollRangei_tmp;
    vector<std::pair<double,double>> pitchRangei_tmp;
    vector<std::pair<double,double>> yawRangei_tmp;
    for(auto &i: parts){
        if(std::get<2>(i) == 1) break;
        const size_t &ps = std::get<2>(i);
        if(std::get<0>(i) == variablePars::TXv){
            const double partsX = abs(txSize / static_cast<double>(ps));
            double first, second;
            for(size_t j = 0; j < ps; ++j){
                first = txRange_init.first + static_cast<double>(j) * partsX;
                second = txRange_init.first + static_cast<double>(j + 1) * partsX;
                double y1 = tyRange_init.first;
                double y2 = tyRange_init.second;
                double z1 = tzRange_init.first;
                double z2 = tzRange_init.second;
                getNewPartitions_internal(first, second, y1, y2, z1, z2,
                                          tyVariable, tzVariable,
                                          variablePars::TXv, variablePars::TYv);
                txRangei_tmp.emplace_back(first, second);
                tyRangei_tmp.emplace_back(y1, y2);
                tzRangei_tmp.emplace_back(z1, z2);
            }
        }else if(std::get<0>(i) == variablePars::TYv){
            const double partsX = abs(tySize / static_cast<double>(ps));
            double first, second;
            for(size_t j = 0; j < ps; ++j) {
                first = tyRange_init.first + static_cast<double>(j) * partsX;
                second = tyRange_init.first + static_cast<double>(j + 1) * partsX;
                double x1 = txRange_init.first;
                double x2 = txRange_init.second;
                double z1 = tzRange_init.first;
                double z2 = tzRange_init.second;
                getNewPartitions_internal(first, second, x1, x2, z1, z2,
                                          txVariable, tzVariable,
                                          variablePars::TYv, variablePars::TXv);
                tyRangei_tmp.emplace_back(first, second);
                txRangei_tmp.emplace_back(x1, x2);
                tzRangei_tmp.emplace_back(z1, z2);
            }
        }else if(std::get<0>(i) == variablePars::TZv){
            const double partsX = abs(tzSize / static_cast<double>(ps));
            double first, second;
            for(size_t j = 0; j < ps; ++j) {
                first = tzRange_init.first + static_cast<double>(j) * partsX;
                second = tzRange_init.first + static_cast<double>(j + 1) * partsX;
                double x1 = txRange_init.first;
                double x2 = txRange_init.second;
                double y1 = tyRange_init.first;
                double y2 = tyRange_init.second;
                getNewPartitions_internal(first, second, x1, x2, y1, y2,
                                          txVariable, tzVariable,
                                          variablePars::TYv, variablePars::TXv);
                tzRangei_tmp.emplace_back(first, second);
                txRangei_tmp.emplace_back(x1, x2);
                tyRangei_tmp.emplace_back(y1, y2);
            }
        }else if(std::get<0>(i) == variablePars::ROLLv){
            const double partsX = abs(rollSize / static_cast<double>(ps));
            double first, second;
            for(size_t j = 0; j < ps; ++j) {
                first = rollRange_init.first + static_cast<double>(j) * partsX;
                second = rollRange_init.first + static_cast<double>(j + 1) * partsX;
                rollRangei_tmp.emplace_back(first, second);
                pitchRangei_tmp.push_back(pitchRange_init);
                yawRangei_tmp.push_back(yawRange_init);
            }
        }else if(std::get<0>(i) == variablePars::PITCHv){
            const double partsX = abs(pitchSize / static_cast<double>(ps));
            double first, second;
            for(size_t j = 0; j < ps; ++j) {
                first = pitchRange_init.first + static_cast<double>(j) * partsX;
                second = pitchRange_init.first + static_cast<double>(j + 1) * partsX;
                pitchRangei_tmp.emplace_back(first, second);
                rollRangei_tmp.push_back(rollRange_init);
                yawRangei_tmp.push_back(yawRange_init);
            }
        }else if(std::get<0>(i) == variablePars::YAWv) {
            const double partsX = abs(yawSize / static_cast<double>(ps));
            double first, second;
            for (size_t j = 0; j < ps; ++j) {
                first = yawRange_init.first + static_cast<double>(j) * partsX;
                second = yawRange_init.first + static_cast<double>(j + 1) * partsX;
                yawRangei_tmp.emplace_back(first, second);
                rollRangei_tmp.push_back(rollRange_init);
                pitchRangei_tmp.push_back(pitchRange_init);
            }
        }
    }

    if(txRangei_tmp.empty() && rollRangei_tmp.empty()){
        txRangei.push_back(txRange_init);
        tyRangei.push_back(tyRange_init);
        tzRangei.push_back(tzRange_init);
        rollRangei.push_back(rollRange_init);
        pitchRangei.push_back(pitchRange_init);
        yawRangei.push_back(yawRange_init);
    }else{
        available = true;
        if(txRangei_tmp.empty()){
            txRangei.push_back(txRange_init);
            tyRangei.push_back(tyRange_init);
            tzRangei.push_back(tzRange_init);
            rollRangei.push_back(rollRange_init);
            pitchRangei.push_back(pitchRange_init);
            yawRangei.push_back(yawRange_init);
        }else {
            for (size_t i = 0; i < txRangei_tmp.size(); i++) {
                txRangei.push_back(txRangei_tmp[i]);
                tyRangei.push_back(tyRangei_tmp[i]);
                tzRangei.push_back(tzRangei_tmp[i]);
                rollRangei.push_back(rollRange_init);
                pitchRangei.push_back(pitchRange_init);
                yawRangei.push_back(yawRange_init);
            }
        }
        for (size_t i = 0; i < rollRangei_tmp.size(); i++) {
            txRangei.push_back(txRange_init);
            tyRangei.push_back(tyRange_init);
            tzRangei.push_back(tzRange_init);
            rollRangei.push_back(rollRangei_tmp[i]);
            pitchRangei.push_back(pitchRangei_tmp[i]);
            yawRangei.push_back(yawRangei_tmp[i]);
        }
        for (size_t i = 0; i < txRangei_tmp.size(); i++) {
            for (size_t j = 0; j < rollRangei_tmp.size(); j++) {
                txRangei.push_back(txRangei_tmp[i]);
                tyRangei.push_back(tyRangei_tmp[i]);
                tzRangei.push_back(tzRangei_tmp[i]);
                rollRangei.push_back(rollRangei_tmp[j]);
                pitchRangei.push_back(pitchRangei_tmp[j]);
                yawRangei.push_back(yawRangei_tmp[j]);
            }
        }
        if(txRangei.size() > 2) {
            vector<std::size_t> idx;
            shuffleVector(idx, txRangei.size());
            reOrderVector(txRangei, idx);
            reOrderVector(tyRangei, idx);
            reOrderVector(tzRangei, idx);
            reOrderVector(rollRangei, idx);
            reOrderVector(pitchRangei, idx);
            reOrderVector(yawRangei, idx);
        }
    }
}

bool partitionPoses::adaptRange(double &first, double &second, double &y1, double &y2,
                                const variablePars &type0, const variablePars &type1) const{
    bool first_adapted = false;
    if(y2 > 0 && y2 + 1e-4 > abs(first)){
        if(y1 > 0 && y1 + 1e-4 > abs(first)){
            const double diff = y2 - y1;
            first -= y1 - abs(first) + 0.1 * diff;
            y2 -= 0.93 * diff;
            if(type0 == variablePars::TXv) {
                first = max(txRange_init.first, first);
            }else if(type0 == variablePars::TYv) {
                first = max(tyRange_init.first, first);
            }else{
                first = max(tzRange_init.first, first);
            }
            first_adapted = true;
        }else{
            y2 -= y2 - abs(first) + 0.1 * (y2 - y1);
        }
    }
    if(nearZero(y1 - first)){
        y1 += 0.1 * (y2 - y1);
    }
    if((first > 0 || nearZero(first)) && first > y1 && type1 != variablePars::TZv){
        const double diff = 0.1 * (y2 - y1);
        if(y1 > 0 || nearZero(y1) || y1 - diff > 0){
            if(type0 == variablePars::TXv) {
                first = txRange_init.first;
            }else if(type0 == variablePars::TYv) {
                first = tyRange_init.first;
            }else{
                first = tzRange_init.first;
            }
        }else{
            first = y1 - diff;
            if(type0 == variablePars::TXv) {
                first = max(txRange_init.first, first);
            }else if(type0 == variablePars::TYv) {
                first = max(tyRange_init.first, first);
            }else{
                first = max(tzRange_init.first, first);
            }
        }
        if((second > 0 || nearZero(second)) && second >= y2){
            second = y2 - 0.1 * (y2 - y1);
            if(type0 == variablePars::TXv) {
                second = min(txRange_init.second, second);
            }else if(type0 == variablePars::TYv) {
                second = min(tyRange_init.second, second);
            }else{
                second = min(tzRange_init.second, second);
            }
        }
        first_adapted = true;
    }
    if(first > second){
        const double tmp = first;
        first = second;
        second = tmp;
    }
    if(y1 > y2){
        const double tmp = y1;
        y1 = y2;
        y2 = tmp;
    }
    return first_adapted;
}

bool partitionPoses::adaptRangeFixed(double &first, double &second, double &y1, double &y2,
                                     const variablePars &type0, const variablePars &type1) const{
    bool first_adapted = false;
    if(y2 > 0 && y2 + 1e-4 > abs(first)){
        const double diff = y2 - y1;
        first -= y2 - abs(first) + 0.1 * diff;
        if(type0 == variablePars::TXv) {
            first = max(txRange_init.first, first);
        }else if(type0 == variablePars::TYv) {
            first = max(tyRange_init.first, first);
        }else{
            first = max(tzRange_init.first, first);
        }
        first_adapted = true;
    }
    if(nearZero(y1 - first)){
        first += 0.05 * (y2 - y1);
        if(type0 == variablePars::TXv) {
            first = max(txRange_init.first, first);
        }else if(type0 == variablePars::TYv) {
            first = max(tyRange_init.first, first);
        }else{
            first = max(tzRange_init.first, first);
        }
        first_adapted = true;
    }
    if((first > 0 || nearZero(first)) && first > y1 && type1 != variablePars::TZv){
        const double diff = 0.1 * (y2 - y1);
        if(y1 > 0 || nearZero(y1) || y1 - diff > 0){
            if(type0 == variablePars::TXv) {
                first = txRange_init.first;
            }else if(type0 == variablePars::TYv) {
                first = tyRange_init.first;
            }else{
                first = tzRange_init.first;
            }
        }else{
            first = y1 - diff;
            if(type0 == variablePars::TXv) {
                first = max(txRange_init.first, first);
            }else if(type0 == variablePars::TYv) {
                first = max(tyRange_init.first, first);
            }else{
                first = max(tzRange_init.first, first);
            }
        }
        if((second > 0 || nearZero(second)) && second >= y2){
            second = y2 - 0.1 * (y2 - y1);
            if(type0 == variablePars::TXv) {
                second = min(txRange_init.second, second);
            }else if(type0 == variablePars::TYv) {
                second = min(tyRange_init.second, second);
            }else{
                second = min(tzRange_init.second, second);
            }
        }
        first_adapted = true;
    }
    if(first > second){
        const double tmp = first;
        first = second;
        second = tmp;
    }
    return first_adapted;
}

bool partitionPoses::adaptRangeZ(double &first, double &second, const double &y1, const double &y2) {
    bool first_adapted = false;
    double y12 = max(abs(y1), abs(y2));
    const double diff = abs(y2 - y1);
    if(y12 + 1e-4 < abs(first)){
        if(first < 0){
            first = 0.1 * diff - y12;
        }else{
            first = y12 - 0.1 * diff;
        }
        first_adapted = true;
    }
    if(y12 + 1e-4 < abs(second)){
        if(first < 0){
            second = 0.1 * diff - y12;
        }else{
            second = y12 - 0.1 * diff;
        }
        second = max(second, first + 0.8 * diff);
        first_adapted = true;
    }
    if(first > second){
        const double tmp = first;
        first = second;
        second = tmp;
    }
    return first_adapted;
}

bool partitionPoses::adaptZ(double &first, double &second, double &y1, double &y2, double &z1, double &z2,
                            const bool &variable_Y, const bool &variable_Z){
    double maxV = abs(min(first, y1));
    bool first_adapted = false;
    const double diff = z2 - z1;
    if(abs(z1) >= maxV){
        if(variable_Z){
            if(z1 < 0){
                z1 = 0.1 * diff - maxV;
            }else{
                z1 = maxV - 0.1 * diff;
            }
        }else{
            if(first < 0 && abs(z1) >= abs(first)){
                first = -0.1 * diff - abs(z1);
                first_adapted = true;
            }
            if(variable_Y && y1 < first && y1 < 0 && abs(z1) >= abs(y1)){
                y1 = -0.15 * diff - abs(z1);
                first_adapted = true;
            }
        }
    }
    if(abs(z2) >= maxV){
        if(variable_Z){
            if(z1 < 0){
                z1 = 0.2 * diff - maxV;
            }else{
                z1 = maxV - 0.2 * diff;
            }
        }else{
            if(first < 0 && abs(z2) >= abs(first)){
                first = -0.1 * diff - abs(z2);
                first_adapted = true;
            }
            if(variable_Y && y1 < first && y1 < 0 && abs(z2) >= abs(y1)){
                y1 = -0.15 * diff - abs(z2);
                first_adapted = true;
            }
        }
    }
    if(first > second){
        const double tmp = first;
        first = second;
        second = tmp;
    }
    if(y1 > y2){
        const double tmp = y1;
        y1 = y2;
        y2 = tmp;
    }
    if(z1 > z2){
        const double tmp = z1;
        z1 = z2;
        z2 = tmp;
    }
    return first_adapted;
}

void partitionPoses::getNewPartitions_internal(double &first, double &second, double &y1, double &y2, double &z1,
                                               double &z2, const bool &variable1, const bool &variable2,
                                               const variablePars &type0, const variablePars &type1) const{
    if(type0 == variablePars::TZv){
        adaptRangeZ(first, second, y1, y2);
        int cnt = 0;
        bool first_adapted = false;
        do{
            first_adapted = adaptRangeZ(first, second, z1, z2);
            if(first_adapted){
                first_adapted = adaptRangeZ(first, second, y1, y2);
            }
            cnt++;
        }while(first_adapted && cnt < 5);
    }else {
        bool first_adapted1 = false;
        int cnt1 = 0;
        do {
            if (variable1) {
                adaptRange(first, second, y1, y2, type0, type1);
            } else {
                adaptRangeFixed(first, second, y1, y2, type0, type1);
            }
            if (variable2) {
                int cnt = 0;
                bool first_adapted = false;
                do {
                    first_adapted = adaptRange(first, second, z1, z2, type0, variablePars::TZv);
                    if (first_adapted) {
                        if (variable1) {
                            adaptRange(first, second, y1, y2, type0, type1);
                        } else {
                            adaptRangeFixed(first, second, y1, y2, type0, type1);
                        }
                    }
                    cnt++;
                } while (first_adapted && cnt < 5);
                cnt = 0;
                do {
                    if (variable1) {
                        first_adapted = adaptRange(y1, y2, z1, z2, type1, variablePars::TZv);
                    } else {
                        first_adapted = false;
                    }
                    if (first_adapted) {
                        if (variable1) {
                            adaptRange(first, second, y1, y2, type0, type1);
                        } else {
                            adaptRangeFixed(first, second, y1, y2, type0, type1);
                        }
                    }
                    cnt++;
                } while (first_adapted && cnt < 5);
            } else {
                bool first_adapted = false;
                int cnt = 0;
                do {
                    first_adapted = adaptRangeFixed(first, second, z1, z2, type0, variablePars::TZv);
                    if (first_adapted) {
                        if (variable1) {
                            adaptRange(first, second, y1, y2, type0, type1);
                        } else {
                            adaptRangeFixed(first, second, y1, y2, type0, type1);
                        }
                    }
                    cnt++;
                } while (first_adapted && cnt < 5);
                if (variable1) {
                    cnt = 0;
                    do {
                        first_adapted = adaptRangeFixed(y1, y2, z1, z2, type1, variablePars::TZv);
                        if (first_adapted) {
                            adaptRange(first, second, y1, y2, type0, type1);
                        }
                        cnt++;
                    } while (first_adapted && cnt < 5);
                }
            }
            first_adapted1 = adaptZ(first, second, y1, y2, z1, z2, variable1, variable2);
        }while(first_adapted1 && cnt1++ < 3);
    }
}

template<typename T, typename T1>
void reOrderVector(std::vector<T> &reOrderVec, std::vector<T1> &idxs){
    CV_Assert(reOrderVec.size() == idxs.size());

    std::vector<T> reOrderVec_tmp;
    reOrderVec_tmp.reserve(reOrderVec.size());
    for(auto& i : idxs){
        reOrderVec_tmp.push_back(reOrderVec[i]);
    }
    reOrderVec = std::move(reOrderVec_tmp);
}

template<typename T>
void shuffleVector(std::vector<T> &idxs, size_t si){
    idxs = vector<T>(si);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::shuffle(idxs.begin(), idxs.end(), std::mt19937{std::random_device{}()});
}