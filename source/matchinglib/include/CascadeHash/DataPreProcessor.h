#ifndef DATAPREPROCESSOR_H_INCLUDED
#define DATAPREPROCESSOR_H_INCLUDED

#include "Share.h"

// this class is used to adjust each SIFT feature vector to zero-mean
class DataPreProcessor
{
public:
    // adjust each SIFT feature vector in <imageDataList> to zero-mean
    static void PreProcess(std::vector<ImageData>& imageDataList);
};

#endif // DATAPREPROCESSOR_H_INCLUDED
