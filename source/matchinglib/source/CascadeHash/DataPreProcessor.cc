#include "..\..\include\CascadeHash\DataPreProcessor.h"

#include <cstdlib>

void DataPreProcessor::PreProcess(std::vector<ImageData>& imageDataList)
{
    int cntSiftVec = 0; // accumulator; the number of SIFT feature vectors in <imageDataList>

    // allocate space for sum and average vectors of SIFT feature
    SiftDataPtr siftVecSum = (int*)malloc(sizeof(int) * kDimSiftData);
    SiftDataPtr siftVecAve = (int*)malloc(sizeof(int) * kDimSiftData);

    // initialize the sum vector
    for (int dimIndex = 0; dimIndex < kDimSiftData; dimIndex++)
    {
        siftVecSum[dimIndex] = 0;
    }

    // calculate the sum vector by adding up all feature vectors
    for (size_t imageDataIndex = 0; imageDataIndex < imageDataList.size(); imageDataIndex++)
    {
        ImageData& imageDataDst = imageDataList[imageDataIndex];

        for (int dataIndex = 0; dataIndex < imageDataDst.cntPoint; dataIndex++)
        {
            SiftDataPtr siftVecNew = imageDataDst.siftDataPtrList[dataIndex]; // obtain the vector pointer for current SIFT point

            cntSiftVec++;
            for (int dimIndex = 0; dimIndex < kDimSiftData; dimIndex++)
            {
                siftVecSum[dimIndex] += siftVecNew[dimIndex];
            }
        }
    }

    // calculate the average vector
    for (int dimIndex = 0; dimIndex < kDimSiftData; dimIndex++)
    {
        siftVecAve[dimIndex] = siftVecSum[dimIndex] / cntSiftVec;
    }

    // substract the average vector from each feature vector
    for (size_t imageDataIndex = 0; imageDataIndex < imageDataList.size(); imageDataIndex++)
    {
        ImageData& imageDataDst = imageDataList[imageDataIndex];

        for (int dataIndex = 0; dataIndex < imageDataDst.cntPoint; dataIndex++)
        {
            SiftDataPtr siftVecNew = imageDataDst.siftDataPtrList[dataIndex]; // obtain the vector pointer for current SIFT point

            for (int dimIndex = 0; dimIndex < kDimSiftData; dimIndex++)
            {
                siftVecNew[dimIndex] -= siftVecAve[dimIndex];
            }
        }
    }

	free(siftVecSum);
	free(siftVecAve);
}
