#include "..\..\include\CascadeHash\BucketBuilder.h"

#include <cstdlib>

void BucketBuilder::Build(ImageData& imageData)
{
    static int cntEleInBucket[kCntBucketPerGroup]; // accumulator; the number of SIFT points in each bucket

    for (int groupIndex = 0; groupIndex < kCntBucketGroup; groupIndex++)
    {
        // initialize <cntEleInBucket>
        for (int bucketID = 0; bucketID < kCntBucketPerGroup; bucketID++)
        {
            cntEleInBucket[bucketID] = 0;
        }
        // count the number of SIFT points falling into each bucket
        for (int dataIndex = 0; dataIndex < imageData.cntPoint; dataIndex++)
        {
            cntEleInBucket[imageData.bucketIDList[groupIndex][dataIndex]]++;
        }
        // allocate space for <imageData.bucketList>
        for (int bucketID = 0; bucketID < kCntBucketPerGroup; bucketID++)
        {
            imageData.cntEleInBucket[groupIndex][bucketID] = cntEleInBucket[bucketID];
            imageData.bucketList[groupIndex][bucketID] = (int*)malloc(sizeof(int) * cntEleInBucket[bucketID]);

            cntEleInBucket[bucketID] = 0;
        }
        // assign the index of each SIFT point to <imageData.bucketList>
        for (int dataIndex = 0; dataIndex < imageData.cntPoint; dataIndex++)
        {
            int bucketID = imageData.bucketIDList[groupIndex][dataIndex];
            imageData.bucketList[groupIndex][bucketID][cntEleInBucket[bucketID]++] = dataIndex;
        }
    }
}
