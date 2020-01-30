#ifndef HASHCONVERTOR_H_INCLUDED
#define HASHCONVERTOR_H_INCLUDED

#include "Share.h"

// this class is used to convert SIFT feature vectors to Hash codes and CompHash codes
class HashConvertor
{
public:
    // constructor function; to initialize private data member variables
    HashConvertor();
    // convert SIFT feature vectors to Hash codes and CompHash codes
    // also, the bucket index for each SIFT point will be determined (different bucket groups correspond to different indexes)
    void SiftDataToHashData(ImageData& imageData);

private:
    int projMatPriTr[kDimHashData][kDimSiftData]; // projection matrix of the primary hashing function
    int projMatSecTr[kCntBucketGroup][kCntBucketBit][kDimSiftData]; // projection matrix of the secondary hashing function
    int bucketBitList[kCntBucketGroup][kCntBucketBit]; // selected bits in the result of primary hashing fuction for bucket construction
    
private:
    // generate random number which follows normal distribution, with <mean = 0> and <variance = 1>
    double GetNormRand(void);
};

#endif // HASHCONVERTOR_H_INCLUDED
