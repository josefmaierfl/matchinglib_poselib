#ifndef BUCKETBUILDER_H_INCLUDED
#define BUCKETBUILDER_H_INCLUDED

#include "Share.h"

// this class is used to generate bucket list based on each SIFT point's bucket index
class BucketBuilder
{
public:
    // generate bucket list based on each SIFT point's bucket index
    static void Build(ImageData& imageData);
};

#endif // BUCKETBUILDER_H_INCLUDED
