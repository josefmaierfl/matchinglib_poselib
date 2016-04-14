#ifndef CASHASHMATCHER_H_INCLUDED
#define CASHASHMATCHER_H_INCLUDED

#include "Share.h"

// this class is used to perform SIFT points matching using CasHash Matching Technique
class CasHashMatcher
{
public:
    // return a list of matched SIFT points between two input images
    MatchList& MatchSpFast(const ImageData& imageData_1, const ImageData& imageData_2);

private:
    MatchList matchList; // list of matched SIFT points
    int candidateIndexList[kMaxCntPoint]; // indexes of candidate SIFT points
    bool dataIndexUsedList[kMaxCntPoint]; // usage flag list of SIFT points; to indicate whether this point has been added to <candidateIndexListTop>
    uint8_t distList[kMaxCntPoint]; // Hamming distance of candidate SIFT points to the query point
    int linkList[kDimHashData + 1][kMaxCntPoint]; // re-assign candidate SIFT points according to their Hamming distance
    int linkListLen[kDimHashData + 1]; // the number of candidates with Hamming distance of [0, 1, ..., 128]
    int candidateIndexListTop[kCntCandidateTopMax]; // indexes of final candidate SIFT points
    double candidateDistListTop[kCntCandidateTopMax]; // Euclidean distance of final candidate SIFT points
};

#endif // CASHASHMATCHER_H_INCLUDED
