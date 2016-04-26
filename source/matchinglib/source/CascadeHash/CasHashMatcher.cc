#include "../../include/CascadeHash/CasHashMatcher.h"

#include <cstring>


#include <nmmintrin.h>
#include <bitset>

MatchList& CasHashMatcher::MatchSpFast(const ImageData& imageData_1, const ImageData& imageData_2)
{
    // initialize <matchList>
    matchList.clear();

	//Check if the CPU instruction popcnt is available
	int cpuInfo[4];
	__cpuid(cpuInfo,0x00000001);
	std::bitset<32> f_1_ECX_ = cpuInfo[2];
	static bool popcnt_available = f_1_ECX_[23];
	static const unsigned char BitsSetTable256[256] = 
	{
	#   define B2(n) n,     n+1,     n+1,     n+2
	#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
	#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
		B6(0), B6(1), B6(1), B6(2)
	};

    // try to find the corresponding SIFT point in <imageData_2> for each point in <imageData_1>
    for (int dataIndex_1 = 0; dataIndex_1 < imageData_1.cntPoint; dataIndex_1++)
    {
        // fetch candidate SIFT points from the buckets in each group
        // only the bucket with the same <bucketID> with the current query point will be fetched
        int cntCandidate = 0;
        for (int groupIndex = 0; groupIndex < kCntBucketGroup; groupIndex++)
        {
            // obtain the <bucketID> for the query SIFT point
            int bucketID = imageData_1.bucketIDList[groupIndex][dataIndex_1];
            // obtain the pointer to the corresponding bucket
            const int* bucketPtr = imageData_2.bucketList[groupIndex][bucketID];

            for (int eleIndex = 0; eleIndex < imageData_2.cntEleInBucket[groupIndex][bucketID]; eleIndex++)
            {
                candidateIndexList[cntCandidate++] = bucketPtr[eleIndex]; // fetch candidates from the bucket
                dataIndexUsedList[bucketPtr[eleIndex]] = false; // indicate this candidate has not been added to <candidateIndexListTop>
            }
        }
        
        // calculate the Hamming distance of all candidates based on the CompHash code
        uint8_t* distPtr = distList;
        CompHashDataPtr ptr_1 = imageData_1.compHashDataPtrList[dataIndex_1];

		if(popcnt_available)
		{
			for (int candidateIndex = 0; candidateIndex < cntCandidate; candidateIndex++)
			{
				CompHashDataPtr ptr_2 = imageData_2.compHashDataPtrList[candidateIndexList[candidateIndex]];
                *(distPtr++) = (uint8_t)_mm_popcnt_u64(ptr_1[0] ^ ptr_2[0]) + _mm_popcnt_u64(ptr_1[1] ^ ptr_2[1]);
			}
		}
		else
		{
			uint8_t *bytesOf64Bit11 = reinterpret_cast<uint8_t*>(ptr_1);
			uint8_t *bytesOf64Bit12 = reinterpret_cast<uint8_t*>(ptr_1 + 1);
			for (int candidateIndex = 0; candidateIndex < cntCandidate; candidateIndex++)
			{
				CompHashDataPtr ptr_2 = imageData_2.compHashDataPtrList[candidateIndexList[candidateIndex]];
				uint8_t *bytesOf64Bit21 = reinterpret_cast<uint8_t*>(ptr_2);
				uint8_t *bytesOf64Bit22 = reinterpret_cast<uint8_t*>(ptr_2 + 1);
				distPtr++;
				for(uint8_t bytecnt = 0; bytecnt < 8; bytecnt++)
				{
					*(distPtr) = (uint8_t)BitsSetTable256[*(bytesOf64Bit11 + bytecnt) ^ *(bytesOf64Bit21 + bytecnt)] +
								(uint8_t)BitsSetTable256[*(bytesOf64Bit12 + bytecnt) ^ *(bytesOf64Bit22 + bytecnt)];
				}
			}
		}

        // re-assign candidates to a linked list based on their Hamming distance
        memset(linkListLen, 0, sizeof(int) * (kDimHashData + 1));
        for (int candidateIndex = 0; candidateIndex < cntCandidate; candidateIndex++)
        {
            uint8_t hashDist = distList[candidateIndex];
            linkList[hashDist][linkListLen[hashDist]++] = candidateIndexList[candidateIndex];
        }

        // add top-ranked candidates in Hamming distance to <candidateIndexListTop>
        int cntCandidateFound = 0;
        for (int hashDist = 0; hashDist <= kDimHashData; hashDist++)
        {
            for (int linkListIndex = linkListLen[hashDist] - 1; linkListIndex >= 0; linkListIndex--)
            {
                int dataIndex_2 = linkList[hashDist][linkListIndex];

                if (!dataIndexUsedList[dataIndex_2])
                {
                    dataIndexUsedList[dataIndex_2] = true; // avoid selecting same candidate multiple times
                    candidateIndexListTop[cntCandidateFound++] = dataIndex_2; // add candidate to <candidateIndexListTop>
                }

                // if enough candidates have been selected, then break
                if (cntCandidateFound >= kCntCandidateTopMax)
                {
                    break;
                }
            }

            // if enough candidates have been selected, then break
            if (cntCandidateFound >= kCntCandidateTopMin)
            {
                break;
            }
        }

        // calculate Euclidean distance for candidates in <candidateIndexListTop>
        for (int candidateIndex = 0; candidateIndex < cntCandidateFound; candidateIndex++)
        {
            int dataIndex_2 = candidateIndexListTop[candidateIndex];
            int distEuclid = 0;

            // fetch the pointers to two SIFT feature vectors
            SiftDataPtr ptr_1 = &(imageData_1.siftDataPtrList[dataIndex_1][kDimSiftData - 1]);
            SiftDataPtr ptr_2 = &(imageData_2.siftDataPtrList[dataIndex_2][kDimSiftData - 1]);

            for (int dimSiftIndex = kDimSiftData - 1; dimSiftIndex >= 0; dimSiftIndex--)
            {
                int diff = *(ptr_1--) - *(ptr_2--);
                distEuclid += diff * diff;
            }
            candidateDistListTop[candidateIndex] = distEuclid;
        }

        // find the top-2 candidates with minimal Euclidean distance
        double minVal_1 = 0.0;
        int minValInd_1 = -1;
        double minVal_2 = 0.0;
        int minValInd_2 = -1;
        for (int candidateIndex = 0; candidateIndex < cntCandidateFound; candidateIndex++)
        {
            if (minValInd_2 == -1 || minVal_2 > candidateDistListTop[candidateIndex])
            {
                minVal_2 = candidateDistListTop[candidateIndex];
                minValInd_2 = candidateIndexListTop[candidateIndex];
            }
            if (minValInd_1 == -1 || minVal_1 > minVal_2)
            {
                minVal_1 = minVal_1 + minVal_2;
                minVal_2 = minVal_1 - minVal_2;
                minVal_1 = minVal_1 - minVal_2;
                minValInd_1 = minValInd_1 + minValInd_2;
                minValInd_2 = minValInd_1 - minValInd_2;
                minValInd_1 = minValInd_1 - minValInd_2;
            }
        }

        // apply the threshold for matching rejection
        if (minVal_1 < minVal_2 * 0.32)
        {
            matchList.push_back(std::pair<int, int>(dataIndex_1, minValInd_1));
        }
    }

    return matchList;
}
