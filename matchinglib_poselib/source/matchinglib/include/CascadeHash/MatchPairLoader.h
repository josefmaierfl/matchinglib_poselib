#ifndef MATCHPAIRLOADER_H_INCLUDED
#define MATCHPAIRLOADER_H_INCLUDED

#include <vector>

// this class is used to determine the SIFT point matching operation should be performed on which image pairs
class MatchPairLoader
{
public:
    // complete matching; all image pairs are involved 
    void LoadMatchPairList(const int cntImage);
    // selected matching; only involve image pairs specified in the file
    void LoadMatchPairList(const char* filePath);
    // return the list of image indexes to perform SIFT point matching
    std::vector<int> GetMatchPairList(const int imageIndexQuery);

//private:
    // image pairs to be matched
    std::vector<std::vector<int> > matchPairList;
};

#endif // MATCHPAIRLOADER_H_INCLUDED
