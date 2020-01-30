#include "../../include/CascadeHash/MatchPairLoader.h"

#include <fstream>

#include <cstring>
#include <cstdlib>
#include <cassert>

#include <stdint.h>

void MatchPairLoader::LoadMatchPairList(const int cntImage)
{
    std::vector<int> matchPairListNew;

    // initialize <matchPairList>
    matchPairList.clear();

    // each image will be matched with all previous images
    for (int imageIndexQuery = 0; imageIndexQuery < cntImage; imageIndexQuery++)
    {
        matchPairListNew.clear();
        for (int imageIndexMatch = 0; imageIndexMatch < imageIndexQuery; imageIndexMatch++)
        {
            matchPairListNew.push_back(imageIndexMatch);
        }
        matchPairList.push_back(matchPairListNew);
    }
}

void MatchPairLoader::LoadMatchPairList(const char* filePath)
{
    std::ifstream fin(filePath);
    std::vector<std::vector<int> > matchPairListTmp;

    // read image pairs from file and temporally store them in <matchPairListTmp>
    int matchListLen;
    fin >> matchListLen;
    matchPairListTmp.clear();
    while (true)
    {
        int imageIndexQuery;
        int imageIndexMatch;

        fin >> imageIndexQuery;
        if (fin.eof())
        {
            fin.close();
            break;
        }
        else
        {
            std::vector<int> matchPairVecNew(matchListLen - 1);
            for (int index = 0; index < matchListLen; index++)
            {
                fin >> imageIndexMatch;
                if (index != 0) // the first element is <imageIndexQuery> itself, so abandon it
                {
                    matchPairVecNew[index - 1] = imageIndexMatch;
                }
            }
            assert(static_cast<int>(matchPairListTmp.size()) == imageIndexQuery); // make sure <matchPairVecNew> will be appended in the right position
            matchPairListTmp.push_back(matchPairVecNew);
        }
    }
    
    // allocate space for <matchPairMat>
    int cntImage = static_cast<int>(matchPairListTmp.size());
    uint8_t* matchPairMat = (uint8_t*)malloc(sizeof(uint8_t) * cntImage * cntImage);
    
    // set 1/0 values in <matchPairMat>
    // for each required image pair, the corresponding value is set to 1; otherwise, the value is 0
    memset(matchPairMat, 0, cntImage * cntImage);
    for (int imageIndex = 0; imageIndex < cntImage; imageIndex++)
    {
        const std::vector<int>& matchPairVecSel = matchPairListTmp[imageIndex];
        for (std::vector<int>::const_iterator iter = matchPairVecSel.begin(); iter != matchPairVecSel.end(); iter++)
        {
            matchPairMat[imageIndex * cntImage + *iter] = 1;
            matchPairMat[*iter * cntImage + imageIndex] = 1;
        }
    }
    
    // convert <matchPairMat> to <matchPairList>
    matchPairList.clear();
    for (int imageIndexQuery = 0; imageIndexQuery < cntImage; imageIndexQuery++)
    {
        std::vector<int> matchPairListNew(0);
        for (int imageIndexMatch = 0; imageIndexMatch < imageIndexQuery; imageIndexMatch++)
        {
            if (matchPairMat[imageIndexQuery * cntImage + imageIndexMatch] == 1)
            {
                matchPairListNew.push_back(imageIndexMatch);
            }
        }
        matchPairList.push_back(matchPairListNew);
    }

    // free allocated space for <matchPairMat>
    free(matchPairMat);
}

std::vector<int> MatchPairLoader::GetMatchPairList(const int imageIndexQuery)
{
    // return the corresponding <matchPairList>
    return matchPairList[imageIndexQuery];
}
