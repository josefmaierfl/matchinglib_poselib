/**********************************************************************************************************
 FILE: vfcMatches.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: February 2016

 LOCATION: TechGate Vienna, Donau-City-Straße 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: Interface for filtering matches with the VFC algorithm
**********************************************************************************************************/
#include "../include/matchinglib/vfcMatches.h"
#include "../include/vfc.h"

using namespace std;

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

/* --------------------- Functions --------------------- */

/* This function filters a given set of feature matches utilizing the vector field consensus (VFC) algorithm
 *
 * vector<KeyPoint> keypL						Input  -> The matched keypoints in the left (first) image
 * vector<KeyPoint> keypR						Input  -> The matched keypoints in the right (second) image
 * vector<cv::DMatch> matches_in				Input  -> The matches that should be filtered
 * vector<DMatch> matches_out					Output -> Fitered matches
 *
 * Return value:								 0:		  Everything ok
 *												-1:		  Too less matches for refinement
 *												-2:		  Maybe VFC failed
 */
int filterWithVFC(std::vector<cv::KeyPoint> const& keypL, std::vector<cv::KeyPoint> const& keypR,
                  std::vector<cv::DMatch> const& matches_in, std::vector<cv::DMatch> & matches_out)
{
    //Clear variables
    matches_out.clear();
    //Remove mismatches by vector field consensus (VFC)
    // preprocess data format
    vector<cv::Point2f> X;
    vector<cv::Point2f> Y;
    X.clear();
    Y.clear();
    for (unsigned int i = 0; i < matches_in.size(); i++) {
        int idx1 = matches_in[i].queryIdx;
        int idx2 = matches_in[i].trainIdx;
        X.push_back(keypL[idx1].pt);
        Y.push_back(keypR[idx2].pt);
    }

    // main process
    VFC myvfc;
    if(!myvfc.setData(X, Y))
    {
        cout << "Too less matches for refinement with VFC!" << endl;
        return -1; //Too less matches for refinement
    }
    myvfc.optimize();
    vector<int> matchIdx = myvfc.obtainCorrectMatch();

    for (unsigned int i = 0; i < matchIdx.size(); i++) {
        int idx = matchIdx[i];
        matches_out.push_back(matches_in[idx]);
    }

    if((double)matches_out.size() / (double)matches_in.size() < 0.1)
        return -2; //Maybe VFC failed

    return 0;
}
