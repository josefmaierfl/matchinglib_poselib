/**********************************************************************************************************
 FILE: base_matcher.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: October 2015

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for testing different matching algorithms. The class
 baseMatcher provides all functionalities necessary for before and after matching, like feature and 
 descriptor extraction, quality measurement on the final matches as well as refinement of the found
 matches. The matching algorithms themself must be implemented as a child class of this base class.
**********************************************************************************************************/

#include "GTM/base_matcher.h"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/ximgproc/sparse_match_interpolator.hpp>
#include "helper_funcs.h"
#include "io_data.h"

#include <Eigen/Core>
//#include <opencv2/core/eigen.hpp>
//#include <Eigen/Dense>
//#include <Eigen/StdVector>

#include "nanoflann.hpp"

//#include "vfc.h"

#include <bitset>

#include <fstream>

//#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "GTM/inscribeRectangle.h"

#include "imgFeatures.h"

#if __linux__
#if defined(USE_MANUAL_ANNOTATION)
#include "QMessageBox"
#endif
#else
#include "winuser.h"
#endif

using namespace cv;
using namespace std;

/* --------------------------- Defines --------------------------- */

#define LEFT_TO_RIGHT 0 //For testing the GT matches: Changes the order of images from left-right to right-left if 0

typedef Eigen::Matrix<float,Eigen::Dynamic,2, Eigen::RowMajor> EMatFloat2;
typedef nanoflann::KDTreeEigenMatrixAdaptor< EMatFloat2, nanoflann::metric_L2_Simple>  KDTree_D2float;

#if defined(USE_MANUAL_ANNOTATION)
enum SpecialKeyCode{
        NONE, SPACE, BACKSPACE, ESCAPE, CARRIAGE_RETURN, ARROW_UP, ARROW_RIGHT, ARROW_DOWN, ARROW_LEFT, PAGE_UP, PAGE_DOWN, POS1, END_KEY ,INSERT, DELETE_KEY
    };
#endif

/* --------------------- Function prototypes --------------------- */

bool readDoubleVal(ifstream & gtFromFile, const std::string& keyWord, double *value);
void getSubPixPatchDiff(const cv::Mat& patch, const cv::Mat& image, cv::Point2f &diff);
void iterativeTemplMatch(cv::InputArray patch, cv::InputArray img, int maxPatchSize, cv::Point2f & minDiff, int maxiters = INT_MAX);
#if defined(USE_MANUAL_ANNOTATION)
void on_mouse_click(int event, int x, int y, int flags, void* param);
SpecialKeyCode getSpecialKeyCode(int & val);
#endif
float BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y);
void calcErrorToSpatialGT(const cv::Point2f& perfectMatchesFirst, const cv::Point2f& perfectMatchesSecond,
						  std::vector<cv::Mat> channelsFlow, bool flowGtIsUsed, std::vector<cv::Point2f> & errvecsGT, 
						  std::vector<int> & validityValGT, const cv::Mat& homoGT, const cv::Point2f& lkp, const cv::Point2f& rkp);
void intersectPolys(std::vector<cv::Point2d> pntsPoly1, std::vector<cv::Point2d> pntsPoly2, std::vector<cv::Point2f> &pntsRes);
void findLocalMin(const Mat& patchL, const Mat& patchR, float quarterpatch, float eigthpatch, cv::Point2f &winPos, float patchsizeOrig);


/* --------------------- Functions --------------------- */


/* Constructor of class baseMatcher. 
 *
 * string _featuretype			Input  -> Feature type like FAST, SIFT, ... that are defined in the OpenCV (FeatureDetector::create).
 *										  Only feature types MSER, Dense, and SimpleBlob are not allowed.
 * string _descriptortype		Input  -> Descriptor type for filtering GT like SIFT, FREAK, ... that are defined in OpenCV.
 * string _imgsPath				Input  -> Path to the images which is necessary for loading and storing the ground truth matches
 *
 */
baseMatcher::baseMatcher(std::string _featuretype, std::string _imgsPath, std::string _descriptortype, bool refineGTM_) :
        imgsPath(move(_imgsPath)),
        GTfilterExtractor(move(_descriptortype)),
        featuretype(move(_featuretype)),
        refineGTM(refineGTM_)
{
//	memset(&qpm, 0, sizeof(matchQualParams));
//	memset(&qpr, 0, sizeof(matchQualParams));
}

/* Feature extraction in both images without filtering
 */
bool baseMatcher::detectFeatures()
{
    //Clear variables
    keypL.clear();
    keypR.clear();

    if (matchinglib::getKeypoints(imgs[0], keypL, featuretype, false, INT_MAX) != 0)
        return false;

    if(keypL.size() < 15)
    {
        return false; //Too less features detected
    }

#define showfeatures 0
#if showfeatures
    cv::Mat img1c;
    drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("Keypoints 1", img1c );
#endif

    if (matchinglib::getKeypoints(imgs[1], keypR, featuretype, false, INT_MAX) != 0)
        return false;
    if(keypR.size() < 15)
    {
        return false; //Too less features detected
    }

#if showfeatures
    cv::Mat img2c;
    drawKeypoints( imgs[1], keypR, img2c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("Keypoints 2", img2c );
    waitKey(0);
#endif
	return true;
}

/* Generates ground truth matches from the keypoints and calculates a threshold for checking
 * the correctness of matches by searching for the nearest and second nearest neighbors. The
 * threshold is then set between the largest distance of the nearest neighbors and the shortest
 * distance to the second nearest neighbors. Therefore, some keypoints might be removed, if
 * they are too close to each other so that a clear identification of the correct keypoint
 * would be impossible. Moreover, keypoints in the left image are removed that point to the 
 * same keypoint in the right image.
 *
 * Return value:				 0:		  Everything ok
 *								-1:		  Too less features detected
 */
int baseMatcher::filterInitFeaturesGT()
{
#define showfeatures 0
	Mat descriptors1, descriptors22nd;
	std::vector<cv::KeyPoint> keypR_tmp, keypR_tmp1;//Right found keypoints
	std::vector<cv::KeyPoint> keypL_tmp, keypL_tmp1;//Left found keypoints
	vector<size_t> border_dellist; //Indices of keypoints in the right image that are too near at the border
	vector<size_t> border_dellistL; //Indices of keypoints in the left image that are too near at the border
	auto h = static_cast<float>(flowGT.rows);
	auto w = static_cast<float>(flowGT.cols);
	vector<Mat> channelsFlow(3);
	vector<std::pair<size_t, float>> nearest_dist, second_nearest_dist;
    leftInlier.clear();
    rightInlier.clear();
    matchesGT.clear();

    //Split 3 channel matrix for access
    if(flowGtIsUsed)
    {
        cv::split(flowGT, channelsFlow);
    }

    //Get descriptors from left keypoints
    matchinglib::getDescriptors(imgs[0], keypL, GTfilterExtractor, descriptors1, featuretype);

    //Get descriptors from right keypoints
    matchinglib::getDescriptors(imgs[1], keypR, GTfilterExtractor, descriptors22nd, featuretype);

    //Prepare the coordinates of the keypoints for the KD-tree (must be after descriptor extractor because keypoints near the border are removed)
    EMatFloat2 eigkeypts2(keypR.size(),2);
    for(unsigned int i = 0;i<keypR.size();i++)
    {
        eigkeypts2(i,0) = keypR[i].pt.x;
        eigkeypts2(i,1) = keypR[i].pt.y;
    }

    //Generate the KD-tree index for the keypoint coordinates
    float searchradius = static_cast<float>(INITMATCHDISTANCETH_GT) * static_cast<float>(INITMATCHDISTANCETH_GT);
    const int maxLeafNum     = 20;
    const int maxDepthSearch = 32;

    Eigen::Vector2f x2e;

    KDTree_D2float keypts2idx(eigkeypts2, maxLeafNum);
    keypts2idx.index->buildIndex();
        vector<std::pair<KDTree_D2float::IndexType, float>> radius_matches;

    //Search for the ground truth matches in the right (second) image by searching for the nearest and second nearest neighbor by a radius search
    if(flowGtIsUsed)
    {
        int xd, yd;
        //Search for the ground truth matches using optical flow data
        for(size_t i = 0; i < keypL.size(); i++)
        {
            cv::Point2i hlp;
            hlp.x = static_cast<int>(floor(keypL[i].pt.x + 0.5f)); //Round to nearest integer
            hlp.y = static_cast<int>(floor(keypL[i].pt.y + 0.5f)); //Round to nearest integer
            if(channelsFlow[2].at<float>(hlp.y, hlp.x) == 1.0)
            {
                x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
                x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
            }
            else if(channelsFlow[2].at<float>(hlp.y, hlp.x) > 1.0) //Check if the filled flow (with median interpolation) is near a border with invalid flow -> if yes, reject the keypoint
            {
                int dx;
                for(dx = -5; dx < 6; dx++)
                {
                    int dy;
                    for(dy = -5; dy < 6; dy++)
                    {
                        xd = hlp.x + dx;
                        yd = hlp.y + dy;
                        if((xd > 0) && (xd < static_cast<int>(w)) && (yd > 0) && (yd < static_cast<int>(h)))
                        {
                            if(channelsFlow[2].at<float>(yd, xd) == 0)
                                break;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    if(dy < 6)
                        break;
                }
                if(dx < 6)
                {
                    keypL.erase(keypL.begin()+i);
                    i--;
                    continue;
                }
                else
                {
                    x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
                    x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
                }
            }
            else
            {
                keypL.erase(keypL.begin()+i);
                i--;
                continue;
            }

            keypts2idx.index->radiusSearch(&x2e(0), searchradius, radius_matches, nanoflann::SearchParams(maxDepthSearch));
            if(radius_matches.empty())
            {
                leftInlier.push_back(false);
                radius_matches.clear();
                continue;
            }
            leftInlier.push_back(true);
            nearest_dist.push_back(radius_matches[0]);
            if(radius_matches.size() > 1)
            {
                second_nearest_dist.insert(second_nearest_dist.end(), radius_matches.begin()+1, radius_matches.end());
            }
            radius_matches.clear();
        }
    }
    else
    {
        //Search for ground truth matches using a homography
        for(auto &i: keypL)
        {
            float hlp = static_cast<float>(homoGT.at<double>(2,0)) * i.pt.x + static_cast<float>(homoGT.at<double>(2,1)) * i.pt.y + static_cast<float>(homoGT.at<double>(2,2));
            x2e(0) = static_cast<float>(homoGT.at<double>(0,0)) * i.pt.x + static_cast<float>(homoGT.at<double>(0,1)) * i.pt.y + static_cast<float>(homoGT.at<double>(0,2));
            x2e(1) = static_cast<float>(homoGT.at<double>(1,0)) * i.pt.x + static_cast<float>(homoGT.at<double>(1,1)) * i.pt.y + static_cast<float>(homoGT.at<double>(1,2));
            x2e(0) /= hlp;
            x2e(1) /= hlp;
            keypts2idx.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
            if(radius_matches.empty())
            {
                leftInlier.push_back(false);
                radius_matches.clear();
                continue;
            }
            leftInlier.push_back(true);
            nearest_dist.push_back(radius_matches[0]);
            if(radius_matches.size() > 1)
            {
                second_nearest_dist.insert(second_nearest_dist.end(), radius_matches.begin()+1, radius_matches.end());
            }
            radius_matches.clear();
        }
    }

    if(nearest_dist.empty())
        return -1; //No corresponding keypoints found

#if showfeatures
    cv::Mat img1c;
    drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("Keypoints 1 after invalid GT filtering", img1c );
    img1c.release();
    drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("Keypoints 2", img1c );
    cv::waitKey(0);
#endif

    //Sort distances to get the largest distances first
    sort(nearest_dist.begin(), nearest_dist.end(),
        [](pair<size_t,float> first, pair<size_t,float> second){return first.second > second.second;});

    if(!second_nearest_dist.empty())
    {
        //Check for outliers in the distances to the nearest matches
        float median, hlp, medstdsum = 0, medStd, distantth;
        if(nearest_dist.size() % 2)
            median = nearest_dist[(nearest_dist.size()-1)/2].second;
        else
            median = (nearest_dist[nearest_dist.size() / 2].second + nearest_dist[nearest_dist.size() / 2 - 1].second) / 2;

        int startexcludeval = static_cast<int>(floor(static_cast<float>(nearest_dist.size()) * 0.2f)); //to exclude the first 20% of large distances
        for(auto i = static_cast<size_t>(startexcludeval); i < nearest_dist.size(); i++)
        {
            hlp = (nearest_dist[i].second - median);
            medstdsum += hlp * hlp;
        }
        if(std::abs(medstdsum) < 1e-6)
            medStd = 0.0;
        else
            medStd = std::sqrt(medstdsum/(static_cast<float>(nearest_dist.size()) - static_cast<float>(startexcludeval) - 1.0f));

        distantth = median + 3.5f * medStd;
        if(distantth >= INITMATCHDISTANCETH_GT * INITMATCHDISTANCETH_GT)
        {
            distantth = static_cast<float>(INITMATCHDISTANCETH_GT) / 2.f;
            distantth *= distantth;
        }

        //Reject outliers in the distances to the nearest matches
        while(nearest_dist[0].second >= distantth)
        {
            nearest_dist.erase(nearest_dist.begin());
        }

        //Sort second nearest distances to get smallest distances first
        sort(second_nearest_dist.begin(), second_nearest_dist.end(),
            [](pair<size_t,float> first, pair<size_t,float> second){return first.second < second.second;});
        //Mark too near keypoints for deleting
        size_t k = 0;
        while(second_nearest_dist[k].second <= nearest_dist[0].second)
        {
            k++;
        }

        //Set the threshold
        usedMatchTH = (static_cast<double>(std::sqrt(nearest_dist[0].second)) + static_cast<double>(std::sqrt(second_nearest_dist[k].second))) / 2.0;
        if(usedMatchTH < 2.0)
            usedMatchTH = 2.0;
    }
    else
    {
        usedMatchTH = static_cast<double>(std::sqrt(nearest_dist[0].second)) + 0.5;
        usedMatchTH = usedMatchTH > static_cast<double>(INITMATCHDISTANCETH_GT) ? static_cast<double>(INITMATCHDISTANCETH_GT):usedMatchTH;
    }
    searchradius = static_cast<float>(usedMatchTH) * static_cast<float>(usedMatchTH);//floor(usedMatchTH * usedMatchTH + 0.5f);

    //Search for the ground truth matches in the right (second) image by searching for the nearest and second nearest neighbor by a radius search
    //Recalculate descriptors to exclude descriptors from deleted left keypoints
    descriptors1.release();
    matchinglib::getDescriptors(imgs[0], keypL, GTfilterExtractor, descriptors1, featuretype);
    leftInlier.clear();
    nearest_dist.clear();
    vector<vector<std::pair<size_t,float>>> second_nearest_dist_vec;
    if(flowGtIsUsed)
    {
        //Search for the ground truth matches using optical flow data
        for(size_t i = 0; i < keypL.size(); i++)
        {
            Mat descriptors2;
            float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
            size_t minDescrDist = 0;
            vector<size_t> border_marklist;//Number of points that are too near to the image border
            cv::Point2i hlp;
            hlp.x = static_cast<int>(floor(keypL[i].pt.x + 0.5)); //Round to nearest integer
            hlp.y = static_cast<int>(floor(keypL[i].pt.y + 0.5)); //Round to nearest integer
            if(channelsFlow[2].at<float>(hlp.y, hlp.x) >= 1.0)
            {
                x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
                x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
            }
            else
            {
                keypL.erase(keypL.begin() + i);
                if(i == 0)
                {
                    descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
                }
                else
                {
                    Mat descr_tmp = descriptors1.rowRange(0, i);
                    descr_tmp.push_back(descriptors1.rowRange(static_cast<int>(i) + 1, descriptors1.rows));
                    descriptors1.release();
                    descr_tmp.copyTo(descriptors1);
                }
                i--;
                continue;
            }

            keypts2idx.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
            if(radius_matches.empty())
            {
                leftInlier.push_back(false);
                second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                radius_matches.clear();
                continue;
            }

            //Get descriptors from found right keypoints
            for(auto &j: radius_matches)
            {
                keypR_tmp.push_back(keypR[j.first]);
            }
            keypR_tmp1 = keypR_tmp;
            matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
            if(radius_matches.size() > keypR_tmp.size())
            {
                if(keypR_tmp.empty())
                {
                    for(auto &j: radius_matches)
                    {
                        border_dellist.push_back(j.first);
                    }
                    leftInlier.push_back(false);
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    radius_matches.clear();
                    continue;
                }
                else
                {
                    size_t k = 0;
                    for(size_t j = 0; j < radius_matches.size(); j++)
                    {
                        if((keypR_tmp1[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp1[j].pt.y == keypR_tmp[k].pt.y))
                        {
                            k++;
                            if(k == keypR_tmp.size())
                                k--;
                        }
                        else
                        {
                            border_dellist.push_back(radius_matches[j].first);
                            border_marklist.push_back(j);
                        }
                    }
                }
            }
            keypR_tmp.clear();

            //Get index of smallest descriptor distance
            descr_dist1 = static_cast<float>(getDescriptorDistance(descriptors1.row(i), descriptors2.row(0)));
            for(size_t j = 1; j < (size_t)descriptors2.rows; j++)
            {
                descr_dist2 = static_cast<float>(getDescriptorDistance(descriptors1.row(static_cast<int>(i)), descriptors2.row(static_cast<int>(j))));
                if(descr_dist1 > descr_dist2)
                {
                    if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
                    {
                        minDescrDist = j;
                        for(auto &k: border_marklist)
                        {
                            if(minDescrDist >= k)
                                minDescrDist++;
                            else
                                break;
                        }
                    }
                    else
                    {
                        minDescrDist = j;
                    }
                    descr_dist1 = descr_dist2;
                }
            }
            if((descr_dist1 > 160.f) && (descriptors1.type() == CV_8U))
            {
                keypL.erase(keypL.begin() + i);
                if(i == 0)
                {
                    descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
                }
                else
                {
                    Mat descr_tmp = descriptors1.rowRange(0, i);
                    descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
                    descriptors1.release();
                    descr_tmp.copyTo(descriptors1);
                }
                i--;
                continue;
            }

            leftInlier.push_back(true);
            nearest_dist.push_back(radius_matches[minDescrDist]);
            if(radius_matches.size() > 1)
            {
                if(minDescrDist == 0)
                {
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
                }
                else
                {
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
                    second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
                }
            }
            else
            {
                second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
            }
            radius_matches.clear();
        }
    }
    else
    {
        //Search for ground truth matches using a homography
        for(unsigned int i = 0;i<keypL.size();i++)
        {
            Mat descriptors2;
            float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
            size_t minDescrDist = 0;
            vector<size_t> border_marklist;//Number of points that are too near to the image border
            float hlp = static_cast<float>(homoGT.at<double>(2,0)) * keypL[i].pt.x + static_cast<float>(homoGT.at<double>(2,1)) * keypL[i].pt.y + static_cast<float>(homoGT.at<double>(2,2));
            x2e(0) = static_cast<float>(homoGT.at<double>(0,0)) * keypL[i].pt.x + static_cast<float>(homoGT.at<double>(0,1)) * keypL[i].pt.y + static_cast<float>(homoGT.at<double>(0,2));
            x2e(1) = static_cast<float>(homoGT.at<double>(1,0)) * keypL[i].pt.x + static_cast<float>(homoGT.at<double>(1,1)) * keypL[i].pt.y + static_cast<float>(homoGT.at<double>(1,2));
            x2e(0) /= hlp;
            x2e(1) /= hlp;
            keypts2idx.index->radiusSearch(&x2e(0), searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
            if(radius_matches.empty())
            {
                leftInlier.push_back(false);
                second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                radius_matches.clear();
                continue;
            }
            //Get descriptors from found right keypoints
            for(auto &j: radius_matches)
            {
                keypR_tmp.push_back(keypR[j.first]);
            }
            keypR_tmp1 = keypR_tmp;
            matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
            if(radius_matches.size() > keypR_tmp.size())
            {
                if(keypR_tmp.empty())
                {
                    for(auto &j: radius_matches)
                    {
                        border_dellist.push_back(j.first);
                    }
                    leftInlier.push_back(false);
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    radius_matches.clear();
                    continue;
                }
                else
                {
                    size_t k = 0;
                    for(size_t j = 0; j < radius_matches.size(); j++)
                    {
                        if((keypR_tmp1[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp1[j].pt.y == keypR_tmp[k].pt.y))
                        {
                            k++;
                            if(k == keypR_tmp.size())
                                k--;
                        }
                        else
                        {
                            border_dellist.push_back(radius_matches[j].first);
                            border_marklist.push_back(j);
                        }
                    }
                }
            }
            keypR_tmp.clear();

            //Get index of smallest descriptor distance
            descr_dist1 = static_cast<float>(getDescriptorDistance(descriptors1.row(static_cast<int>(i)), descriptors2.row(0)));
            for(size_t j = 1; j < static_cast<size_t>(descriptors2.rows); j++)
            {
                descr_dist2 = static_cast<float>(getDescriptorDistance(descriptors1.row(static_cast<int>(i)), descriptors2.row(static_cast<int>(j))));
                if(descr_dist1 > descr_dist2)
                {
                    if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
                    {
                        minDescrDist = j;
                        for(auto &k: border_marklist)
                        {
                            if(minDescrDist >= k)
                                minDescrDist++;
                            else
                                break;
                        }
                    }
                    else
                    {
                        minDescrDist = j;
                    }
                    descr_dist1 = descr_dist2;
                }
            }
            if((descr_dist1 > 160) && (descriptors1.type() == CV_8U))
            {
                keypL.erase(keypL.begin() + i);
                if(i == 0)
                {
                    descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
                }
                else
                {
                    Mat descr_tmp = descriptors1.rowRange(0, static_cast<int>(i));
                    descr_tmp.push_back(descriptors1.rowRange(static_cast<int>(i) + 1, descriptors1.rows));
                    descriptors1.release();
                    descr_tmp.copyTo(descriptors1);
                }
                i--;
                continue;
            }

            leftInlier.push_back(true);
            nearest_dist.push_back(radius_matches[minDescrDist]);
            if(radius_matches.size() > 1)
            {
                if(minDescrDist == 0)
                {
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
                }
                else
                {
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
                    second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
                }
            }
            else
            {
                second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
            }
            radius_matches.clear();
        }
    }

#if showfeatures
    //cv::Mat img1c;
    img1c.release();
    drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("Keypoints 1 after min. Similarity filtering", img1c );
    img1c.release();
    drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("Keypoints 2", img1c );
    cv::waitKey(0);
#endif

    //Generate flow from right to left image using neighbor interpolation
    vector<Mat> channels21;
    if(flowGtIsUsed)
    {
        int x, y;
        float nfx, nfy;
        channels21.emplace_back(Mat(flowGT.rows, flowGT.cols, CV_32FC1, -1.0));
        channels21.emplace_back(Mat(flowGT.rows, flowGT.cols, CV_32FC1, -1.0));
        channels21.emplace_back(Mat(flowGT.rows, flowGT.cols, CV_32FC1, -1.0));
        for(int v = 0; v < flowGT.rows; v++)
        {
            for(int u = 0; u < flowGT.cols; u++)
            {
                if(channelsFlow[2].at<float>(v,u))
                {
                    nfx = channelsFlow[0].at<float>(v, u);
                    nfy = channelsFlow[1].at<float>(v, u);
                    x = static_cast<int>(floor(static_cast<float>(u) + nfx + 0.5f));
                    y = static_cast<int>(floor(static_cast<float>(v) + nfy + 0.5f));
                    if((x > 0) && (static_cast<float>(x) < w) && (y > 0) && (static_cast<float>(y) < h))
                    {
                        channels21[0].at<float>(static_cast<int>(y), static_cast<int>(x)) = -1.0f * nfx;
                        channels21[1].at<float>(static_cast<int>(y), static_cast<int>(x)) = -1.0f * nfy;
                        channels21[2].at<float>(static_cast<int>(y), static_cast<int>(x)) = channelsFlow[2].at<float>(v, u);
                    }
                }
            }
        }
        //Interpolate missing values using the median from the neighborhood
        for(int v = 0; v < channels21[0].rows; v++)
        {
            for(int u = 0; u < channels21[0].cols; u++)
            {
                if(channels21[2].at<float>(v,u) == -1.0)
                {
                    vector<float> mx, my;
                    float mv = 0;
                    for(int dx = -1; dx < 2; dx++)
                    {
                        for(int dy = -1; dy < 2; dy++)
                        {
                            x = u + dx;
                            y = v + dy;
                            if((x > 0) && (static_cast<float>(x) < w) && (y > 0) && (static_cast<float>(y) < h))
                            {
                                if(channels21[2].at<float>(y, x) > 0)
                                {
                                    mx.push_back(channels21[0].at<float>(y, x));
                                    my.push_back(channels21[1].at<float>(y, x));
                                    mv += channels21[2].at<float>(y, x);
                                }
                            }
                        }
                    }
                    if(mx.size() > 2)
                    {
                        sort(mx.begin(), mx.end());
                        sort(my.begin(), my.end());
                        mv = floor(mv / static_cast<float>(mx.size()) + 0.5f);
                        if(mx.size() % 2)
                        {
                            channels21[0].at<float>(v,u) = mx[(mx.size()-1)/2];
                            channels21[1].at<float>(v,u) = my[(my.size()-1)/2];
                        }
                        else
                        {
                            channels21[0].at<float>(v,u) = (float)(mx[mx.size() / 2] + mx[mx.size() / 2 - 1]) / 2.0f;
                            channels21[1].at<float>(v,u) = (float)(my[my.size() / 2] + my[my.size() / 2 - 1]) / 2.0f;
                        }
                        channels21[2].at<float>(v,u) = mv;
                    }
                    else
                    {
                        channels21[0].at<float>(v,u) = 0;
                        channels21[1].at<float>(v,u) = 0;
                        channels21[2].at<float>(v,u) = 0;
                    }
                }
            }
        }
    }

    int roundcounter = 0;
    bool additionalRound = true;
    bool initround = false;
    do
    {
        if((additionalRound || (roundcounter < 2)) && initround)
        {
            roundcounter++;
            additionalRound = false;
        }
        initround = true;
        //Remove matches that match the same keypoints in the right image
        if(roundcounter > 0)
        {
            size_t k = 0, k1;
            int i = 0;
            vector<pair<size_t,size_t>> dellist;
            while(i < static_cast<int>(leftInlier.size()))
            {
                if(leftInlier[i])
                {
                    k1 = k;
                    for(size_t j = (size_t)i+1; j < leftInlier.size(); j++)
                    {
                        if(leftInlier[j])
                        {
                            k1++;
                            if(nearest_dist[k].first == nearest_dist[k1].first)
                            {
                                dellist.emplace_back(make_pair(j, k1));
                            }
                        }
                    }
                    if(!dellist.empty())
                    {
                        additionalRound = true;
                        if(dellist.size() > 1)
                        {
                            sort(dellist.begin(), dellist.end(),
                                [](pair<size_t,size_t> first, pair<size_t,size_t> second){return first.first > second.first;});
                        }
                        for(k1 = 0; k1 < dellist.size(); k1++)
                        {
                            keypL.erase(keypL.begin() + dellist[k1].first);
                            leftInlier.erase(leftInlier.begin() + dellist[k1].first);
                            nearest_dist.erase(nearest_dist.begin() + dellist[k1].second);
                            second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + dellist[k1].first);
                        }
                        keypL.erase(keypL.begin() + i);
                        leftInlier.erase(leftInlier.begin() + i);
                        nearest_dist.erase(nearest_dist.begin() + k);
                        second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + i);
                        i--;
                    }
                    else
                    {
                        k++;
                    }
                    dellist.clear();
                }
                i++;
            }
        }

        //Check for better matches of the second nearest matches
        eigkeypts2.resize(keypL.size(),2);
        for(unsigned int i = 0;i<keypL.size();i++)
        {
            eigkeypts2(i,0) = keypL[i].pt.x;
            eigkeypts2(i,1) = keypL[i].pt.y;
        }
        KDTree_D2float keypts2idx2(eigkeypts2, maxLeafNum);
        keypts2idx2.index->buildIndex();
        float searchradius1;
        if(flowGtIsUsed)
        {
            searchradius1 = std::sqrt(static_cast<float>(searchradius));
            searchradius1 += 0.5f; //Compensate for max. error during interpolation of channels21
            searchradius1 *= searchradius1;
        }
        else
        {
            searchradius1 = searchradius;
        }
        vector<size_t> delInvalR, delNoCorrL, delCorrR;
        typedef struct match2QPar{
            size_t indexL;
            size_t indexR;
            size_t indexMarkInlR;
            float similarity;
            float similMarkInl;
            bool isQestInl;
        }match2QPar;
        if(flowGtIsUsed)
        {
            for(size_t i = 0; i < second_nearest_dist_vec.size(); i++)
            {
                if(second_nearest_dist_vec[i].empty())
                    continue;

                vector<match2QPar> m2qps;
                for(auto &j: second_nearest_dist_vec[i])
                {
                    Mat descriptorsL1;
                    size_t borderidx = 0;
                    size_t idxm = j.first;
                    vector<size_t> border_marklist;//Number of points that are too near to the image border
                    cv::Point2i hlp;
                    hlp.x = static_cast<int>(floor(keypR[idxm].pt.x + 0.5f)); //Round to nearest integer
                    hlp.y = static_cast<int>(floor(keypR[idxm].pt.y + 0.5f)); //Round to nearest integer
                    if(channels21[2].at<float>(hlp.y, hlp.x) >= 1.0)
                    {
                        x2e(0) = keypR[idxm].pt.x + channels21[0].at<float>(hlp.y, hlp.x);
                        x2e(1) = keypR[idxm].pt.y + channels21[1].at<float>(hlp.y, hlp.x);
                    }
                    //else if(channels21[2].at<float>(hlp.y, hlp.x) > 1.0) //Check if the filled flow (with median interpolation) is near a border with invalid flow -> if yes, reject the keypoint
                    //{
                    //	int dx;
                    //	for(dx = -5; dx < 6; dx++)
                    //	{
                    //		int dy;
                    //		for(dy = -5; dy < 6; dy++)
                    //		{
                    //			xd = hlp.x + dx;
                    //			yd = hlp.y + dy;
                    //			if((xd > 0) && (xd < (int)w) && (yd > 0) && (yd < (int)h))
                    //			{
                    //				if(channels21[2].at<float>(yd, xd) == 0)
                    //					break;
                    //			}
                    //			else
                    //			{
                    //				continue;
                    //			}
                    //		}
                    //		if(dy < 6)
                    //			break;
                    //	}
                    //	if(dx < 6)
                    //	{
                    //		delInvalR.push_back(i);
                    //		continue;
                    //	}
                    //	else
                    //	{
                    //		x2e(0) = keypR[idxm].pt.x + channels21[0].at<float>(hlp.y, hlp.x);
                    //		x2e(1) = keypR[idxm].pt.y + channels21[1].at<float>(hlp.y, hlp.x);
                    //	}
                    //}
                    else
                    {
                        delInvalR.push_back(idxm);
                        continue;
                    }

                    keypts2idx2.index->radiusSearch(&x2e(0),searchradius1,radius_matches,nanoflann::SearchParams(maxDepthSearch));
                    if(radius_matches.empty())
                    {
                        radius_matches.clear();
                        continue;
                    }

                    //Get descriptors from found left keypoints
                    for(auto &k: radius_matches)
                    {
                        keypL_tmp.push_back(keypL[k.first]);
                    }
                    keypL_tmp1 = keypL_tmp;
                    matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL1, featuretype);
                    if(radius_matches.size() > keypL_tmp.size())
                    {
                        if(keypL_tmp.empty())
                        {
                            for(auto &k: radius_matches)
                            {
                                border_dellistL.push_back(k.first);
                            }
                            radius_matches.clear();
                            continue;
                        }
                        else
                        {
                            size_t k = 0;
                            for(size_t j1 = 0; j1 < radius_matches.size(); j1++)
                            {
                                if((keypL_tmp1[j1].pt.x == keypL_tmp[k].pt.x) && (keypL_tmp1[j1].pt.y == keypL_tmp[k].pt.y))
                                {
                                    k++;
                                    if(k == keypL_tmp.size())
                                        k--;
                                }
                                else
                                {
                                    border_dellistL.push_back(radius_matches[j1].first);
                                    border_marklist.push_back(j1);
                                }
                            }
                        }
                    }
                    keypL_tmp.clear();

                    //Get index, descriptor distance, distance to GT, left index and right index for every found keypoint
                    for(size_t j1 = 0; j1 < static_cast<size_t>(descriptorsL1.rows); j1++)
                    {
                        match2QPar hlpq;
                        hlpq.similarity = static_cast<float>(getDescriptorDistance(descriptors22nd.row(static_cast<int>(idxm)), descriptorsL1.row(static_cast<int>(j1))));
                        if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
                        {
                            borderidx = j1;
                            for(auto &k: border_marklist)
                            {
                                if(borderidx >= k)
                                    borderidx++;
                                else
                                    break;
                            }
                        }
                        else
                        {
                            borderidx = j1;
                        }
                        hlpq.indexL = radius_matches[borderidx].first;
                        hlpq.indexR = idxm;
                        hlpq.similMarkInl = FLT_MAX;
                        hlpq.isQestInl = false;
                        m2qps.push_back(hlpq);
                    }
                }
                if(m2qps.empty())
                {
                    continue;
                }
                {
                    match2QPar hlpq;
                    Mat descriptorsL1;
                    size_t k3 = 0;
                    for(size_t k1 = 0; k1 <= i; k1++)
                    {
                        if(leftInlier[k1])
                            k3++;
                    }
                    hlpq.indexL = i;
                    hlpq.indexR = nearest_dist[k3 - 1].first;
                    keypL_tmp.push_back(keypL[i]);
                    matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL1, featuretype);
                    keypL_tmp.clear();
                    hlpq.similarity = static_cast<float>(getDescriptorDistance(descriptors22nd.row(static_cast<int>(hlpq.indexR)), descriptorsL1));
                    hlpq.similMarkInl = hlpq.similarity;
                    hlpq.indexMarkInlR = hlpq.indexR;
                    hlpq.isQestInl = true;
                    m2qps.push_back(hlpq);
                }
                for(int k = static_cast<int>(m2qps.size()) - 2; k >= 0; k--) //use m2qps.size() - 2 to exclude the nearest match added two lines above
                {
                    size_t idxm = m2qps[k].indexL;
                    if(!leftInlier[idxm])//If an outlier was found as inlier
                    {
                        delCorrR.push_back(m2qps[k].indexR);
                        m2qps.erase(m2qps.begin() + k);
                        continue;
                    }
                    //Generate the indexes of the already found nearest matches
                    if(idxm == i)
                    {
                        m2qps[k].indexMarkInlR = m2qps.back().indexMarkInlR;
                        m2qps[k].similMarkInl = m2qps.back().similMarkInl;
                    }
                    else
                    {
                        size_t k2 = 0;
                        for(size_t k1 = 0; k1 <= idxm; k1++)
                        {
                            if(leftInlier[k1])
                                k2++;
                        }
                        m2qps[k].indexMarkInlR = nearest_dist[k2-1].first;
                        //Calculate the similarity to the already found nearest matches
                        keypL_tmp.push_back(keypL[idxm]);
                        Mat descriptorsL1;
                        matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL1, featuretype);
                        keypL_tmp.clear();
                        m2qps[k].similMarkInl = static_cast<float>(getDescriptorDistance(descriptors22nd.row(static_cast<int>(m2qps[k].indexMarkInlR)), descriptorsL1));
                    }
                }
                {
                    int cntidx = 0;
                    int corrMatchIdx = INT_MAX;
                    bool delLR = false;
                    for(int k = 0; k < static_cast<int>(m2qps.size()) - 1; k++) //The ckeck of the last element can be neglected sine it is the nearest match (and there is only one index of this keypoint)
                    {
                        if(m2qps[k].indexMarkInlR == m2qps[k].indexR)
                        {
                            corrMatchIdx = k;
                            cntidx++;
                            if((m2qps[k].indexR != m2qps[k+1].indexR) && (cntidx > 0))
                            {
                                cntidx--;
                                if(delLR)
                                {
                                    goto posdelLRf;
                                }
                                for(int k1 = k - cntidx; k1 < k; k1++)
                                {
                                    if(m2qps[k1].similarity < 1.5f * m2qps[corrMatchIdx].similMarkInl)
                                    {
                                        delLR = true;
                                        break;
                                    }
                                }
                                goto posdelLRf;
                            }
                            continue;
                        }
                        if(m2qps[k].similarity < 1.5f * m2qps[k].similMarkInl)
                        {
                            delLR = true;
                        }
                        if(m2qps[k].indexR == m2qps[k+1].indexR)
                        {
                            cntidx++;
                        }
                        else
                        {
                            if((corrMatchIdx < INT_MAX) && !delLR)
                            {
                                for(int k1 = k - cntidx; k1 <= k; k1++)
                                {
                                    if(m2qps[k1].indexMarkInlR != m2qps[k1].indexR)
                                    {
                                        if(m2qps[k1].similarity < 1.5f * m2qps[corrMatchIdx].similMarkInl)
                                        {
                                            delLR = true;
                                            break;
                                        }
                                    }
                                }
                            }

                            if(corrMatchIdx == INT_MAX)
                            {
                                if(m2qps[k].indexL != i)//if the forward-backward search does not result in the same correspondence, delete the right keypoint
                                {
                                    delCorrR.push_back(m2qps[k].indexR);
                                    m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
                                    k -= cntidx;
                                    delLR = false;
                                }
                                else if(delLR)
                                {
                                    delCorrR.push_back(m2qps[k].indexR);
                                    delNoCorrL.push_back(i);
                                    m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
                                    m2qps.pop_back();
                                    k -= cntidx;
                                    delLR = false;
                                }
                            }
                            posdelLRf:
                            if(delLR)
                            {
                                delCorrR.push_back(m2qps[k].indexR);
                                delNoCorrL.push_back(m2qps[corrMatchIdx].indexL);
                                m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
                                k -= cntidx;
                            }
                            corrMatchIdx = INT_MAX;
                            delLR = false;
                            cntidx = 0;
                        }
                    }
                }
                for(int k = static_cast<int>(m2qps.size()) - 1; k >= 0; k--)
                {
                    size_t idxm = m2qps[k].indexL;
                    if((i == idxm) && !m2qps[k].isQestInl) //If the left match corresponds to the already found nearest match
                    {
                        m2qps.erase(m2qps.begin() + k);
                        continue;
                    }
                    //Remove entries of m2qps for which their found match is not a already found nearest match
                    if(m2qps[k].indexMarkInlR != m2qps[k].indexR)
                    {
                        m2qps.erase(m2qps.begin() + k);
                    }
                }
                if(m2qps.size() > 1)
                {
                    float simith, smallestSimi;
                    //Get the smallest similarity
                    smallestSimi = m2qps[0].similarity;
                    for(size_t k = 1; k < m2qps.size(); k++)
                    {
                        if(m2qps[k].similarity < smallestSimi)
                            smallestSimi = m2qps[k].similarity;
                    }
                    simith = 1.25f * smallestSimi; //Generate a threshold to delete matches with a too large similarity
                    for(int k = static_cast<int>(m2qps.size()) - 1; k >= 0; k--)
                    {
                        if(m2qps[k].similarity > simith)
                        {
                            delNoCorrL.push_back(m2qps[k].indexL);
                        }
                    }
                }
            }
        }
        else
        {
            Mat H1 = homoGT.inv();
            for(size_t i = 0; i < second_nearest_dist_vec.size(); i++)
            {
                if(second_nearest_dist_vec[i].empty())
                    continue;

                vector<match2QPar> m2qps;
                for(auto &j: second_nearest_dist_vec[i])
                {
                    Mat descriptorsL1;
                    size_t borderidx = 0;
                    size_t idxm = j.first;
                    vector<size_t> border_marklist;//Number of points that are too near to the image border
                    float hlp = static_cast<float>(H1.at<double>(2,0)) * keypR[idxm].pt.x + static_cast<float>(H1.at<double>(2,1)) * keypR[idxm].pt.y + static_cast<float>(H1.at<double>(2,2));
                    x2e(0) = static_cast<float>(H1.at<double>(0,0)) * keypR[idxm].pt.x + static_cast<float>(H1.at<double>(0,1)) * keypR[idxm].pt.y + static_cast<float>(H1.at<double>(0,2));
                    x2e(1) = static_cast<float>(H1.at<double>(1,0)) * keypR[idxm].pt.x + static_cast<float>(H1.at<double>(1,1)) * keypR[idxm].pt.y + static_cast<float>(H1.at<double>(1,2));
                    x2e(0) /= hlp;
                    x2e(1) /= hlp;

                    keypts2idx2.index->radiusSearch(&x2e(0),searchradius1,radius_matches,nanoflann::SearchParams(maxDepthSearch));
                    if(radius_matches.empty())
                    {
                        radius_matches.clear();
                        continue;
                    }

                    //Get descriptors from found left keypoints
                    for(auto &k: radius_matches)
                    {
                        keypL_tmp.push_back(keypL[k.first]);
                    }
                    keypL_tmp1 = keypL_tmp;
                    matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL1, featuretype);
                    if(radius_matches.size() > keypL_tmp.size())
                    {
                        if(keypL_tmp.empty())
                        {
                            for(auto &k: radius_matches)
                            {
                                border_dellistL.push_back(k.first);
                            }
                            radius_matches.clear();
                            continue;
                        }
                        else
                        {
                            size_t k = 0;
                            for(size_t j1 = 0; j1 < radius_matches.size(); j1++)
                            {
                                if((keypL_tmp1[j1].pt.x == keypL_tmp[k].pt.x) && (keypL_tmp1[j1].pt.y == keypL_tmp[k].pt.y))
                                {
                                    k++;
                                    if(k == keypL_tmp.size())
                                        k--;
                                }
                                else
                                {
                                    border_dellistL.push_back(radius_matches[j1].first);
                                    border_marklist.push_back(j1);
                                }
                            }
                        }
                    }
                    keypL_tmp.clear();

                    //Get index, descriptor distance, distance to GT, left index and right index for every found keypoint
                    for(size_t j1 = 0; j1 < static_cast<size_t>(descriptorsL1.rows); j1++)
                    {
                        match2QPar hlpq;
                        hlpq.similarity = static_cast<float>(getDescriptorDistance(descriptors22nd.row(static_cast<int>(idxm)), descriptorsL1.row(static_cast<int>(j1))));
                        if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
                        {
                            borderidx = j1;
                            for(auto &k: border_marklist)
                            {
                                if(borderidx >= k)
                                    borderidx++;
                                else
                                    break;
                            }
                        }
                        else
                        {
                            borderidx = j1;
                        }
                        hlpq.indexL = radius_matches[borderidx].first;
                        hlpq.indexR = idxm;
                        hlpq.similMarkInl = FLT_MAX;
                        hlpq.isQestInl = false;
                        m2qps.push_back(hlpq);
                    }
                }
                if(m2qps.empty())
                {
                    continue;
                }
                {
                    match2QPar hlpq;
                    Mat descriptorsL1;
                    size_t k3 = 0;
                    for(size_t k1 = 0; k1 <= i; k1++)
                    {
                        if(leftInlier[k1])
                            k3++;
                    }
                    hlpq.indexL = i;
                    hlpq.indexR = nearest_dist[k3 - 1].first;
                    keypL_tmp.push_back(keypL[i]);
                    matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL1, featuretype);
                    keypL_tmp.clear();
                    hlpq.similarity = static_cast<float>(getDescriptorDistance(descriptors22nd.row(static_cast<int>(hlpq.indexR)), descriptorsL1));
                    hlpq.similMarkInl = hlpq.similarity;
                    hlpq.indexMarkInlR = hlpq.indexR;
                    hlpq.isQestInl = true;
                    m2qps.push_back(hlpq);
                }
                for(int k = static_cast<int>(m2qps.size()) - 2; k >= 0; k--) //use m2qps.size() - 2 to exclude the nearest match added two lines above
                {
                    size_t idxm = m2qps[k].indexL;
                    if(!leftInlier[idxm])//If an outlier was found as inlier
                    {
                        delCorrR.push_back(m2qps[k].indexR);
                        m2qps.erase(m2qps.begin() + k);
                        continue;
                    }
                    //Generate the indexes of the already found nearest matches
                    if(idxm == i)
                    {
                        m2qps[k].indexMarkInlR = m2qps.back().indexMarkInlR;
                        m2qps[k].similMarkInl = m2qps.back().similMarkInl;
                    }
                    else
                    {
                        size_t k2 = 0;
                        for(size_t k1 = 0; k1 <= idxm; k1++)
                        {
                            if(leftInlier[k1])
                                k2++;
                        }
                        m2qps[k].indexMarkInlR = nearest_dist[k2-1].first;
                        //Calculate the similarity to the already found nearest matches
                        keypL_tmp.push_back(keypL[idxm]);
                        Mat descriptorsL1;
                        matchinglib::getDescriptors(imgs[0], keypL_tmp, GTfilterExtractor, descriptorsL1, featuretype);
                        keypL_tmp.clear();
                        m2qps[k].similMarkInl = static_cast<float>(getDescriptorDistance(descriptors22nd.row(static_cast<int>(m2qps[k].indexMarkInlR)), descriptorsL1));
                    }
                }
                {
                    int cntidx = 0;
                    int corrMatchIdx = INT_MAX;
                    bool delLR = false;
                    for(int k = 0; k < static_cast<int>(m2qps.size()) - 1; k++) //The ckeck of the last element can be neglected sine it is the nearest match (and there is only one index of this keypoint)
                    {
                        if(m2qps[k].indexMarkInlR == m2qps[k].indexR)
                        {
                            corrMatchIdx = k;
                            cntidx++;
                            if((m2qps[k].indexR != m2qps[k+1].indexR) && (cntidx > 0))
                            {
                                cntidx--;
                                if(delLR)
                                {
                                    goto posdelLRh;
                                }
                                for(int k1 = k - cntidx; k1 < k; k1++)
                                {
                                    if(m2qps[k1].similarity < 1.5f * m2qps[corrMatchIdx].similMarkInl)
                                    {
                                        delLR = true;
                                        break;
                                    }
                                }
                                goto posdelLRh;
                            }
                            continue;
                        }
                        if(m2qps[k].similarity < 1.5f * m2qps[k].similMarkInl)
                        {
                            delLR = true;
                        }
                        if(m2qps[k].indexR == m2qps[k+1].indexR)
                        {
                            cntidx++;
                        }
                        else
                        {
                            if((corrMatchIdx < INT_MAX) && !delLR)
                            {
                                for(int k1 = k - cntidx; k1 <= k; k1++)
                                {
                                    if(m2qps[k1].indexMarkInlR != m2qps[k1].indexR)
                                    {
                                        if(m2qps[k1].similarity < 1.5f * m2qps[corrMatchIdx].similMarkInl)
                                        {
                                            delLR = true;
                                            break;
                                        }
                                    }
                                }
                            }

                            if(corrMatchIdx == INT_MAX)
                            {
                                if(m2qps[k].indexL != i)
                                {
                                    delCorrR.push_back(m2qps[k].indexR);
                                    m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
                                    k -= cntidx;
                                    delLR = false;
                                }
                                else if(delLR)
                                {
                                    delCorrR.push_back(m2qps[k].indexR);
                                    delNoCorrL.push_back(i);
                                    m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
                                    m2qps.pop_back();
                                    k -= cntidx;
                                    delLR = false;
                                }
                            }
                            posdelLRh:
                            if(delLR)
                            {
                                delCorrR.push_back(m2qps[k].indexR);
                                delNoCorrL.push_back(m2qps[corrMatchIdx].indexL);
                                m2qps.erase(m2qps.begin() + k - cntidx, m2qps.begin() + k + 1);
                                k -= cntidx;
                            }
                            corrMatchIdx = INT_MAX;
                            delLR = false;
                            cntidx = 0;
                        }
                    }
                }
                for(int k = static_cast<int>(m2qps.size()) - 1; k >= 0; k--)
                {
                    size_t idxm = m2qps[k].indexL;
                    if((i == idxm) && !m2qps[k].isQestInl) //If the left match corresponds to the already found nearest match
                    {
                        m2qps.erase(m2qps.begin() + k);
                        continue;
                    }
                    //Remove entries of m2qps for which their found match is not a already found nearest match
                    if(m2qps[k].indexMarkInlR != m2qps[k].indexR)
                    {
                        m2qps.erase(m2qps.begin() + k);
                    }
                }
                if(m2qps.size() > 1)
                {
                    float simith, smallestSimi;
                    //Get the smallest similarity
                    smallestSimi = m2qps[0].similarity;
                    for(size_t k = 1; k < m2qps.size(); k++)
                    {
                        if(m2qps[k].similarity < smallestSimi)
                            smallestSimi = m2qps[k].similarity;
                    }
                    simith = 1.25f * smallestSimi;//Generate a threshold to delete matches with a too large similarity
                    for(int k = static_cast<int>(m2qps.size()) - 1; k >= 0; k--)
                    {
                        if(m2qps[k].similarity > simith)
                        {
                            delNoCorrL.push_back(m2qps[k].indexL);
                        }
                    }
                }
            }
        }

        {
            //Add keypoints for deletion if they are too near at the border
            vector<size_t> dellist = border_dellist;
            vector<size_t> dellistL = border_dellistL;

            //Add invalid (due to flow) right keypoints
            if(!delInvalR.empty())
                dellist.insert(dellist.end(), delInvalR.begin(), delInvalR.end());

            //Add invalid left keypoints for deletion
            if(!delNoCorrL.empty())
                dellistL.insert(dellistL.end(), delNoCorrL.begin(), delNoCorrL.end());

            //Add right keypoints for deletion
            if(!delCorrR.empty())
                dellist.insert(dellist.end(), delCorrR.begin(), delCorrR.end());

            if((dellistL.size() > 50) || (dellist.size() > 50))
                additionalRound = true;

            delInvalR.clear();
            delNoCorrL.clear();
            delCorrR.clear();
            border_dellist.clear();
            border_dellistL.clear();

            if(!dellistL.empty())
            {
                //Exclude multiple entries in dellistL
                sort(dellistL.begin(), dellistL.end(),
                        [](size_t first, size_t second){return first > second;});
                for(int i = 0; i < static_cast<int>(dellistL.size()) - 1; i++)
                {
                    while(dellistL[i] == dellistL[i+1])
                    {
                        dellistL.erase(dellistL.begin() + i + 1);
                        if(i == static_cast<int>(dellistL.size()) - 1)
                            break;
                    }
                }

                //Delete left keypoints
                int k = static_cast<int>(nearest_dist.size()) - 1, i = 0;
                for(int j = static_cast<int>(leftInlier.size()) - 1; j >= 0; j--)
                {
                    if(static_cast<int>(dellistL[i]) == j)
                    {
                        keypL.erase(keypL.begin() + j);
                        if(leftInlier[j])
                        {
                            nearest_dist.erase(nearest_dist.begin() + k);
                            k--;
                        }
                        leftInlier.erase(leftInlier.begin() + j);
                        i++;
                        if(i == static_cast<int>(dellistL.size()))
                            break;
                    }
                    else if(leftInlier[j])
                    {
                        k--;
                    }
                }
            }

            if(!dellist.empty())
            {
                //Exclude multiple entries in dellist
                sort(dellist.begin(), dellist.end(),
                        [](size_t first, size_t second){return first > second;});
                for(int i = 0; i < static_cast<int>(dellist.size()) - 1; i++)
                {
                    while(dellist[i] == dellist[i+1])
                    {
                        dellist.erase(dellist.begin() + i + 1);
                        if(i == static_cast<int>(dellist.size()) - 1)
                            break;
                    }
                }

                //Remove right keypoints listed in dellist
                sort(dellist.begin(), dellist.end(),
                        [](size_t first, size_t second){return first > second;});
                for(auto &i: dellist)
                {
                    keypR.erase(keypR.begin() + i);
                }
            }
            delNoCorrL.clear();
            delCorrR.clear();
        }

#if showfeatures
        //cv::Mat img1c;
        img1c.release();
        drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("Keypoints 1 after crosscheck", img1c );
        img1c.release();
        drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("Keypoints 2", img1c );
        cv::waitKey(0);
#endif

        //Search for matching keypoints in the right image as the found indexes arent valid anymore
        //Recalculate descriptors to exclude descriptors from deleted left keypoints
        descriptors1.release();
        //extractor->compute(imgs[0],keypL,descriptors1);
        matchinglib::getDescriptors(imgs[0], keypL, GTfilterExtractor, descriptors1, featuretype);
        eigkeypts2.resize(keypR.size(),2);
        for(unsigned int i = 0;i<keypR.size();i++)
        {
            eigkeypts2(i,0) = keypR[i].pt.x;
            eigkeypts2(i,1) = keypR[i].pt.y;
        }
        KDTree_D2float keypts2idx1(eigkeypts2,maxLeafNum);
        keypts2idx1.index->buildIndex();

        //Search for the ground truth matches in the right (second) image by searching for the nearest and second nearest neighbor by a radius search
        leftInlier.clear();
        nearest_dist.clear();
        second_nearest_dist_vec.clear();
        if(flowGtIsUsed)
        {
            //int xd, yd;
            //Search for the ground truth matches using optical flow data
            for(int i = 0; i < static_cast<int>(keypL.size());i++)
            {
                Mat descriptors2;
                float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
                size_t minDescrDist = 0;
                vector<size_t> border_marklist;//Number of points that are too near to the image border
                cv::Point2i hlp;
                hlp.x = static_cast<int>(floor(keypL[i].pt.x + 0.5f)); //Round to nearest integer
                hlp.y = static_cast<int>(floor(keypL[i].pt.y + 0.5f)); //Round to nearest integer
                if(channelsFlow[2].at<float>(hlp.y, hlp.x) >= 1.0)
                {
                    x2e(0) = keypL[i].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
                    x2e(1) = keypL[i].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
                }
                else
                {
                    keypL.erase(keypL.begin() + i);
                    if(i == 0)
                    {
                        descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
                    }
                    else
                    {
                        Mat descr_tmp = descriptors1.rowRange(0, i);
                        descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
                        descriptors1.release();
                        descr_tmp.copyTo(descriptors1);
                    }
                    i--;
                    continue;
                }

                keypts2idx1.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
                if(radius_matches.empty())
                {
                    leftInlier.push_back(false);
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    radius_matches.clear();
                    continue;
                }

                //Get descriptors from found right keypoints
                for(auto &j: radius_matches)
                {
                    keypR_tmp.push_back(keypR[j.first]);
                }
                //extractor->compute(imgs[1],keypR_tmp,descriptors2);
                matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
                if(radius_matches.size() > keypR_tmp.size())
                {
                    if(keypR_tmp.empty())
                    {
                        for(auto &j: radius_matches)
                        {
                            border_dellist.push_back(j.first);
                        }
                        leftInlier.push_back(false);
                        second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                        radius_matches.clear();
                        continue;
                    }
                    else
                    {
                        size_t k = 0;
                        vector<cv::KeyPoint> keypR_tmp11;
                        for(auto &j: radius_matches)
                        {
                            keypR_tmp11.push_back(keypR[j.first]);
                        }
                        for(size_t j = 0; j < radius_matches.size(); j++)
                        {
                            if((keypR_tmp11[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp11[j].pt.y == keypR_tmp[k].pt.y))
                            {
                                k++;
                                if(k == keypR_tmp.size())
                                    k--;
                            }
                            else
                            {
                                border_dellist.push_back(radius_matches[j].first);
                                border_marklist.push_back(j);
                            }
                        }
                    }
                }
                keypR_tmp.clear();

                //Get index of smallest descriptor distance
                descr_dist1 = static_cast<float>(getDescriptorDistance(descriptors1.row(i), descriptors2.row(0)));
                for(size_t j = 1; j < static_cast<size_t>(descriptors2.rows); j++)
                {
                    descr_dist2 = static_cast<float>(getDescriptorDistance(descriptors1.row(i), descriptors2.row(static_cast<int>(j))));
                    if(descr_dist1 > descr_dist2)
                    {
                        if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
                        {
                            minDescrDist = j;
                            for(auto &k: border_marklist)
                            {
                                if(minDescrDist >= k)
                                    minDescrDist++;
                                else
                                    break;
                            }
                        }
                        else
                        {
                            minDescrDist = j;
                        }
                        descr_dist1 = descr_dist2;
                    }
                }
                if((descr_dist1 > 160.f) && (descriptors1.type() == CV_8U))
                {
                    additionalRound = true;
                    keypL.erase(keypL.begin() + i);
                    if(i == 0)
                    {
                        descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
                    }
                    else
                    {
                        Mat descr_tmp = descriptors1.rowRange(0, i);
                        descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
                        descriptors1.release();
                        descr_tmp.copyTo(descriptors1);
                    }
                    i--;
                    continue;
                }

                leftInlier.push_back(true);
                nearest_dist.push_back(radius_matches[minDescrDist]);
                if(radius_matches.size() > 1)
                {
                    if(minDescrDist == 0)
                    {
                        second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                        second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
                    }
                    else
                    {
                        second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                        second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
                        second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
                    }
                }
                else
                {
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                }
                radius_matches.clear();
            }
        }
        else
        {
            //Search for ground truth matches using a homography
            for(size_t i = 0; i < keypL.size(); i++)
            {
                Mat descriptors2;
                float descr_dist1, descr_dist2; //Descriptor distance of found right keypoints
                size_t minDescrDist = 0;
                vector<size_t> border_marklist;//Number of points that are too near to the image border
                float hlp = static_cast<float>(homoGT.at<double>(2,0)) * keypL[i].pt.x + static_cast<float>(homoGT.at<double>(2,1)) * keypL[i].pt.y + static_cast<float>(homoGT.at<double>(2,2));
                x2e(0) = static_cast<float>(homoGT.at<double>(0,0)) * keypL[i].pt.x + static_cast<float>(homoGT.at<double>(0,1)) * keypL[i].pt.y + static_cast<float>(homoGT.at<double>(0,2));
                x2e(1) = static_cast<float>(homoGT.at<double>(1,0)) * keypL[i].pt.x + static_cast<float>(homoGT.at<double>(1,1)) * keypL[i].pt.y + static_cast<float>(homoGT.at<double>(1,2));
                x2e(0) /= hlp;
                x2e(1) /= hlp;
                keypts2idx1.index->radiusSearch(&x2e(0),searchradius,radius_matches,nanoflann::SearchParams(maxDepthSearch));
                if(radius_matches.empty())
                {
                    leftInlier.push_back(false);
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                    radius_matches.clear();
                    continue;
                }

                //Get descriptors from found right keypoints
                for(auto &j: radius_matches)
                {
                    keypR_tmp.push_back(keypR[j.first]);
                }
                //extractor->compute(imgs[1],keypR_tmp,descriptors2);
                matchinglib::getDescriptors(imgs[1], keypR_tmp, GTfilterExtractor, descriptors2, featuretype);
                if(radius_matches.size() > keypR_tmp.size())
                {
                    if(keypR_tmp.empty())
                    {
                        for(auto &j: radius_matches)
                        {
                            border_dellist.push_back(j.first);
                        }
                        leftInlier.push_back(false);
                        second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                        radius_matches.clear();
                        continue;
                    }
                    else
                    {
                        size_t k = 0;
                        vector<cv::KeyPoint> keypR_tmp11;
                        for(auto &j: radius_matches)
                        {
                            keypR_tmp11.push_back(keypR[j.first]);
                        }
                        for(size_t j = 0; j < radius_matches.size(); j++)
                        {
                            if((keypR_tmp11[j].pt.x == keypR_tmp[k].pt.x) && (keypR_tmp11[j].pt.y == keypR_tmp[k].pt.y))
                            {
                                k++;
                                if(k == keypR_tmp.size())
                                    k--;
                            }
                            else
                            {
                                border_dellist.push_back(radius_matches[j].first);
                                border_marklist.push_back(j);
                            }
                        }
                    }
                }
                keypR_tmp.clear();

                //Get index of smallest descriptor distance
                descr_dist1 = static_cast<float>(getDescriptorDistance(descriptors1.row(i), descriptors2.row(0)));
                for(size_t j = 1; j < static_cast<size_t>(descriptors2.rows); j++)
                {
                    descr_dist2 = static_cast<float>(getDescriptorDistance(descriptors1.row(static_cast<int>(i)), descriptors2.row(static_cast<int>(j))));
                    if(descr_dist1 > descr_dist2)
                    {
                        if(!border_marklist.empty()) //If a keypoint was deleted, restore the index
                        {
                            minDescrDist = j;
                            for(auto &k: border_marklist)
                            {
                                if(minDescrDist >= k)
                                    minDescrDist++;
                                else
                                    break;
                            }
                        }
                        else
                        {
                            minDescrDist = j;
                        }
                        descr_dist1 = descr_dist2;
                    }
                }
                if((descr_dist1 > 160.f) && (descriptors1.type() == CV_8U))
                {
                    additionalRound = true;
                    keypL.erase(keypL.begin() + i);
                    if(i == 0)
                    {
                        descriptors1 = descriptors1.rowRange(1, descriptors1.rows);
                    }
                    else
                    {
                        Mat descr_tmp = descriptors1.rowRange(0, i);
                        descr_tmp.push_back(descriptors1.rowRange(i+1, descriptors1.rows));
                        descriptors1.release();
                        descr_tmp.copyTo(descriptors1);
                    }
                    i--;
                    continue;
                }

                leftInlier.push_back(true);
                nearest_dist.push_back(radius_matches[minDescrDist]);
                if(radius_matches.size() > 1)
                {
                    if(minDescrDist == 0)
                    {
                        second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                        second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin()+1, radius_matches.end());
                    }
                    else
                    {
                        second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                        second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin(), radius_matches.begin() + minDescrDist);
                        second_nearest_dist_vec.back().insert(second_nearest_dist_vec.back().end(), radius_matches.begin() + minDescrDist + 1, radius_matches.end());
                    }
                }
                else
                {
                    second_nearest_dist_vec.emplace_back(vector<pair<size_t,float>>());
                }
                radius_matches.clear();
            }
        }

#if showfeatures
        //cv::Mat img1c;
        img1c.release();
        drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("Keypoints 1 after final radius search", img1c );
        img1c.release();
        drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("Keypoints 2", img1c );
        cv::waitKey(0);
#endif

    }while((additionalRound || (roundcounter < 2)));


    //Remove matches that match the same keypoints in the right image
    if(roundcounter > 0)
    {
        size_t k = 0, k1;
        int i = 0;
        vector<pair<size_t,size_t>> dellist;
        while(i < static_cast<int>(leftInlier.size()))
        {
            if(leftInlier[i])
            {
                k1 = k;
                for(size_t j = static_cast<size_t>(i)+1; j < leftInlier.size(); j++)
                {
                    if(leftInlier[j])
                    {
                        k1++;
                        if(nearest_dist[k].first == nearest_dist[k1].first)
                        {
                            dellist.emplace_back(make_pair(j, k1));
                        }
                    }
                }
                if(!dellist.empty())
                {
                    if(dellist.size() > 1)
                    {
                        sort(dellist.begin(), dellist.end(),
                            [](pair<size_t,size_t> first, pair<size_t,size_t> second){return first.first > second.first;});
                    }
                    for(k1 = 0; k1 < dellist.size(); k1++)
                    {
                        keypL.erase(keypL.begin() + dellist[k1].first);
                        leftInlier.erase(leftInlier.begin() + dellist[k1].first);
                        nearest_dist.erase(nearest_dist.begin() + dellist[k1].second);
                        second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + dellist[k1].first);
                    }
                    keypL.erase(keypL.begin() + i);
                    leftInlier.erase(leftInlier.begin() + i);
                    nearest_dist.erase(nearest_dist.begin() + k);
                    second_nearest_dist_vec.erase(second_nearest_dist_vec.begin() + i);
                    i--;
                }
                else
                {
                    k++;
                }
                dellist.clear();
            }
            i++;
        }
    }

	//Delete TN to have equal number of TN in both images
    //Get outliers in both images
    size_t leftInlSize = nearest_dist.size();
    size_t leftOutlSize = leftInlier.size() - leftInlSize;
    size_t rightOutlSize = keypR.size() - leftInlSize;
    int outlDiff = static_cast<int>(rightOutlSize) - static_cast<int>(leftOutlSize);
    vector<size_t> outlR, outlL;
    vector<cv::KeyPoint> keypLOutl_tmp;

    if(rightOutlSize > 0)
    {
        vector<pair<size_t,float>> nearest_dist_tmp = nearest_dist;
        sort(nearest_dist_tmp.begin(), nearest_dist_tmp.end(),
            [](pair<size_t,float> first, pair<size_t,float> second){return first.first < second.first;});

        size_t j = 0;
        for(size_t i = 0; i < keypR.size(); i++)
        {
            if(j < leftInlSize)
            {
                if(nearest_dist_tmp[j].first == i)
                {
                    j++;
                }
                else
                {
                    outlR.push_back(i);
                }
            }
            else
            {
                outlR.push_back(i);
            }
        }
    }
    if(leftOutlSize > 0)
    {
        for(size_t i = 0; i < leftInlier.size(); i++)
        {
            if(!leftInlier[i])
            {
                outlL.push_back(i);
            }
        }
    }

    if(outlDiff > 0)
    {
        //Delete all outliers from the right image so that it is equal to the number of outliers in the left image
        for(int i = 0; i < outlDiff; i++)
        {
            int delOutlIdx = rand() % rightOutlSize;
            size_t rKeyPIdx = outlR[delOutlIdx];
            for(auto &k1: nearest_dist)
            {
                if(k1.first > rKeyPIdx)
                {
                    k1.first--;
                }
            }
            keypR.erase(keypR.begin() + rKeyPIdx);
            for(size_t k1 = 0; k1 < rightOutlSize; k1++)
            {
                if(outlR[k1] > rKeyPIdx)
                {
                    outlR[k1]--;
                }
            }
            outlR.erase(outlR.begin() + delOutlIdx);
            rightOutlSize--;
        }
    }
    else if(outlDiff < 0)
    {
        outlDiff *= -1;
        //Delete as many outliers from the left image so that it is equal to the number of outliers in the right image
        for(int i = 0; i < outlDiff; i++)
        {
            int delOutlIdx = rand() % leftOutlSize;
            keypL.erase(keypL.begin() + outlL[delOutlIdx]);
            leftInlier.erase(leftInlier.begin() + outlL[delOutlIdx]);
            for(size_t k1 = 0; k1 < leftOutlSize; k1++)
            {
                if(outlL[k1] > outlL[delOutlIdx])
                {
                    outlL[k1]--;
                }
            }
            outlL.erase(outlL.begin() + delOutlIdx);
            leftOutlSize--;
        }
    }
    CV_Assert(keypR.size() == keypL.size())


#if showfeatures
	//cv::Mat img1c;
	img1c.release();
	drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	imshow("Keypoints 1 after equalization of outliers", img1c );
	img1c.release();
	drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	imshow("Keypoints 2 after equalization of outliers", img1c );
	cv::waitKey(0);
#endif

#if showfeatures
	//cv::Mat img1c;
	img1c.release();
	drawKeypoints( imgs[0], keypL, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	imshow("Keypoints 1 after inlier ratio filtering", img1c );
	img1c.release();
	drawKeypoints( imgs[1], keypR, img1c, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	imshow("Keypoints 2 after inlier ratio filtering", img1c );
	cv::waitKey(0);
#endif

	//Store ground truth matches
	cv::DMatch singleMatch;
	size_t k = 0;
	rightInlier = vector<bool>(keypR.size(), false);
	for(size_t j = 0; j < leftInlier.size(); j++)
	{
		if(leftInlier[j])
		{
			singleMatch.queryIdx = static_cast<int>(j);
			singleMatch.trainIdx = static_cast<int>(nearest_dist[k].first);
			singleMatch.distance = nearest_dist[k].second; //Squared distance of the keypoint in the right image to the calculated position from ground truth
			matchesGT.push_back(singleMatch);
            rightInlier[singleMatch.trainIdx] = true;
			k++;

			//{
			//	int idx = j;//matchesGT[k].queryIdx;
			//	int idx1 = nearest_dist[k-1].first;//matchesGT[k].trainIdx;
			//	float x;
			//	float y;
			//	if(flowGtIsUsed)
			//	{
			//		cv::Point2i hlp;
			//		hlp.x = (int)floor(keypL[idx].pt.x + 0.5f); //Round to nearest integer
			//		hlp.y = (int)floor(keypL[idx].pt.y + 0.5f); //Round to nearest integer
			//		x = keypL[idx].pt.x + channelsFlow[0].at<float>(hlp.y, hlp.x);
			//		y = keypL[idx].pt.y + channelsFlow[1].at<float>(hlp.y, hlp.x);
			//	}
			//	else
			//	{
			//		float hlp = (float)homoGT.at<double>(2,0) * keypL[idx].pt.x + (float)homoGT.at<double>(2,1) * keypL[idx].pt.y + (float)homoGT.at<double>(2,2);;
			//		x = (float)homoGT.at<double>(0,0) * keypL[idx].pt.x + (float)homoGT.at<double>(0,1) * keypL[idx].pt.y + (float)homoGT.at<double>(0,2);
			//		y = (float)homoGT.at<double>(1,0) * keypL[idx].pt.x + (float)homoGT.at<double>(1,1) * keypL[idx].pt.y + (float)homoGT.at<double>(1,2);
			//		x /= hlp;
			//		y /= hlp;
			//	}
			//	x = x - keypR[idx1].pt.x;
			//	y = y - keypR[idx1].pt.y;
			//	x = std::sqrt(x*x + y*y);
			//	if(x > (float)usedMatchTH)
			//	{
			//		cout << "wrong ground truth, dist: " << x << "idx: " << idx << endl;
			//	}
			//}
		}
	}
	if(k < 15)
		return -1; //Too less features remaining
	positivesGT = k;
	negativesGT = leftInlier.size() - k;
	inlRatio = static_cast<double>(positivesGT) / static_cast<double>(keypR.size());

	//Show ground truth matches
	/*{
		Mat img_match;
		std::vector<cv::KeyPoint> keypL_reduced;//Left keypoints
		std::vector<cv::KeyPoint> keypR_reduced;//Right keypoints
		std::vector<cv::DMatch> matches_reduced;
		std::vector<cv::KeyPoint> keypL_reduced1;//Left keypoints
		std::vector<cv::KeyPoint> keypR_reduced1;//Right keypoints
		std::vector<cv::DMatch> matches_reduced1;
		int j = 0;
		size_t keepNMatches = 100;
		size_t keepXthMatch = 1;
		if(matchesGT.size() > keepNMatches)
			keepXthMatch = matchesGT.size() / keepNMatches;
		for (unsigned int i = 0; i < matchesGT.size(); i++)
		{
			int idx = matchesGT[i].queryIdx;
			if(leftInlier[idx])
			{
				keypL_reduced.push_back(keypL[idx]);
				matches_reduced.push_back(matchesGT[i]);
				matches_reduced.back().queryIdx = j;
				keypR_reduced.push_back(keypR[matches_reduced.back().trainIdx]);
				matches_reduced.back().trainIdx = j;
				j++;
			}
		}
		j = 0;
		for (unsigned int i = 0; i < matches_reduced.size(); i++)
		{
			if((i % (int)keepXthMatch) == 0)
			{
				keypL_reduced1.push_back(keypL_reduced[i]);
				matches_reduced1.push_back(matches_reduced[i]);
				matches_reduced1.back().queryIdx = j;
				keypR_reduced1.push_back(keypR_reduced[i]);
				matches_reduced1.back().trainIdx = j;
				j++;
			}
		}
		drawMatches(imgs[0], keypL_reduced1, imgs[1], keypR_reduced1, matches_reduced1, img_match);
		imshow("Ground truth matches", img_match);
		waitKey(0);
	}*/

	return 0;
}

//Prepares the filename for reading or writing GT files including the path
std::string baseMatcher::prepareFileNameGT(const pair<std::string, std::string> &filenamesImg, const string &GTM_path)
{
    string fname_new;
    fname_new = getGTMbasename() + concatImgNames(filenamesImg);
    fname_new += ".yaml.gz";
    return concatPath(GTM_path, fname_new);
}

std::string baseMatcher::concatImgNames(const std::pair<std::string, std::string> &filenamesImg){
    string fname_new;
    string name_only;
    if(!filenamesImg.first.empty()){
        name_only = getFilenameFromPath(filenamesImg.first);
        fname_new = remFileExt(name_only);
    }
    if(!filenamesImg.second.empty()){
        name_only = getFilenameFromPath(filenamesImg.second);
        fname_new += "-" + remFileExt(name_only);
    }
    return fname_new;
}

std::string baseMatcher::getGTMbasename() const{
    string tmp = featuretype;
    tmp += "_GTd-";
    tmp += GTfilterExtractor;
    tmp += '_';
    return tmp;
}

/* Read GTM from disk
 *
 * string filenameGT			Input  -> The path and filename of the ground truth file
 */
bool baseMatcher::readGTMatchesDisk(const std::string &filenameGT)
{
    FileStorage fs(filenameGT, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filenameGT << endl;
        return false;
    }
    fs["keypointType"] >> featuretype;
    fs["descriptorType"] >> GTfilterExtractor;
    fs["usedMatchTH"] >> usedMatchTH;
    fs["inlRatio"] >> inlRatio;
    int tmp = 0;
    fs["positivesGT"] >> tmp;
    positivesGT = static_cast<size_t>(tmp);
    fs["negativesGT"] >> tmp;
    negativesGT = static_cast<size_t>(tmp);
    fs["keypL"] >> keypL;
    fs["keypR"] >> keypR;
    fs["matchesGT"] >> matchesGT;
    FileNode n = fs["leftInlier"];
    if (n.type() != FileNode::SEQ) {
        cerr << "leftInlier is not a sequence! FAIL" << endl;
        return false;
    }
    leftInlier.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    while (it != it_end) {
        bool inl = false;
        it >> inl;
        leftInlier.push_back(inl);
    }
    n = fs["rightInlier"];
    if (n.type() != FileNode::SEQ) {
        cerr << "rightInlier is not a sequence! FAIL" << endl;
        return false;
    }
    rightInlier.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        bool inl = false;
        it >> inl;
        rightInlier.push_back(inl);
    }

    quality.clear();
    n = fs["falseGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "falseGT is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            FileNode n1 = *it;
            cv::Point2f p1, p2;
            n1["first"] >> p1;
            n1["second"] >> p2;
            quality.falseGT.emplace_back(p1, p2);
        }
    }
    n = fs["distanceHisto"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distanceHisto is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            FileNode n1 = *it;
            int v2;
            double v1;
            n1["first"] >> v1;
            n1["second"] >> v2;
            quality.distanceHisto.emplace_back(v1, v2);
        }
    }
    n = fs["distances"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distances is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            double v = 0;
            it >> v;
            quality.distances.push_back(v);
        }
    }
    n = fs["notMatchable"];
    if(!n.empty()) {
        n >> quality.notMatchable;
    }
    n = fs["errvecs"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "errvecs is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            cv::Point2f p1;
            it >> p1;
            quality.errvecs.push_back(p1);
        }
    }
    n = fs["perfectMatches"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "perfectMatches is not a sequence! FAIL" << endl;
            return false;
        }
        refinedGTMAvailable = true;
        it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it) {
            FileNode n1 = *it;
            cv::Point2f p1, p2;
            n1["first"] >> p1;
            n1["second"] >> p2;
            quality.perfectMatches.emplace_back(p1, p2);
        }
    }else{
        refinedGTMAvailable = false;
    }
    n = fs["matchesGT_idx"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "matchesGT_idx is not a sequence! FAIL" << endl;
            return false;
        }
        refinedGTMAvailable &= true;
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int v = 0;
            it >> v;
            quality.matchesGT_idx.push_back(v);
        }
    }else{
        refinedGTMAvailable = false;
    }
    n = fs["HE"];
    if(!n.empty()) {
        n >> quality.HE;
    }
    n = fs["validityValFalseGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "validityValFalseGT is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int v = 0;
            it >> v;
            quality.validityValFalseGT.push_back(v);
        }
    }
    n = fs["errvecsGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "errvecsGT is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            cv::Point2f p1;
            it >> p1;
            quality.errvecsGT.push_back(p1);
        }
    }
    n = fs["distancesGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distancesGT is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            double v = 0;
            it >> v;
            quality.distancesGT.push_back(v);
        }
    }
    n = fs["validityValGT"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "validityValGT is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            int v = 0;
            it >> v;
            quality.validityValGT.push_back(v);
        }
    }
    n = fs["distancesEstModel"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "distancesEstModel is not a sequence! FAIL" << endl;
            return false;
        }
        refinedGTMAvailable &= true;
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            double v = 0;
            it >> v;
            quality.distancesEstModel.push_back(v);
        }
    }else{
        refinedGTMAvailable = false;
    }
    n = fs["id"];
    if(!n.empty()) {
        n >> quality.id;
    }
    n = fs["autoManualAnnot"];
    if(!n.empty()) {
        if (n.type() != FileNode::SEQ) {
            cerr << "autoManualAnnot is not a sequence! FAIL" << endl;
            return false;
        }
        it = n.begin(), it_end = n.end();
        while (it != it_end) {
            char v = 0;
            it >> v;
            quality.autoManualAnnot.push_back(v);
        }
    }

	return true;
}

/* Clears vectors containing ground truth information
 *
 * Return value:				none
 */
void baseMatcher::clearGTvars()
{
	leftInlier.clear();
    rightInlier.clear();
	matchesGT.clear();
	keypL.clear();
	keypR.clear();
    sum_TP = 0;
    sum_TN = 0;
    keypLAll.clear();
    keypRAll.clear();
    leftInlierAll.clear();
    rightInlierAll.clear();
    matchesGTAll.clear();
    imgNamesAll.clear();
}

/* Reads a line from the stream, checks if the first word corresponds to the given keyword and if true, reads the value after the keyword.
 *
 * ifstream gtFromFile			Input  -> Input stream
 * string keyWord				Input  -> Keyword that is compared to the first word of the stream
 * double *value				Output -> Value from the stream after the keyword
 *
 * Return value:				true:	Success
 *								false:	Failed
 */
bool readDoubleVal(ifstream & gtFromFile, const std::string& keyWord, double *value)
{
	//char stringline[100];
	string singleLine;
	size_t strpos;

	//gtFromFile.getline(singleLine);
	std::getline(gtFromFile, singleLine);
	//singleLine = stringline;
	if(singleLine.empty()) return false;
	strpos = singleLine.find(keyWord);
	if(strpos == std::string::npos)
	{
		gtFromFile.close();
		return false;
	}
	singleLine = singleLine.substr(strpos + keyWord.size());
	*value = strtod(singleLine.c_str(), nullptr);
	//if(*value == 0) return false;

	return true;
}

/* Writes ground truth information to the hard disk.
 *
 * string filenameGT			Input  -> The path and filename of the ground truth file
 * bool writeEmptyFile			Input  -> If true [Default = false], the gound truth generation
 *										  was not successful due to too less positives or a
 *										  too small inlier ratio. For this reason the following
 *										  values are written to the file: usedMatchTH = 0, 
 *										  inlRatioL = 0.00034, inlRatioR = 0.00021, positivesGT = 0,
 *										  negativesGT = 0, and negativesGTr = 0. Thus, if these
 *										  values are read from the file, no new ground truth generation
 *										  has to be performed which would fail again.
 *
 * Return value:				 0:	Success
 *								-1: Bad filename or directory
 */
bool baseMatcher::writeGTMatchesDisk(const std::string &filenameGT, bool writeQualityPars) {
    FileStorage fs(filenameGT, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filenameGT << endl;
        return false;
    }

    fs.writeComment("This file contains Ground Truth Matches (GTM) and some related information.\n");

    fs.writeComment("Keypoint type");
    fs << "keypointType" << featuretype;
    fs.writeComment("Used descriptor for generating GTM");
    fs << "descriptorType" << GTfilterExtractor;
    fs.writeComment("Used distance threshold for generating GTM");
    fs << "usedMatchTH" << usedMatchTH;
    fs.writeComment("Inlier ratio");
    fs << "inlRatio" << inlRatio;
    fs.writeComment("Number of true positives (TP)");
    fs << "positivesGT" << static_cast<int>(positivesGT);
    fs.writeComment("Number of true negatives (TN)");
    fs << "negativesGT" << static_cast<int>(negativesGT);
    fs.writeComment("Keypoints of first image");
    fs << "keypL" << keypL;
    fs.writeComment("Keypoints of second image");
    fs << "keypR" << keypR;
    fs.writeComment("TP matches");
    fs << "matchesGT" << matchesGT;
    fs.writeComment("Inlier/Outlier mask for keypoints in first image");
    fs << "leftInlier" << "[";
    for (_Bit_reference i : leftInlier) {
        fs << i;
    }
    fs << "]";
    fs.writeComment("Inlier/Outlier mask for keypoints in second image");
    fs << "rightInlier" << "[";
    for (_Bit_reference i : rightInlier) {
        fs << i;
    }
    fs << "]";
    if(writeQualityPars){
        if(!quality.falseGT.empty()) {
            fs.writeComment("False GT matching coordinates within the GT matches dataset");
            fs << "falseGT" << "[";
            for (auto &i : quality.falseGT) {
                fs << "{" << "first" << i.first;
                fs << "second" << i.second << "}";
            }
            fs << "]";
        }
        if(!quality.distanceHisto.empty()) {
            fs.writeComment("Histogram of the distances from the matching positions to annotated positions");
            fs << "distanceHisto" << "[";
            for (auto &i : quality.distanceHisto) {
                fs << "{" << "first" << i.first;
                fs << "second" << i.second << "}";
            }
            fs << "]";
        }
        if(!quality.distances.empty()) {
            fs.writeComment("Distances from the matching positions to annotated positions");
            fs << "distances" << "[";
            for (auto &i : quality.distances) {
                fs << i;
            }
            fs << "]";
        }
        if(quality.notMatchable) {
            fs.writeComment("Number of matches from the GT that are not matchable in reality");
            fs << "notMatchable" << quality.notMatchable;
        }
        if(!quality.errvecs.empty()) {
            fs.writeComment("Vector from the matching positions to annotated positions");
            fs << "errvecs" << "[";
            for (auto &i : quality.errvecs) {
                fs << i;
            }
            fs << "]";
        }
        if(!quality.perfectMatches.empty()) {
            fs.writeComment("The resulting annotated matches");
            fs << "perfectMatches" << "[";
            for (auto &i : quality.perfectMatches) {
                fs << "{" << "first" << i.first;
                fs << "second" << i.second << "}";
            }
            fs << "]";
        }
        if(!quality.matchesGT_idx.empty()) {
            fs.writeComment("Indices for perfectMatches pointing to corresponding match in matchesGT");
            fs << "matchesGT_idx" << "[";
            for (auto &i : quality.matchesGT_idx) {
                fs << i;
            }
            fs << "]";
        }
        if(!quality.HE.empty()) {
            fs.writeComment("The fundamental matrix or homography calculted from the annotated matches");
            fs << "HE" << quality.HE;
        }
        if(!quality.validityValFalseGT.empty()) {
            fs.writeComment("Validity level of false matches for the filled KITTI GT (1=original GT, 2= filled GT)");
            fs << "validityValFalseGT" << "[";
            for (auto &i : quality.validityValFalseGT) {
                fs << i;
            }
            fs << "]";
        }
        if(!quality.errvecsGT.empty()) {
            fs.writeComment("Vectors from the dataset GT to the annotated positions");
            fs << "errvecsGT" << "[";
            for (auto &i : quality.errvecsGT) {
                fs << i;
            }
            fs << "]";
        }
        if(!quality.distancesGT.empty()) {
            fs.writeComment("Distances from the dataset GT to the annotated positions");
            fs << "distancesGT" << "[";
            for (auto &i : quality.distancesGT) {
                fs << i;
            }
            fs << "]";
        }
        if(!quality.validityValGT.empty()) {
            fs.writeComment("Validity level of all annoted matches for the filled KITTI GT (1=original GT, 2= filled GT, -1= not annotated)");
            fs << "validityValGT" << "[";
            for (auto &i : quality.validityValGT) {
                fs << i;
            }
            fs << "]";
        }
        if(!quality.distancesEstModel.empty()) {
            fs.writeComment("Distances of the annotated positions to an estimated model (Homography or Fundamental matrix) from the annotated positions");
            fs << "distancesEstModel" << "[";
            for (auto &i : quality.distancesEstModel) {
                fs << i;
            }
            fs << "]";
        }
        if(!quality.id.empty()) {
            fs.writeComment(R"(The ID of the image pair in the form "datasetName-datasetPart-leftImageName-rightImageName" or "firstImageName-secondImageName")");
            fs << "id" << quality.id;
        }
        if(!quality.autoManualAnnot.empty()) {
            fs.writeComment("Distances of the annotated positions to an estimated model (Homography or Fundamental matrix) from the annotated positions");
            fs << "autoManualAnnot" << "[";
            for (auto &i : quality.autoManualAnnot) {
                fs << i;
            }
            fs << "]";
        }
    }

	return true;
}

bool baseMatcher::testGTmatches(int & samples, std::vector<std::pair<cv::Point2f,cv::Point2f>> & falseGT, int & usedSamples,
                                std::vector<std::pair<double,int>> & distanceHisto, std::vector<double> & distances, int remainingImgs, int & notMatchable,
                                std::vector<cv::Point2f> & errvecs, std::vector<std::pair<cv::Point2f,cv::Point2f>> & perfectMatches,
                                std::vector<int> &matchesGT_idx, cv::Mat & HE, std::vector<int> & validityValFalseGT,
                                std::vector<cv::Point2f> & errvecsGT, std::vector<double> & distancesGT, std::vector<int> & validityValGT, std::vector<double> & distancesEstModel,
                                std::vector<char> & autoManualAnno, const std::string &featureType, double threshhTh, const int *fullN, const int *fullSamples, const int *fullFails)
{
#if !defined(USE_NON_FREE_CODE)
    if(featureType == "SIFT"){
        cerr << "SIFT features are selected for refining GTM but USE_NON_FREE_CODE is not enabled. Skipping refinement."
        return false;
    }
#endif
	CV_Assert(!imgs[0].empty() && !imgs[1].empty());
	if(matchesGT.empty())
	{
		usedSamples = 0;
		falseGT.clear();
		cout << "No GT matches available for refinement!" << endl;
		return false;
	}

	int maxSampleSize;
	const int patchsizeOrig = 48;//must be devidable by 16
	int GTsi = static_cast<int>(matchesGT.size());
	cv::Size imgSize;
	Mat img_tmp[2], img_exch[2];
#if defined(USE_MANUAL_ANNOTATION)
    int patchsizeShow = 3 * patchsizeOrig;
    int patchSelectShow = selMult * patchsizeOrig;
    const int selMult = 4;
	int textheight = 100;//80;//30;
	int bottomtextheight = 20;
    int maxHorImgSize = 800;
    int maxVerImgSize;
	Mat composed;
#endif
	Mat homoGT_exch;
	vector<int> used_matches;
	vector<int> skipped;
	vector<std::pair<int,double>> wrongGTidxDist;
	vector<Mat> channelsFlow(3);
	static bool autoHestiSift = true;
	maxSampleSize = samples > GTsi ? GTsi:samples;

	CV_Assert(!(patchsizeOrig % 16));

	float halfpatch = floor(static_cast<float>(patchsizeOrig) / 2.0f);
	float quarterpatch = floor(static_cast<float>(patchsizeOrig) / 4.0f);
	float eigthpatch = floor(static_cast<float>(patchsizeOrig) / 8.0f);

	double oldP = 0, newP = 0;

	//Split 3 channel matrix for access
	if(flowGtIsUsed)
	{
		cv::split(flowGT, channelsFlow);
	}

	if(fullN && fullSamples && fullFails)
	{
		oldP = newP = (*fullSamples == 0) ? 0:((double)(*fullFails) / (double)(*fullSamples));
	}

	usedSamples = maxSampleSize;
	falseGT.clear();

	//Generate empty histogram of distances to true matching position
	const double binwidth = 0.25, maxDist = 20.0; //in pixels
	distanceHisto.push_back(std::make_pair<double,int>(0,0));
	while(distanceHisto.back().first < maxDist)
		distanceHisto.push_back(std::make_pair<double,int>(distanceHisto.back().first + binwidth,0));

#if LEFT_TO_RIGHT
	img_exch[0] = imgs[0];
	img_exch[1] = imgs[1];
	homoGT_exch = homoGT;
#else
	img_exch[0] = imgs[1];
	img_exch[1] = imgs[0];
	if(!flowGtIsUsed)
		homoGT_exch = homoGT.inv();
#endif

	imgSize.width = img_exch[0].cols > img_exch[1].cols ? img_exch[0].cols:img_exch[1].cols;
	imgSize.height = img_exch[0].rows > img_exch[1].rows ? img_exch[0].rows:img_exch[1].rows;

	//If the images do not have the same size, enlarge them with a black border
	if((img_exch[0].cols < imgSize.width) || (img_exch[0].rows < imgSize.height))
	{
		img_tmp[0] = Mat::zeros(imgSize, CV_8UC1);
		img_exch[0].copyTo(img_tmp[0](cv::Rect(0, 0, img_exch[0].cols ,img_exch[0].rows)));
	}
	else
	{
		img_exch[0].copyTo(img_tmp[0]);
	}
	if((img_exch[1].cols < imgSize.width) || (img_exch[1].rows < imgSize.height))
	{
		img_tmp[1] = Mat::zeros(imgSize, CV_8UC1);
		img_exch[1].copyTo(img_tmp[1](cv::Rect(0, 0, img_exch[1].cols ,img_exch[1].rows)));
	}
	else
	{
		img_exch[1].copyTo(img_tmp[1]);
	}

#if defined(USE_MANUAL_ANNOTATION)
	if (imgSize.width > maxHorImgSize)
    {
		maxVerImgSize = (int)(static_cast<float>(maxHorImgSize) * static_cast<float>(imgSize.height) / static_cast<float>(imgSize.width));
        composed = cv::Mat(cv::Size(maxHorImgSize * 2, maxVerImgSize + patchsizeShow + textheight + patchSelectShow + bottomtextheight), CV_8UC3);
    }
    else
    {
        composed = cv::Mat(cv::Size(imgSize.width * 2, imgSize.height + patchsizeShow + textheight + patchSelectShow + bottomtextheight), CV_8UC3);
		maxHorImgSize = imgSize.width;
		maxVerImgSize = imgSize.height;
    }
#endif

	//SIFT features & descriptors used for generating local homographies
	Ptr<FeatureDetector> detector;
    if(featureType == "SIFT"){
        detector = xfeatures2d::SIFT::create();
    }else if(featureType == "ORB"){
        detector = cv::ORB::create(500, 1.2, 8, 31, 0, 4);
    }else{
        cerr << "Only SIFT and ORB features are supported for refining GTM." << endl;
        return false;
    }
	if(detector.empty())
	{
		cout << "Cannot create feature detector!" << endl;
		return false; //Cannot create descriptor extractor
	}
    Ptr<DescriptorExtractor> extractor;
    if(featureType == "SIFT"){
        extractor = xfeatures2d::SIFT::create();
    }else{
        extractor = cv::ORB::create(500, 1.2, 8, 31, 0, 4);
    }
	if(extractor.empty())
	{
		cout << "Cannot create descriptor extractor!" << endl;
		return false; //Cannot create descriptor extractor
	}

	if(maxSampleSize < GTsi) {
        srand(time(nullptr));
    }
	for(int i = 0; i < maxSampleSize; i++)
	{
        int idx = i;
        if(maxSampleSize < GTsi) {
            idx = rand() % GTsi;
            {
                int j = 0;
                while (j < (int) used_matches.size()) {
                    idx = rand() % GTsi;
                    for (j = 0; j < (int) used_matches.size(); j++) {
                        if (idx == used_matches[j])
                            break;
                    }
                }
            }
            used_matches.push_back(idx);
        }

#if LEFT_TO_RIGHT
		cv::Point2f lkp = keypL[matchesGT[idx].queryIdx].pt;
		cv::Point2f rkp = keypR[matchesGT[idx].trainIdx].pt;
#else
		cv::Point2f lkp = keypR[matchesGT[idx].trainIdx].pt;
		cv::Point2f rkp = keypL[matchesGT[idx].queryIdx].pt;
#endif

		//Generate homography from SIFT
		Mat Hl, Haff;
		Mat origPos;
		Mat tm;
		if(flowGtIsUsed)
		{
			if(autoHestiSift)
			{
				cv::Mat mask;
				vector<cv::KeyPoint> lkpSift, rkpSift;
				double angle_rot = 0, scale = 1.;
				Mat D = Mat::ones(2,1, CV_64F), Rrot, Rdeform = Mat::eye(2,2,CV_64F);
				mask = Mat::zeros(img_tmp[0].rows + patchsizeOrig, img_tmp[0].cols + patchsizeOrig, CV_8UC1);
				mask(Rect(lkp.x, lkp.y, patchsizeOrig, patchsizeOrig)) = Mat::ones(patchsizeOrig, patchsizeOrig, CV_8UC1); //generate a mask to detect keypoints only within the image roi for the selected match
				mask = mask(Rect(halfpatch, halfpatch, img_tmp[0].cols, img_tmp[0].rows));
				detector->detect(img_tmp[0], lkpSift, mask);//extract SIFT keypoints in the left patch
				if(!lkpSift.empty())
				{
					mask = Mat::zeros(img_tmp[1].rows + patchsizeOrig, img_tmp[1].cols + patchsizeOrig, CV_8UC1);
					mask(Rect(rkp.x, rkp.y, patchsizeOrig, patchsizeOrig)) = Mat::ones(patchsizeOrig, patchsizeOrig, CV_8UC1);
					mask = mask(Rect(halfpatch, halfpatch, img_tmp[1].cols, img_tmp[1].rows));
					detector->detect(img_tmp[1], rkpSift, mask);//extract SIFT keypoints in the right patch
					if(!rkpSift.empty())
					{
						Mat ldescSift, rdescSift;
						vector<vector<cv::DMatch>> matchesBf;
						vector<cv::DMatch> matchesBfTrue;
						extractor->compute(img_tmp[0], lkpSift, ldescSift);//compute descriptors for the SIFT keypoints
						extractor->compute(img_tmp[1], rkpSift, rdescSift);//compute descriptors for the SIFT keypoints
						if(!lkpSift.empty() && !rkpSift.empty())
						{
							cv::BFMatcher matcher(NORM_L2, false);
							matcher.knnMatch(ldescSift, rdescSift, matchesBf, 2);//match the SIFT features with a BF matcher and 2 nearest neighbors
							cv::Point2f deltaP = lkp - rkp;
							for(auto & k : matchesBf)
							{
								if(k.size() < 2)
									continue;
								if(k[0].distance < (0.75 * k[1].distance))//ratio test
								{
									float dc;
									cv::Point2f deltaPe = lkpSift[k[0].queryIdx].pt - rkpSift[k[0].trainIdx].pt;
									deltaPe = deltaP - deltaPe;
									dc = deltaPe.x * deltaPe.x + deltaPe.y * deltaPe.y;
									if(dc < halfpatch * halfpatch / 4)//the flow of the new matching features sould not differ more than a quarter of the patchsize to the flow of the match under test
									{
										matchesBfTrue.push_back(k[0]);
									}
								}
							}
							if(!matchesBfTrue.empty())
							{
								if(matchesBfTrue.size() > 2)//for a minimum of 3 matches, an affine homography can be calculated
								{
									int k1;
									vector<bool> lineardep(matchesBfTrue.size(), true);
									vector<cv::DMatch> matchesBfTrue_tmp;
									for(k1 = 0; k1 < static_cast<int>(matchesBfTrue.size()); k1++ )
									{
										// check that the i-th selected point does not belong
										// to a line connecting some previously selected points
										for(int j = 0; j < k1; j++ )
										{
											if(!lineardep[j]) continue;
											double dx1 = lkpSift[matchesBfTrue[j].queryIdx].pt.x - lkpSift[matchesBfTrue[k1].queryIdx].pt.x;
											double dy1 = lkpSift[matchesBfTrue[j].queryIdx].pt.y - lkpSift[matchesBfTrue[k1].queryIdx].pt.y;
											for(int k = 0; k < j; k++ )
											{
												if(!lineardep[k]) continue;
												double dx2 = lkpSift[matchesBfTrue[k].queryIdx].pt.x - lkpSift[matchesBfTrue[k1].queryIdx].pt.x;
												double dy2 = lkpSift[matchesBfTrue[k].queryIdx].pt.y - lkpSift[matchesBfTrue[k1].queryIdx].pt.y;
												if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
													lineardep[k] = false;
											}
										}
									}

									for(k1 = 0; k1 < static_cast<int>(matchesBfTrue.size()); k1++ )
									{
										if(lineardep[k1])
											matchesBfTrue_tmp.push_back(matchesBfTrue[k1]);
									}
									matchesBfTrue = matchesBfTrue_tmp;
								}
						
								if(matchesBfTrue.size() > 2)//for a minimum of 3 matches, an affine homography can be calculated
								{
									vector<cv::Point2f> lps, rps;
									int msize = static_cast<int>(matchesBfTrue.size());
									for(int k = 0; k < msize; k++)
									{
										lps.push_back(lkpSift[matchesBfTrue[k].queryIdx].pt);
										rps.push_back(rkpSift[matchesBfTrue[k].trainIdx].pt);
									}

									if(matchesBfTrue.size() > 7) //Estimate a projective homography with RANSAC and remove outliers
									{
										Mat Hprmask;
										Mat Hpr = findHomography(lps, rps, cv::RANSAC, 1.5, Hprmask);
										if(!Hpr.empty())
										{
											vector<cv::DMatch> matchesBfTrue_tmp;
											vector<cv::Point2f> lps_tmp, rps_tmp;
											for(int k = 0; k < Hprmask.rows; k++)
											{
												if(Hprmask.at<unsigned char>(k) > 0)
												{
													matchesBfTrue_tmp.push_back(matchesBfTrue[k]);
													lps_tmp.push_back(lps[k]);
													rps_tmp.push_back(rps[k]);
												}
											}
											if(matchesBfTrue_tmp.size() > 3)
											{
												matchesBfTrue = matchesBfTrue_tmp;
												lps = lps_tmp;
												rps = rps_tmp;
												msize = static_cast<int>(matchesBfTrue.size());
											}
										}
									}

									if(matchesBfTrue.size() > 2)
									{
										Hl = cv::estimateAffine2D(lps, rps);//estimate an affine homography
										if(!Hl.empty())
										{
											scale = cv::norm(Hl.colRange(0,2));//get the scale from the homography
											double scales_med, angles_med;
											vector<double> scales, angles_rot;
											vector<cv::KeyPoint> lks, rks;
											for(int k = 0; k < msize; k++)
											{
												lks.push_back(lkpSift[matchesBfTrue[k].queryIdx]);
												rks.push_back(rkpSift[matchesBfTrue[k].trainIdx]);
											}
											for(int k = 0; k < msize; k++)//get the scales and angles from the matching SIFT keypoints
											{
												double sil = lks[k].size;
												double sir = rks[k].size;
												scales.push_back(sil / sir);

												double angl = lks[k].angle;
												double angr = rks[k].angle;
												angles_rot.push_back(angl - angr);
											}
											std::sort(scales.begin(), scales.end());
											std::sort(angles_rot.begin(), angles_rot.end());
											if(msize % 2 == 0)//get the median of the scales and angles from the matching SIFT keypoints
											{
												scales_med = (scales[msize / 2 - 1] + scales[msize / 2]) / 2.0;
												angles_med = (angles_rot[msize / 2 - 1] + angles_rot[msize / 2]) / 2.0;
											}
											else
											{
												scales_med = scales[msize / 2];
												angles_med = angles_rot[msize / 2];
											}

											cv::SVD Hsvd;
											Mat U, Vt;
											Hsvd.compute(Hl.colRange(0,2), D, U, Vt, cv::SVD::FULL_UV);//perform a SVD decomposition to extract the angles from the homography
											Rrot = U * Vt;//Calculate the rotation matrix
											Rdeform = Vt;//This is the rotation matrix for the deformation (shear) in conjuction with the diagonal matrix D
											angle_rot = -1.0 * std::atan2(Rrot.at<double>(1,0), Rrot.at<double>(0,0)) * 180.0 / 3.14159265; //As sin, cos are defined counterclockwise, but cv::keypoint::angle is clockwise, multiply it with -1
											double scalechange = scales_med / scale;
											double rotchange = abs(angles_med - angle_rot);
											if((scalechange > 1.1) || (1. / scalechange > 1.1))//Check the correctness of the scale and angle using the values from the homography and the SIFT keypoints
											{
												if(rotchange < 11.25)
												{
													if((scale < 2.) && (scalechange < 2.))
														scale = abs(1. - scale) < abs(1. - scales_med) ? scale : scales_med;
													scale = (scale > 1.2) || (scale < 0.8) ? 1.0:scale;

													if(angles_med * angle_rot > 0)//angles should have the same direction
													{
														if(abs(angle_rot) > abs(angles_med))
															angle_rot = (angle_rot + angles_med) / 2.0;
													}
													else
													{
														angle_rot = 0;
														Rdeform = Mat::eye(2, 2, Rdeform.type());
														D.setTo(1.0);
													}
												}
												else
												{
													scale = 1.0;
													angle_rot = 0;
													Rdeform = Mat::eye(2, 2, Rdeform.type());
													D.setTo(1.0);
												}
											}
											else if(rotchange > 11.25)
											{
												angle_rot = 0;
												Rdeform = Mat::eye(2, 2, Rdeform.type());
												D.setTo(1.0);
											}
										}
									}
								}
							}
						}
					}
				}
				angle_rot = angle_rot * 3.14159265 / 180.0;
				Rrot = (Mat_<double>(2,2) << std::cos(angle_rot), (-1. * std::sin(angle_rot)), std::sin(angle_rot), std::cos(angle_rot)); //generate the new rotation matrix
				Haff = scale * Rrot * Rdeform.t() * Mat::diag(D) * Rdeform; //Calculate the new affine homography (without translation)
				origPos = (Mat_<double>(2,1) << static_cast<double>(halfpatch), static_cast<double>(halfpatch));
				tm = Haff * origPos;//align the translkation vector of the homography in a way that the middle of the patch has the same coordinate
				tm = origPos - tm;
				cv::hconcat(Haff, tm, Haff);//Construct the final affine homography
			}
			else
			{
				Haff = Mat::eye(2, 3, CV_64FC1);
				tm = Mat::zeros(2, 1, CV_64FC1);
			}
		}
		else
		{
			Haff = Mat::eye(3,3,homoGT_exch.type());
			Haff.at<double>(0,2) = static_cast<double>(lkp.x) - static_cast<double>(halfpatch);
			Haff.at<double>(1,2) = static_cast<double>(lkp.y) - static_cast<double>(halfpatch);
			Haff = homoGT_exch * Haff;
			Mat origPos1 = (Mat_<double>(3,1) << static_cast<double>(halfpatch), static_cast<double>(halfpatch), 1.0);
			tm = Haff * origPos1;//align the translation vector of the homography in a way that the middle of the patch has the same coordinate
			tm /= tm.at<double>(2);
			tm = origPos1 - tm;
			tm = tm.rowRange(0,2);
			Mat helpH1 = Mat::eye(3,3,homoGT_exch.type());
			helpH1.at<double>(0,2) = tm.at<double>(0);
			helpH1.at<double>(1,2) = tm.at<double>(1);
			Haff = helpH1 * Haff;
		}

		// create images to display
#if defined(USE_MANUAL_ANNOTATION)
		composed.setTo(0);
		string str;
		vector<cv::Mat> show_color(2), patches(2), show_color_tmp(2);
		Mat blended, diffImg;
		cv::cvtColor(img_tmp[0], show_color[0], cv::COLOR_GRAY2RGB);
		cv::cvtColor(img_tmp[1], show_color[1], cv::COLOR_GRAY2RGB);
		show_color_tmp[0] = show_color[0].clone();
		show_color_tmp[1] = show_color[1].clone();

		//mark the keypoint positions
		cv::line(show_color[0], lkp - Point2f(5.0f, 0), lkp + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
		cv::line(show_color[0], lkp - Point2f(0, 5.0f), lkp + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));
		cv::line(show_color[1], rkp - Point2f(5.0f, 0), rkp + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
		cv::line(show_color[1], rkp - Point2f(0, 5.0f), rkp + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));

		//Extract the patches around the matching keypoints
		vector<cv::Mat> show_color_border(2);
		copyMakeBorder(show_color_tmp[0], show_color_border[0], halfpatch, halfpatch, halfpatch, halfpatch, BORDER_CONSTANT, Scalar(0, 0, 0));
		copyMakeBorder(show_color_tmp[1], show_color_border[1], halfpatch, halfpatch, halfpatch, halfpatch, BORDER_CONSTANT, Scalar(0, 0, 0));

		patches[0] = show_color_border[0](Rect(lkp, Size(patchsizeOrig, patchsizeOrig))).clone();
		patches[1] = show_color_border[1](Rect(rkp, Size(patchsizeOrig, patchsizeOrig))).clone();

		//Generate a blended patch
		double alpha = 0.5;
		double beta = 1.0 - alpha;
		vector<Mat> patches_color(2);
		patches[0].copyTo(patches_color[0]);
		patches[1].copyTo(patches_color[1]);
		patches_color[0].reshape(1, patches_color[0].rows * patches_color[0].cols).col(1).setTo(Scalar(0));
		patches_color[0].reshape(1, patches_color[0].rows * patches_color[0].cols).col(0).setTo(Scalar(0));
		patches_color[1].reshape(1, patches_color[1].rows * patches_color[1].cols).col(0).setTo(Scalar(0));
		patches_color[1].reshape(1, patches_color[1].rows * patches_color[1].cols).col(2).setTo(Scalar(0));
		addWeighted(patches_color[0], alpha, patches_color[1], beta, 0.0, blended);
#endif

		//Generate image diff
		cv::Point2f pdiff[2];
		int halfpatch_int = static_cast<int>(halfpatch);
		cv::Mat img_wdiff[3], patches_wdiff[2], patch_wdhist2[2], patch_equal1, largerpatch, largerpatchhist, shiftedpatch, patch_equal1_tmp;
		copyMakeBorder(img_tmp[0], img_wdiff[0], halfpatch_int, halfpatch_int, halfpatch_int, halfpatch_int, BORDER_CONSTANT, Scalar(0, 0, 0));//Extract new images without keypoint markers
		copyMakeBorder(img_tmp[1], img_wdiff[1], halfpatch_int, halfpatch_int, halfpatch_int, halfpatch_int, BORDER_CONSTANT, Scalar(0, 0, 0));
		patches_wdiff[0] = img_wdiff[0](Rect(lkp, Size(patchsizeOrig, patchsizeOrig))).clone();
		patches_wdiff[1] = img_wdiff[1](Rect(rkp, Size(patchsizeOrig, patchsizeOrig))).clone();
		equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
		equalizeHist(patches_wdiff[1], patch_wdhist2[1]);
		copyMakeBorder(img_tmp[1], img_wdiff[2], patchsizeOrig, patchsizeOrig, patchsizeOrig, patchsizeOrig, BORDER_CONSTANT, Scalar(0, 0, 0));//Get a larger border in the right image to extract a larger patch
		largerpatch = img_wdiff[2](Rect(rkp, Size(2 * patchsizeOrig, 2 * patchsizeOrig))).clone();//Extract a larger patch to enable template matching
		equalizeHist(largerpatch, largerpatchhist);

		iterativeTemplMatch(patch_wdhist2[0], largerpatchhist, patchsizeOrig, pdiff[0]);

		Mat compDiff = Mat::eye(2,2,CV_64F);
		Mat tdiff = (Mat_<double>(2,1) << static_cast<double>(pdiff[0].x), static_cast<double>(pdiff[0].y));
		cv::hconcat(compDiff, tdiff, compDiff);
		warpAffine(patch_wdhist2[0], shiftedpatch, compDiff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
		{
			cv::Point2f newpdiff = cv::Point2f(0,0);
			findLocalMin(shiftedpatch, patch_wdhist2[1], quarterpatch, eigthpatch, newpdiff, patchsizeOrig);
			compDiff = Mat::eye(2,2,CV_64F);
			Mat tdiff2 = (Mat_<double>(2,1) << static_cast<double>(newpdiff.x), static_cast<double>(newpdiff.y));
			cv::hconcat(compDiff, tdiff2, compDiff);
			warpAffine(shiftedpatch, shiftedpatch, compDiff, shiftedpatch.size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
			pdiff[0] += newpdiff;
		}
		absdiff(patch_wdhist2[0], patch_wdhist2[1], patch_equal1);
#if defined(USE_MANUAL_ANNOTATION)
		cv::cvtColor(patch_equal1, diffImg, cv::COLOR_GRAY2RGB);
#endif
		absdiff(shiftedpatch, patch_wdhist2[1], patch_equal1);//Diff with compensated shift but without warp
		patch_equal1_tmp = patch_equal1.clone();

		//Generate warped image diff
		double errlevels[2] = {0, 0};
		Mat border[4];
		double minmaxXY[4];
		cv::Mat patch_wdhist1, diffImg2, warped_patch0;
		if(flowGtIsUsed)
		{
			warpAffine(patch_wdhist2[0],warped_patch0,Haff,patch_wdhist2[0].size(), INTER_LANCZOS4); //Warp the left patch
		}
		else
		{
			warpPerspective(patch_wdhist2[0], warped_patch0, Haff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Warp the left patch
		}

		iterativeTemplMatch(warped_patch0, largerpatchhist, patchsizeOrig, pdiff[1]);

		compDiff = Mat::eye(2,2,CV_64F);
		tdiff = (Mat_<double>(2,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y));
		cv::hconcat(compDiff, tdiff, compDiff);
		warpAffine(warped_patch0, warped_patch0, compDiff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Shift the left patch
		{
			cv::Point2f newpdiff = cv::Point2f(0,0);
			findLocalMin(warped_patch0, patch_wdhist2[1], quarterpatch, eigthpatch, newpdiff, patchsizeOrig);
			compDiff = Mat::eye(2,2,CV_64F);
			Mat tdiff2 = (Mat_<double>(2,1) << static_cast<double>(newpdiff.x), static_cast<double>(newpdiff.y));
			cv::hconcat(compDiff, tdiff2, compDiff);
			warpAffine(warped_patch0, warped_patch0, compDiff, warped_patch0.size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
			pdiff[1] += newpdiff;
			tdiff = (Mat_<double>(2,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y));
		}
		absdiff(warped_patch0, patch_wdhist2[1], patch_wdhist1);//Diff with compensated shift and warped
		for(int k = 0; k < 2; k++) //get the new (smaller) patch size after warping
		{
			for(int m = 0; m < 2; m++)
			{
				Mat tmp;
				int midx = k * 2 + m;
				if(flowGtIsUsed)
				{
					origPos = (Mat_<double>(2,1) << static_cast<double>(k) * static_cast<double>(patchsizeOrig), static_cast<double>(m) * static_cast<double>(patchsizeOrig));
					border[midx] = Haff.colRange(0,2) * origPos;
					border[midx] += tm;
				}
				else
				{
					origPos = (Mat_<double>(3,1) << static_cast<double>(k) * static_cast<double>(patchsizeOrig), static_cast<double>(m) * static_cast<double>(patchsizeOrig), 1.0);
					border[midx] = Haff * origPos;
					border[midx] /= border[midx].at<double>(2);
					border[midx] = border[midx].rowRange(0,2);
				}
				border[midx].at<double>(0) = border[midx].at<double>(0) > patchsizeOrig ? static_cast<double>(patchsizeOrig):border[midx].at<double>(0);
				border[midx].at<double>(0) = border[midx].at<double>(0) < 0 ? 0:border[midx].at<double>(0);
				border[midx].at<double>(1) = border[midx].at<double>(1) > patchsizeOrig ? static_cast<double>(patchsizeOrig):border[midx].at<double>(1);
				border[midx].at<double>(1) = border[midx].at<double>(1) < 0 ? 0:border[midx].at<double>(1);
				//tmp = (Mat_<double>(2,1) << (k ? (double)min(pdiff[0].x, pdiff[1].x):(double)max(pdiff[0].x, pdiff[1].x)), (m ? (double)min(pdiff[0].y, pdiff[1].y):(double)max(pdiff[0].y, pdiff[1].y)));//from shifting
				//border[midx] += tmp;
			}
		}
		{
			vector<std::pair<int,double>> border_tmp;
			for(int k = 0; k < 4; k++)
				border_tmp.emplace_back(k, border[k].at<double>(0));
			sort(border_tmp.begin(), border_tmp.end(),
				[](pair<int,double> first, pair<int,double> second){return first.second < second.second;});
			border[border_tmp[0].first].at<double>(0) += static_cast<double>(max(pdiff[0].x, pdiff[1].x));
			border[border_tmp[1].first].at<double>(0) += static_cast<double>(max(pdiff[0].x, pdiff[1].x));
			border[border_tmp[2].first].at<double>(0) += static_cast<double>(min(pdiff[0].x, pdiff[1].x));
			border[border_tmp[3].first].at<double>(0) += static_cast<double>(min(pdiff[0].x, pdiff[1].x));
			border_tmp.clear();
			for(int k = 0; k < 4; k++)
				border_tmp.emplace_back(k, border[k].at<double>(1));
			sort(border_tmp.begin(), border_tmp.end(),
				[](pair<int,double> first, pair<int,double> second){return first.second < second.second;});
			border[border_tmp[0].first].at<double>(1) += static_cast<double>(max(pdiff[0].y, pdiff[1].y));
			border[border_tmp[1].first].at<double>(1) += static_cast<double>(max(pdiff[0].y, pdiff[1].y));
			border[border_tmp[2].first].at<double>(1) += static_cast<double>(min(pdiff[0].y, pdiff[1].y));
			border[border_tmp[3].first].at<double>(1) += static_cast<double>(min(pdiff[0].y, pdiff[1].y));
		}

		std::vector<cv::Point2d> pntsPoly1, pntsPoly2, pntsRes;
		cv::Point2d retLT, retRB;
		pntsPoly1.emplace_back(0,0);
		pntsPoly1.emplace_back(patchsizeOrig,0);
		pntsPoly1.emplace_back(patchsizeOrig,patchsizeOrig);
		pntsPoly1.emplace_back(0,patchsizeOrig);
		pntsPoly2.emplace_back(border[0].at<double>(0),border[0].at<double>(1));
		pntsPoly2.emplace_back(border[2].at<double>(0),border[2].at<double>(1));
		pntsPoly2.emplace_back(border[3].at<double>(0),border[3].at<double>(1));
		pntsPoly2.emplace_back(border[1].at<double>(0),border[1].at<double>(1));
		intersectPolys(pntsPoly1, pntsPoly2, pntsRes);
		if(maxInscribedRect(pntsRes, retLT, retRB))
		{
			minmaxXY[0] = retLT.x;//minX
			minmaxXY[1] = retRB.x;//maxX
			minmaxXY[2] = retLT.y;//minY
			minmaxXY[3] = retRB.y;//maxY
		}
		else
		{
			minmaxXY[0] = max(border[0].at<double>(0), border[1].at<double>(0));//minX
			minmaxXY[1] = min(border[2].at<double>(0), border[3].at<double>(0));//maxX
			minmaxXY[2] = max(border[0].at<double>(1), border[2].at<double>(1));//minY
			minmaxXY[3] = min(border[1].at<double>(1), border[3].at<double>(1));//maxY
		}

		for(int k = 0; k < 4; k++)
		{
			minmaxXY[k] = std::floor(minmaxXY[k] + 0.5);
			minmaxXY[k] = minmaxXY[k] < 0 ? 0:minmaxXY[k];
			if(k % 2 == 0)
				minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? 0:minmaxXY[k];
			else
				minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? patchsizeOrig:minmaxXY[k];
		}
		if((minmaxXY[0] >= minmaxXY[1]) || (minmaxXY[2] >= minmaxXY[3]))
			errlevels[0] = -1.0;
		else
		{
			patch_wdhist1 = patch_wdhist1(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2])).clone(); //with warp
			patch_equal1 = patch_equal1(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2])).clone(); //without warp
			errlevels[0] = cv::sum(patch_equal1)[0];
			errlevels[1] = cv::sum(patch_wdhist1)[0];
		}
		if(errlevels[0] < errlevels[1])//Check if the diff without warp is better
		{
			minmaxXY[0] = static_cast<double>(pdiff[0].x);//minX
			minmaxXY[1] = static_cast<double>(patchsizeOrig) + static_cast<double>(pdiff[0].x);//maxX
			minmaxXY[2] = static_cast<double>(pdiff[0].y);//minY
			minmaxXY[3] = static_cast<double>(patchsizeOrig) + static_cast<double>(pdiff[0].y);//maxY
			for(int k = 0; k < 4; k++)
			{
				minmaxXY[k] = std::floor(minmaxXY[k] + 0.5);
				minmaxXY[k] = minmaxXY[k] < 0 ? 0:minmaxXY[k];
				if(k % 2 == 0)
					minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? 0:minmaxXY[k];
				else
					minmaxXY[k] = minmaxXY[k] > patchsizeOrig ? patchsizeOrig:minmaxXY[k];
			}
			patch_wdhist1 = patch_equal1_tmp(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2]));
			warped_patch0 = shiftedpatch;
			pdiff[1] = -pdiff[0];
		}
		else
		{
			if(flowGtIsUsed)
			{
				tdiff = Haff.colRange(0,2).inv() * (-tdiff);//Calculate the displacement in pixels of the original image (not warped)
			}
			else
			{
				pdiff[1] = -pdiff[1];
				tdiff = (Mat_<double>(3,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y), 1.0);

				Mat Haffi = Mat::eye(3,3,homoGT_exch.type());
				Haffi.at<double>(0,2) = static_cast<double>(rkp.x);
				Haffi.at<double>(1,2) = static_cast<double>(rkp.y);
				Haffi = homoGT_exch.inv() * Haffi;
				Mat helpH1 = Mat::eye(3,3,homoGT_exch.type());
				helpH1.at<double>(0,2) = -Haffi.at<double>(0,2) / Haffi.at<double>(2,2);
				helpH1.at<double>(1,2) = -Haffi.at<double>(1,2) / Haffi.at<double>(2,2);
				Haffi = helpH1 * Haffi;
				tdiff = Haffi * tdiff;//Calculate the displacement in pixels of the original image (not warped)
				tdiff /= tdiff.at<double>(2);
			}
			pdiff[1].x = tdiff.at<double>(0);
			pdiff[1].y = tdiff.at<double>(1);
		}
		cv::cvtColor(patch_wdhist1, diffImg2, cv::COLOR_GRAY2RGB);

		//Get thresholded and eroded image
#if defined(USE_MANUAL_ANNOTATION)
        Mat resultimg_color;
#endif
		Mat resultimg, erdielement;
		threshold( patch_wdhist1, resultimg, threshhTh, 255.0, 3 );//3: Threshold to Zero; th-values: 64 for normal images, 20 for synthetic
		erdielement = getStructuringElement( cv::MORPH_RECT, Size( 2, 2 ));
		erode(resultimg, resultimg, erdielement );
		dilate(resultimg, resultimg, erdielement );
		int nrdiff[2];
		nrdiff[0] = cv::countNonZero(resultimg);
		Mat resultimg2 = resultimg(Rect(resultimg.cols / 4, resultimg.rows / 4, resultimg.cols / 2, resultimg.rows / 2));
		nrdiff[1] = cv::countNonZero(resultimg2);
		double badFraction[2];
		badFraction[0] = static_cast<double>(nrdiff[0]) / static_cast<double>(resultimg.rows * resultimg.cols);
		badFraction[1] = static_cast<double>(nrdiff[1]) / static_cast<double>(resultimg2.rows * resultimg2.cols);
#if defined(USE_MANUAL_ANNOTATION)
		cv::cvtColor(resultimg, resultimg_color, cv::COLOR_GRAY2RGB);
#endif

		cv::Point2f singleErrVec = pdiff[1];
		auto diffdist = static_cast<double>(std::sqrt(pdiff[1].x * pdiff[1].x + pdiff[1].y * pdiff[1].y));
		if(!(((badFraction[0] < badFraction[1]) && (badFraction[1] > 0.05)) || (badFraction[0] > 0.18) || (badFraction[1] > 0.1) || (diffdist >= usedMatchTH)) || ((diffdist < 0.7) && (badFraction[1] < 0.04)))
		{
			stringstream ss;
//#ifdef _DEBUG
//			ss << "OK! Diff: " << diffdist;
//			str = ss.str();
//			putText(composed, str.c_str(), cv::Point2d(15, 70), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
//#endif
			distances.push_back(diffdist);
			autoManualAnno.push_back('A');
			int k = 1;
			while((k < static_cast<int>(distanceHisto.size())) && (distanceHisto[k].first < diffdist))
				k++;
			distanceHisto[k - 1].second++;
			errvecs.push_back(singleErrVec);
            matchesGT_idx.push_back(idx);
#if LEFT_TO_RIGHT
			perfectMatches.emplace_back(std::make_pair(lkp + singleErrVec, rkp));
#else
			perfectMatches.emplace_back(rkp, lkp + singleErrVec);
#endif
			calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
			distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
			if(diffdist >= usedMatchTH)
			{
#if LEFT_TO_RIGHT
				falseGT.emplace_back(std::make_pair(lkp, rkp));
#else
				falseGT.emplace_back(rkp, lkp);
#endif
				wrongGTidxDist.emplace_back(idx, diffdist);
				if(flowGtIsUsed)
				{
#if LEFT_TO_RIGHT
					validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(lkp.y + 0.5f)), static_cast<int>(floor(lkp.x + 0.5f))));
#else
					validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(rkp.y + 0.5f)), static_cast<int>(floor(rkp.x + 0.5f))));
#endif
				}
				else
				{
					validityValFalseGT.push_back(1);
				}
			}
//#ifndef _DEBUG
			continue;
//#endif
		}else {
#if defined(USE_MANUAL_ANNOTATION)//If defined, found GT matches can be additionally annotated manually
            autoManualAnno.push_back('M');
            stringstream ss;
#ifndef _DEBUG
            ss << "Diff: " << diffdist;
#else
            ss << "Fail! Diff: " << diffdist;
#endif
            str = ss.str();
            putText(composed, str.c_str(), cv::Point2d(15, 90), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));

            //Mark "new" matching position in left patch
            Mat warped_patch0_c;
            cv::cvtColor(warped_patch0, warped_patch0_c, cv::COLOR_GRAY2RGB);
            cv::Point2f newmidpos = cv::Point2f(floor(static_cast<float>(warped_patch0.cols) / 2.f), floor(static_cast<float>(warped_patch0.rows) / 2.f));
            /*cv::line(warped_patch0_c, newmidpos - Point2f(5.0f, 0), newmidpos + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(warped_patch0_c, newmidpos - Point2f(0, 5.0f), newmidpos + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));*/

            //Show right patch with equalized histogram
            Mat patch_wdhist2_color, patch_wdhist2_color2;
            cv::cvtColor(patch_wdhist2[1], patch_wdhist2_color, cv::COLOR_GRAY2RGB);
            patch_wdhist2_color2 = patch_wdhist2_color.clone();
            newmidpos = cv::Point2f(floor(static_cast<float>(patch_wdhist2[1].cols) / 2.f), floor(static_cast<float>(patch_wdhist2[1].rows) / 2.f));
            /*cv::line(patch_wdhist2_color, newmidpos - Point2f(5.0f, 0), newmidpos + Point2f(5.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(patch_wdhist2_color, newmidpos - Point2f(0, 5.0f), newmidpos + Point2f(0, 5.0f), cv::Scalar(0, 0, 255));*/

            //Show left patch with equalized histogram
            Mat leftHist_color;
            cv::cvtColor(patch_wdhist2[0], leftHist_color, cv::COLOR_GRAY2RGB);

            //Generate warped, shifted & blended patches
            Mat patches_warp[2], wsp_color[2], blended_w;
            cv::cvtColor(warped_patch0, wsp_color[0], cv::COLOR_GRAY2RGB);
            cv::cvtColor(patch_wdhist2[1], wsp_color[1], cv::COLOR_GRAY2RGB);
            wsp_color[0].copyTo(patches_warp[0]);
            wsp_color[1].copyTo(patches_warp[1]);
            patches_warp[0].reshape(1, patches_warp[0].rows * patches_warp[0].cols).col(1).setTo(Scalar(0));
            patches_warp[0].reshape(1, patches_warp[0].rows * patches_warp[0].cols).col(0).setTo(Scalar(0));
            patches_warp[1].reshape(1, patches_warp[1].rows * patches_warp[1].cols).col(0).setTo(Scalar(0));
            patches_warp[1].reshape(1, patches_warp[1].rows * patches_warp[1].cols).col(2).setTo(Scalar(0));
            addWeighted(patches_warp[0], alpha, patches_warp[1], beta, 0.0, blended_w);
            blended_w = blended_w(Rect(minmaxXY[0], minmaxXY[2], minmaxXY[1] - minmaxXY[0], minmaxXY[3] - minmaxXY[2])).clone();


            //Generate both images including the marked match
            Mat bothimgs(imgSize.height, 2 * imgSize.width, CV_8UC3);
#if LEFT_TO_RIGHT
            str = "I1";
#else
            str = "I2";
#endif
            putText(show_color[0], str.c_str(), cv::Point2d(20, 20), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.6, cv::Scalar(0, 0, 255));
            show_color[0].copyTo(bothimgs(Rect(Point(0, 0), imgSize)));
#if LEFT_TO_RIGHT
            str = "I2";
#else
            str = "I1";
#endif
            putText(show_color[1], str.c_str(), cv::Point2d(20, 20), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.6, cv::Scalar(0, 0, 255));
            show_color[1].copyTo(bothimgs(Rect(Point(imgSize.width, 0), imgSize)));
            cv::line(bothimgs, lkp, rkp + Point2f(imgSize.width, 0), cv::Scalar(0, 0, 255), 2);

            //Scale images and form a single image out of them
            cv::resize(bothimgs, composed(Rect(0, textheight, maxHorImgSize * 2, maxVerImgSize)), cv::Size(maxHorImgSize * 2, maxVerImgSize), 0, 0, INTER_LANCZOS4);
            cv::resize(blended, composed(Rect(0, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow));
            newmidpos = cv::Point2f(floor(static_cast<float>(patchsizeShow) / 2.f), floor(static_cast<float>(patchsizeShow) / 2.f));
            cv::line(composed(Rect(0, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(composed(Rect(0, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
            str = "blended";
            putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(diffImg, composed(Rect(patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            str = "Diff";
            putText(composed, str.c_str(), cv::Point2d(patchsizeShow + 5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(patches[0], composed(Rect(2 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            cv::line(composed(Rect(2 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(composed(Rect(2 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
            str = "left";
#else
            str = "right";
#endif
            putText(composed, str.c_str(), cv::Point2d(2 * patchsizeShow + 5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(patches[1], composed(Rect(3 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            cv::line(composed(Rect(3 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(composed(Rect(3 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
            str = "right";
#else
            str = "left";
#endif
            putText(composed, str.c_str(), cv::Point2d(3 * patchsizeShow + 5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(warped_patch0_c, composed(Rect(4 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            cv::line(composed(Rect(4 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(composed(Rect(4 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
            str = "NEW left position";
#else
            str = "NEW right position";
#endif
            putText(composed, str.c_str(), cv::Point2d(4 * patchsizeShow + 5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(patch_wdhist2_color, composed(Rect(5 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            cv::line(composed(Rect(5 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(composed(Rect(5 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
            str = "right equal hist";
#else
            str = "left equal hist";
#endif
            putText(composed, str.c_str(), cv::Point2d(5 * patchsizeShow + 5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(blended_w, composed(Rect(6 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            str = "NEW (warped) blended";
            putText(composed, str.c_str(), cv::Point2d(6 * patchsizeShow + 5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(diffImg2, composed(Rect(7 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            str = "NEW (warped) diff";
            putText(composed, str.c_str(), cv::Point2d(7 * patchsizeShow + 5, maxVerImgSize + textheight + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            /*cv::resize(resultimg_color, composed(Rect(8 * patchsizeShow, maxVerImgSize + textheight, patchsizeShow, patchsizeShow)), cv::Size(patchsizeShow, patchsizeShow), 0, 0, INTER_LANCZOS4);
            str = "Result";
            putText(composed, str.c_str(), cv::Point2d(9 * patchsizeShow + 5, maxVerImgSize + textheight + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));*/
            cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
            newmidpos = cv::Point2f(floor(static_cast<float>(patchSelectShow) / 2.f), floor(static_cast<float>(patchSelectShow) / 2.f));
            cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 140, 0));
            cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 140, 0));
#if LEFT_TO_RIGHT
            str = "left equal hist - select pt";
#else
            str = "right equal hist - select pt";
#endif
            putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
            cv::resize(patch_wdhist2_color2, composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
            cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
            cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
            str = "right equal hist";
#else
            str = "left equal hist";
#endif
            putText(composed, str.c_str(), cv::Point2d(patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));

            str = "Press 'space' to accept the estimated or selected (preferred, if available) location, 'e' for the estimated, and 'i' for the initial location. Hit 'n' if not matchable, 's' to skip the match, and";
            putText(composed, str.c_str(), cv::Point2d(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
            str = "ESC to stop testing. To specify a new matching position, click at the desired position inside the area of 'left equal hist - select pt'. To refine the location, use the arrow keys.";
#else
            str = "ESC to stop testing. To specify a new matching position, click at the desired position inside the area of 'right equal hist - select pt'. To refine the location, use the arrow keys.";
#endif
            putText(composed, str.c_str(), cv::Point2d(15, 35), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
            str = "To select a new patch for manual refinement at a different location, hold 'Strg' while clicking at the desired center position inside the left image 'I1'. Press 'h' for more options.";
#else
            str = "To select a new patch for manual refinement at a different location, hold 'Strg' while clicking at the desired center position inside the left image 'I2'. Press 'h' for more options.";
#endif
            putText(composed, str.c_str(), cv::Point2d(15, 55), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
            {
                stringstream ss;
                ss << "Remaining matches: " << maxSampleSize - i - 1 << "  Remaining Images: " << remainingImgs;
                str = ss.str();
                putText(composed, str.c_str(), cv::Point2d(15, 73), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
            }

            cv::namedWindow("GT match");
            cv::Point2f winPos = cv::Point2f(-FLT_MAX, -FLT_MAX), winPos2 = cv::Point2f(-FLT_MAX, -FLT_MAX), winPosOld = cv::Point2f(-FLT_MAX, -FLT_MAX);
            cv::setMouseCallback("GT match", on_mouse_click, (void*)(&winPos) );
            cv::imshow("GT match",composed);
            minmaxXY[0] = 0;//minX
            minmaxXY[1] = patchSelectShow;//maxX
            minmaxXY[2] = maxVerImgSize + textheight + patchsizeShow;//minY
            minmaxXY[3] = minmaxXY[2] + patchSelectShow;//maxY
            double shownLeftImgBorder[4];
            double diffdist2 = 0, diffdist2last = 0;
            cv::Point2f singleErrVec2;
            unsigned int refiters = 0;
            int c;
            bool noIterRef = false;
            static bool noIterRefProg = false;
            unsigned int manualMode = 3, diffManualMode = 3;
            bool manualHomoDone = false;
            int autocalc2 = 0;
            cv::Mat Hmanual, Hmanual2 = Mat::eye(2,3,CV_64FC1);
            double manualScale = 1.0, manualRot = 0;
            cv::Point2f oldpos = cv::Point2f(0,0);;
            SpecialKeyCode skey = NONE;
            pdiff[1] = cv::Point2f(0,0);
            shownLeftImgBorder[0] = 0;//minX
            shownLeftImgBorder[1] = maxHorImgSize;//maxX
            shownLeftImgBorder[2] = textheight;//minY
            shownLeftImgBorder[3] = textheight + maxVerImgSize;//maxY
            do
            {
                c = cv::waitKey(30);

                if(c != -1)
                    skey = getSpecialKeyCode(c);
                else
                    skey = NONE;

                if(!manualHomoDone)
                {
                    manualHomoDone = true;
                    if(manualMode != 3)//Use homography for left shown image
                    {
                            equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
                            switch(manualMode)
                            {
                            case 0://Use the homography from GT or the one that was calculated using matches within the left and right patches and properties of SIFT keypoints
                                if(flowGtIsUsed)
                                    warpAffine(patch_wdhist2[0], patch_wdhist2[0], Haff, patch_wdhist2[0].size(), INTER_LANCZOS4);
                                else
                                    warpPerspective(patch_wdhist2[0], patch_wdhist2[0], Haff, patch_wdhist2[0].size(), INTER_LANCZOS4);
                                break;
                            case 1://Use the homography that was calculated using matches within the left and right patches only
                                if(Hl.empty())
                                {
#if __linux__
                                    QMessageBox msgBox;
                                    msgBox.setText("Homography not available");
                                    msgBox.setInformativeText("The feature-based homography is not available for this match!");
                                    msgBox.exec();
#else
                                    MessageBox(NULL, "The feature-based homography is not available for this match!", "Homography not available", MB_OK | MB_ICONINFORMATION);
#endif
                                    manualMode = 3;
                                    manualHomoDone = false;
                                }
                                else
                                {
                                    Hl = Hl.colRange(0,2);
                                    Mat origPos1 = (Mat_<double>(2,1) << static_cast<double>(halfpatch), static_cast<double>(halfpatch));
                                    Mat tm1;
                                    tm1 = Hl * origPos1;//align the translkation vector of the homography in a way that the middle of the patch has the same coordinate
                                    tm1 = origPos1 - tm1;
                                    cv::hconcat(Hl, tm1, Hl);//Construct the final affine homography
                                    warpAffine(patch_wdhist2[0], patch_wdhist2[0], Hl, patch_wdhist2[0].size(), INTER_LANCZOS4);
                                }
                                break;
                            case 2://Select 4 points to calculate a new homography
                                {
#if __linux__
                                    QMessageBox msgBox;
                                    msgBox.setText("Manual homography estimation");
                                    msgBox.setInformativeText("Select 4 matching points in each patch to generate a homography (in the order 1st left, 1st right, 2nd left, 2nd right, ...). They must not be on a line! To abort while selecting, press 'ESC'.\", \"Manual homography estimation");
                                    msgBox.exec();
#else
                                    MessageBox(NULL, "Select 4 matching points in each patch to generate a homography (in the order 1st left, 1st right, 2nd left, 2nd right, ...). They must not be on a line! To abort while selecting, press 'ESC'.", "Manual homography estimation", MB_OK | MB_ICONINFORMATION);
#endif
                                    cv::cvtColor(patch_wdhist2[0], leftHist_color, cv::COLOR_GRAY2RGB);
                                    cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                                    cv::resize(patch_wdhist2_color2, composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                                    Mat selectPatches[2];
                                    vector<cv::Point2f> lps, rps;
                                    cv::resize(leftHist_color, selectPatches[0], cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                                    cv::resize(patch_wdhist2_color2, selectPatches[1], cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                                    string numstr[4] = {"first","second","third","fourth"};
                                    cv::Scalar colours[4] = {cv::Scalar(0, 0, 255),cv::Scalar(0, 255, 0),cv::Scalar(255, 0, 0),cv::Scalar(209, 35, 233)};
                                    double minmaxXYrp[2];
                                    bool noSkipmode = true;
                                    minmaxXYrp[0] = patchSelectShow;//minX
                                    minmaxXYrp[1] = 2* patchSelectShow;//maxX
                                    for(int k = 0; k < 4; k++)
                                    {
                                        str = "Select " + numstr[k] + " point";
                                        putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
                                        winPos = cv::Point2f(-FLT_MAX, -FLT_MAX);
                                        cv::imshow("GT match",composed);
                                        bool lineardep = false;
                                        do
                                        {
                                            lineardep = false;
                                            while((winPos.x < minmaxXY[0]) || (winPos.x >= minmaxXY[1]) || (winPos.y < minmaxXY[2]) || (winPos.y >= minmaxXY[3]))
                                            {
                                                c = cv::waitKey(30);
                                                if(c != -1)
                                                    skey = getSpecialKeyCode(c);
                                                else
                                                    skey = NONE;
                                                if(skey == ESCAPE)
                                                {
                                                    skey = NONE;
                                                    c = -1;
                                                    k = 4;
                                                    noSkipmode = false;
                                                    break;
                                                }
                                            }
                                            if(noSkipmode)
                                            {
                                                lps.emplace_back(cv::Point2f(winPos.x, winPos.y - (maxVerImgSize + textheight + patchsizeShow)));
                                                // check that the i-th selected point does not belong
                                                // to a line connecting some previously selected points
                                                for(int j = 0; j < k; j++ )
                                                {
                                                    double dx1 = lps[j].x - lps.back().x;
                                                    double dy1 = lps[j].y - lps.back().y;
                                                    for(int k1 = 0; k1 < j; k1++ )
                                                    {
                                                        double dx2 = lps[k1].x - lps.back().x;
                                                        double dy2 = lps[k1].y - lps.back().y;
                                                        if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                                                            lineardep = true;
                                                    }
                                                }
                                                if(lineardep)
                                                {
                                                    lps.pop_back();
#if __linux__
                                                    QMessageBox msgBox1;
                                                    msgBox1.setText("Linear dependency");
                                                    msgBox1.setInformativeText("Selection is linear dependent - select a different one!");
                                                    msgBox1.exec();
#else
                                                    MessageBox(NULL, "Selection is linear dependent - select a different one!", "Linear dependency", MB_OK | MB_ICONINFORMATION);
#endif
                                                }
                                            }
                                        }
                                        while(lineardep);
                                        if(noSkipmode)
                                        {
                                            cv::line(selectPatches[0], lps.back() - Point2f(10.0f, 0), lps.back() + Point2f(10.0f, 0), colours[k]);
                                            cv::line(selectPatches[0], lps.back() - Point2f(0, 10.0f), lps.back() + Point2f(0, 10.0f), colours[k]);
                                            selectPatches[0].copyTo(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)));
                                            //cv::resize(selectPatches[0], composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                                            lps.back().x /= static_cast<float>(selMult);
                                            lps.back().y /= static_cast<float>(selMult);

                                            putText(composed, str.c_str(), cv::Point2d(patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
                                            cv::imshow("GT match",composed);
                                        }
                                        while(((winPos.x < minmaxXYrp[0]) || (winPos.x >= minmaxXYrp[1]) || (winPos.y < minmaxXY[2]) || (winPos.y >= minmaxXY[3])) && noSkipmode)
                                        {
                                            c = cv::waitKey(30);
                                            if(c != -1)
                                                skey = getSpecialKeyCode(c);
                                            else
                                                skey = NONE;
                                            if(skey == ESCAPE)
                                            {
                                                skey = NONE;
                                                c = -1;
                                                k = 4;
                                                noSkipmode = false;
                                                break;
                                            }
                                        }
                                        if(noSkipmode)
                                        {
                                            rps.emplace_back(cv::Point2f(winPos.x - patchSelectShow, winPos.y - (maxVerImgSize + textheight + patchsizeShow)));
                                            cv::line(selectPatches[1], rps.back() - Point2f(10.0f, 0), rps.back() + Point2f(10.0f, 0), colours[k]);
                                            cv::line(selectPatches[1], rps.back() - Point2f(0, 10.0f), rps.back() + Point2f(0, 10.0f), colours[k]);
                                            selectPatches[1].copyTo(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)));
                                            //cv::resize(selectPatches[1], composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                                            rps.back().x /= static_cast<float>(selMult);
                                            rps.back().y /= static_cast<float>(selMult);
                                        }
                                    }
                                    if(noSkipmode)
                                    {
                                        cv::imshow("GT match",composed);
                                        c = cv::waitKey(1300);
                                        Hmanual = cv::estimateAffine2D(lps, rps);//estimate an affine homography
                                        if(Hmanual.empty())
                                        {
#if __linux__
                                            QMessageBox msgBox1;
                                            msgBox1.setText("Homography not available");
                                            msgBox1.setInformativeText("It was not possible to estimate a homography from these correspondences!");
                                            msgBox1.exec();
#else
                                            MessageBox(NULL, "It was not possible to estimate a homography from these correspondences!", "Homography not available", MB_OK | MB_ICONINFORMATION);
#endif
                                            manualMode = 3;
                                            manualHomoDone = false;
                                        }
                                        else
                                        {
                                            Mat Hmanual1 = Hmanual.colRange(0,2);
                                            Mat origPos1 = (Mat_<double>(2,1) << static_cast<double>(halfpatch), static_cast<double>(halfpatch));
                                            Mat tm1;
                                            tm1 = Hmanual1 * origPos1;//align the translation vector of the homography in a way that the middle of the patch has the same coordinate
                                            tm1 = origPos1 - tm1;
                                            cv::hconcat(Hmanual1, tm1, Hmanual1);//Construct the final affine homography
                                            warpAffine(patch_wdhist2[0], patch_wdhist2[0], Hmanual1, patch_wdhist2[0].size(), INTER_LANCZOS4);
                                            winPos.x = static_cast<float>(selMult) * (static_cast<float>(Hmanual.at<double>(0,2)) - static_cast<float>(tm1.at<double>(0)) + halfpatch);
                                            winPos.y = static_cast<float>(selMult) * (static_cast<float>(Hmanual.at<double>(1,2)) - static_cast<float>(tm1.at<double>(1)) + halfpatch) + static_cast<float>(maxVerImgSize + textheight + patchsizeShow);
                                            noIterRef = false;
                                        }
                                    }
                                    cv::resize(patch_wdhist2_color2, composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                                    cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
                                    cv::line(composed(Rect(patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
                                    str = "right equal hist";
#else
                                    str = "left equal hist";
#endif
                                    putText(composed, str.c_str(), cv::Point2d(patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
                                break;
                                }
                                case 4://Use the homography that was calculated using the user input for scale and rotation
                                {
                                    Hmanual2 = Hmanual2.colRange(0,2);
                                    Mat origPos1 = (Mat_<double>(2,1) << static_cast<double>(halfpatch), static_cast<double>(halfpatch));
                                    Mat tm1;
                                    tm1 = Hmanual2 * origPos1;//align the translkation vector of the homography in a way that the middle of the patch has the same coordinate
                                    tm1 = origPos1 - tm1;
                                    cv::hconcat(Hmanual2, tm1, Hmanual2);//Construct the final affine homography
                                    warpAffine(patch_wdhist2[0], patch_wdhist2[0], Hmanual2, patch_wdhist2[0].size(), INTER_LANCZOS4);
                                break;
                                }
                            }
                            cv::cvtColor(patch_wdhist2[0], leftHist_color, cv::COLOR_GRAY2RGB);
                    }
                    else
                    {
                        equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
                        cv::cvtColor(patch_wdhist2[0], leftHist_color, cv::COLOR_GRAY2RGB);
                    }
                    cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                    cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 140, 0));
                    cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 140, 0));
#if LEFT_TO_RIGHT
                    str = "left equal hist - select pt";
#else
                    str = "right equal hist - select pt";
#endif
                    putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));
                    cv::imshow("GT match",composed);
                }

                if(autocalc2 > 0)//find a global minimum (best template match within patch) on the patches for manual selection
                {
                    iterativeTemplMatch(patch_wdhist2[0], largerpatchhist, patchsizeOrig, winPos, autocalc2);
                    winPos = -winPos;
                    winPos.y += halfpatch;
                    winPos.x += halfpatch;
                    winPos.y *= static_cast<float>(selMult);
                    winPos.x *= static_cast<float>(selMult);
                    winPos.y += (float)(maxVerImgSize + textheight + patchsizeShow);
                    noIterRef = true;//deactivate local refinement
                    autocalc2 = 0;
                }

                if((winPos.x >= minmaxXY[0]) && (winPos.x < minmaxXY[1]) && (winPos.y >= minmaxXY[2]) && (winPos.y < minmaxXY[3]))//if a position inside the "manual selection" area was chosen
                {
                    cv::Point2f newmidpos2, singleErrVec2Old;
                    int autseait = 0;
                    cv::Point2f addWinPos = cv::Point2f(0,0);
                    int direction = 0;//0=left, 1=right, 2=up, 3=down
                    int oldDir[4] = {0,0,0,0};
                    int horVerCnt[2] = {0,0};
                    int atMinimum = 0;
                    double diffdist2Old;

                    if(skey == ARROW_UP) //up key is pressed
                        winPos.y -= 0.25f;
                    else if(skey == ARROW_DOWN) //down key is pressed
                        winPos.y += 0.25f;
                    else if(skey == ARROW_LEFT) //left key is pressed
                        winPos.x -= 0.25f;
                    else if(skey == ARROW_RIGHT) //right key is pressed
                        winPos.x += 0.25f;

                    winPos2 = winPos;
                    do
                    {
                        cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                        if((abs(pdiff[1].x) <= 10 * FLT_MIN) && (abs(pdiff[1].y) <= 10 * FLT_MIN))
                        {
                            cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 140, 0));
                            cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 140, 0));
                        }

                        newmidpos2 = cv::Point2f(winPos.x, winPos.y - (maxVerImgSize + textheight + patchsizeShow));
                        cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos2 - Point2f(10.0f, 0), newmidpos2 + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
                        cv::line(composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos2 - Point2f(0, 10.0f), newmidpos2 + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
#if LEFT_TO_RIGHT
                        str = "left equal hist - select pt";
#else
                        str = "right equal hist - select pt";
#endif
                        putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));

                        //shift left img
                        compDiff = Mat::eye(2,2,CV_64F);
                        pdiff[0] = cv::Point2f((float)(newmidpos.x - newmidpos2.x) / static_cast<float>(selMult), (float)(newmidpos.y - newmidpos2.y) / static_cast<float>(selMult));
                        //newmidpos2 = pdiff[1] - pdiff[0];
                        if(refiters == 0)
                        {
                            diffManualMode = manualMode;
                        }

                        bool recalc;
                        do
                        {
                            recalc = false;
                            if(diffManualMode == 3)
                            {
                                newmidpos2 = pdiff[1] - pdiff[0];
                                singleErrVec2 = newmidpos2;
                                diffdist2 = static_cast<double>(std::sqrt(newmidpos2.x * newmidpos2.x + newmidpos2.y * newmidpos2.y));
                            }
                            else if(diffManualMode == 2)
                            {
                                newmidpos2 = -pdiff[0];
                                Mat tdiff1 = (Mat_<double>(2,1) << static_cast<double>(newmidpos2.x), static_cast<double>(newmidpos2.y));
                                tdiff1 = Hmanual.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
                                tdiff1 += (Mat_<double>(2,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y));
                                diffdist2 = static_cast<double>(std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1)));
                                singleErrVec2 = cv::Point2f(static_cast<float>(tdiff1.at<double>(0)), static_cast<float>(tdiff1.at<double>(1)));
                            }
                            else if(diffManualMode == 1)
                            {
                                newmidpos2 = -pdiff[0];
                                Mat tdiff1 = (Mat_<double>(2,1) << static_cast<double>(newmidpos2.x), static_cast<double>(newmidpos2.y));
                                tdiff1 = Hl.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
                                tdiff1 += (Mat_<double>(2,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y));
                                diffdist2 = static_cast<double>(std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1)));
                                singleErrVec2 = cv::Point2f(static_cast<float>(tdiff1.at<double>(0)), static_cast<float>(tdiff1.at<double>(1)));
                            }
                            else if(diffManualMode == 0)
                            {
                                Mat tdiff1;
                                newmidpos2 = -pdiff[0];
                                if(flowGtIsUsed)
                                {
                                    tdiff1 = (Mat_<double>(2,1) << static_cast<double>(newmidpos2.x), static_cast<double>(newmidpos2.y));
                                    tdiff1 = Haff.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
                                    tdiff1 += (Mat_<double>(2,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y));
                                }
                                else
                                {
                                    tdiff1 = (Mat_<double>(3,1) << static_cast<double>(newmidpos2.x), static_cast<double>(newmidpos2.y), 1.0);

                                    Mat Haffi = Mat::eye(3,3,homoGT_exch.type());
                                    Haffi.at<double>(0,2) = static_cast<double>(rkp.x);
                                    Haffi.at<double>(1,2) = static_cast<double>(rkp.y);
                                    Haffi = homoGT_exch.inv() * Haffi;
                                    Mat helpH1 = Mat::eye(3,3,homoGT_exch.type());
                                    helpH1.at<double>(0,2) = -Haffi.at<double>(0,2) / Haffi.at<double>(2,2);
                                    helpH1.at<double>(1,2) = -Haffi.at<double>(1,2) / Haffi.at<double>(2,2);
                                    Haffi = helpH1 * Haffi;
                                    tdiff1 = Haffi * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
                                    tdiff1 /= tdiff1.at<double>(2);
                                    tdiff1 += (Mat_<double>(3,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y), 0);
                                }
                                diffdist2 = static_cast<double>(std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1)));
                                singleErrVec2 = cv::Point2f(static_cast<float>(tdiff1.at<double>(0)), static_cast<float>(tdiff1.at<double>(1)));
                            }
                            else if(diffManualMode == 4)
                            {
                                newmidpos2 = -pdiff[0];
                                Mat tdiff1 = (Mat_<double>(2,1) << static_cast<double>(newmidpos2.x), static_cast<double>(newmidpos2.y));
                                tdiff1 = Hmanual2.colRange(0,2).inv() * tdiff1;//Calculate the displacement in pixels of the original image (not warped)
                                tdiff1 += (Mat_<double>(2,1) << static_cast<double>(pdiff[1].x), static_cast<double>(pdiff[1].y));
                                diffdist2 = static_cast<double>(std::sqrt(tdiff1.at<double>(0) * tdiff1.at<double>(0) + tdiff1.at<double>(1) * tdiff1.at<double>(1)));
                                singleErrVec2 = cv::Point2f(static_cast<float>(tdiff1.at<double>(0)), static_cast<float>(tdiff1.at<double>(1)));
                            }
                            if((abs(diffdist2last - diffdist2) > DBL_EPSILON) || (refiters == 0))
                            {
                                diffdist2last = diffdist2;
                                if(diffManualMode != manualMode)
                                    recalc = true;
                                diffManualMode = manualMode;
                            }
                            if(refiters < UINT_MAX)
                                refiters++;
                            else
                                refiters = 1;
                        }
                        while(recalc);

                        tdiff = (Mat_<double>(2,1) << static_cast<double>(pdiff[0].x), static_cast<double>(pdiff[0].y));
                        cv::hconcat(compDiff, tdiff, compDiff);
                        warpAffine(patch_wdhist2[0], shiftedpatch, compDiff, patch_wdhist2[0].size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching

                        //show blended image
                        Mat newshift_color[2];
                        cv::cvtColor(shiftedpatch, newshift_color[0], cv::COLOR_GRAY2RGB);
                        patch_wdhist2_color2.copyTo(newshift_color[1]);
                        newshift_color[0].reshape(1, newshift_color[0].rows * newshift_color[0].cols).col(1).setTo(Scalar(0));
                        newshift_color[0].reshape(1, newshift_color[0].rows * newshift_color[0].cols).col(0).setTo(Scalar(0));
                        newshift_color[1].reshape(1, newshift_color[1].rows * newshift_color[1].cols).col(0).setTo(Scalar(0));
                        newshift_color[1].reshape(1, newshift_color[1].rows * newshift_color[1].cols).col(2).setTo(Scalar(0));
                        addWeighted(newshift_color[0], alpha, newshift_color[1], beta, 0.0, blended_w);
                        cv::resize(blended_w, composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                        cv::line(composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
                        cv::line(composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
                        str = "shifted blended";
                        putText(composed, str.c_str(), cv::Point2d(2 * patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));

                        //show Diff
                        Mat userDiff_color;
                        absdiff(shiftedpatch, patch_wdhist2[1], patch_equal1);
                        cv::cvtColor(patch_equal1, userDiff_color, cv::COLOR_GRAY2RGB);
                        cv::resize(userDiff_color, composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
                        cv::line(composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(10.0f, 0), newmidpos + Point2f(10.0f, 0), cv::Scalar(0, 0, 255));
                        cv::line(composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), newmidpos - Point2f(0, 10.0f), newmidpos + Point2f(0, 10.0f), cv::Scalar(0, 0, 255));
                        str = "shifted Diff";
                        putText(composed, str.c_str(), cv::Point2d(3 * patchSelectShow + 5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));

                        //get sum of difference near match
                        if(autseait == 0)
                        {
                            errlevels[0] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
                        }
                        else
                        {
                            errlevels[1] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
                        }

                        stringstream ss;
                        ss << "New Diff: " << diffdist2 << " Sum of differences: " << errlevels[0];
                        str = ss.str();
                        composed(Rect(0, maxVerImgSize + textheight + patchsizeShow + patchSelectShow, composed.cols, bottomtextheight)).setTo(0);
                        putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + patchSelectShow + 10), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));

                        if(noIterRef || noIterRefProg)
                        {
                            cv::imshow("GT match",composed);
                        }
                        else
                        {
                            if(autseait == 0)
                            {
                                cv::imshow("GT match",composed);
                                winPosOld = winPos;
                                diffdist2Old = diffdist2;
                                singleErrVec2Old = singleErrVec2;
                            }
                            else //local refinement of the manual selected match position
                            {
                                if(errlevels[1] >= errlevels[0])
                                {
                                    switch(direction)
                                    {
                                    case 0:
                                        addWinPos.x += 0.25f;
                                        if((oldDir[0] > 0) || (horVerCnt[0] >= 2))
                                        {
                                            direction = 2;
                                            if(oldDir[0] > 0)
                                                atMinimum = 0;
                                        }
                                        else
                                        {
                                            direction = 1;
                                            atMinimum++;
                                            horVerCnt[0]++;
                                        }
                                        oldDir[0] = 0;
                                        horVerCnt[1] = 0;
                                        break;
                                    case 1:
                                        addWinPos.x -= 0.25f;
                                        if((oldDir[1] > 0) || (horVerCnt[0] >= 2))
                                        {
                                            direction = 2;
                                            if(oldDir[1] > 0)
                                                atMinimum = 0;
                                        }
                                        else
                                        {
                                            direction = 0;
                                            atMinimum++;
                                            horVerCnt[0]++;
                                        }
                                        oldDir[1] = 0;
                                        horVerCnt[1] = 0;
                                        break;
                                    case 2:
                                        addWinPos.y += 0.25f;
                                        if((oldDir[2] > 0) || (horVerCnt[1] >= 2))
                                        {
                                            direction = 0;
                                            if(oldDir[2] > 0)
                                                atMinimum = 0;
                                        }
                                        else
                                        {
                                            direction = 3;
                                            atMinimum++;
                                            horVerCnt[1]++;
                                        }
                                        oldDir[2] = 0;
                                        horVerCnt[0] = 0;
                                        break;
                                    case 3:
                                        addWinPos.y -= 0.25f;
                                        if((oldDir[3] > 0) || (horVerCnt[1] >= 2))
                                        {
                                            direction = 0;
                                            if(oldDir[3] > 0)
                                                atMinimum = 0;
                                        }
                                        else
                                        {
                                            direction = 2;
                                            atMinimum++;
                                            horVerCnt[1]++;
                                        }
                                        oldDir[3] = 0;
                                        horVerCnt[0] = 0;
                                        break;
                                    }
                                }
                                else
                                {
                                    oldDir[direction]++;
                                    errlevels[0] = errlevels[1];
                                    if((abs(winPos.x - winPosOld.x) > 0.3) || (abs(winPos.y - winPosOld.y) > 0.3))
                                        noIterRef = true;
                                    winPosOld = winPos;
                                    diffdist2Old = diffdist2;
                                    singleErrVec2Old = singleErrVec2;
                                    cv::imshow("GT match",composed);
                                    c = cv::waitKey(100);
                                    if(c != -1)
                                        skey = getSpecialKeyCode(c);
                                    else
                                        skey = NONE;
                                    if(skey == ESCAPE)
                                    {
                                        winPos = winPos2;
                                        noIterRef = true;
                                        skey = NONE;
                                        c = -1;
                                    }

                                }
                            }
                            switch(direction)
                            {
                            case 0:
                                addWinPos.x -= 0.25f;
                                break;
                            case 1:
                                addWinPos.x += 0.25f;
                                break;
                            case 2:
                                addWinPos.y -= 0.25f;
                                break;
                            case 3:
                                addWinPos.y += 0.25f;
                                break;
                            }
                            if(!noIterRef && ((abs(winPos.x - winPosOld.x) <= 0.3) && (abs(winPos.y - winPosOld.y) <= 0.3)))
                            {
                                if(atMinimum < 5)
                                    winPos = winPos2 + addWinPos;
                                else
                                {
                                    diffdist2 = diffdist2Old;
                                    singleErrVec2 = singleErrVec2Old;
                                }
                            }
                            autseait++;
                        }
                    }
                    while((winPos.x >= minmaxXY[0]) && (winPos.x < minmaxXY[1]) && (winPos.y >= minmaxXY[2]) && (winPos.y < minmaxXY[3])
                          && (addWinPos.x < 5.f) && (addWinPos.y < 5.f) && !noIterRef && (atMinimum < 5) && !noIterRefProg);
                    if((winPos.x < minmaxXY[0]) || (winPos.x >= minmaxXY[1]) || (winPos.y < minmaxXY[2]) || (winPos.y >= minmaxXY[3])
                          || (addWinPos.x >= 5.f) || (addWinPos.y >= 5.f) || noIterRef)
                    {
                        noIterRef = true;
                    }
                    if(atMinimum > 4)
                    {
                        noIterRef = true;
                        winPos2 = winPosOld;//winPos;
                    }
                }
                else if((-winPos.x >= shownLeftImgBorder[0]) && (-winPos.x < shownLeftImgBorder[1]) && (-winPos.y >= shownLeftImgBorder[2]) && (-winPos.y < shownLeftImgBorder[3]))
                {
                    pdiff[1] = -winPos;
                    pdiff[1].y -= textheight;
                    pdiff[1].x *= static_cast<float>(imgSize.width) / static_cast<float>(maxHorImgSize);
                    pdiff[1].y *= static_cast<float>(imgSize.height) / static_cast<float>(maxVerImgSize);

                    patches_wdiff[0] = img_wdiff[0](Rect(pdiff[1], Size(patchsizeOrig, patchsizeOrig))).clone();
                    equalizeHist(patches_wdiff[0], patch_wdhist2[0]);
                    cv::cvtColor(patch_wdhist2[0], leftHist_color, cv::COLOR_GRAY2RGB);
                    cv::resize(leftHist_color, composed(Rect(0, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)), cv::Size(patchSelectShow, patchSelectShow), 0, 0, INTER_LANCZOS4);
#if LEFT_TO_RIGHT
                    str = "left equal hist - select pt";
#else
                    str = "right equal hist - select pt";
#endif
                    putText(composed, str.c_str(), cv::Point2d(5, maxVerImgSize + textheight + patchsizeShow + 10), FONT_HERSHEY_SIMPLEX | FONT_ITALIC, 0.4, cv::Scalar(0, 0, 255));

                    composed(Rect(2 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)).setTo(0);
                    composed(Rect(3 * patchSelectShow, maxVerImgSize + textheight + patchsizeShow, patchSelectShow, patchSelectShow)).setTo(0);
                    composed(Rect(0, maxVerImgSize + textheight + patchsizeShow + patchSelectShow, composed.cols, bottomtextheight)).setTo(0);
                    cv::imshow("GT match",composed);

                    pdiff[1] -= lkp;
                    winPos = cv::Point2f(-FLT_MAX, -FLT_MAX);
                    winPos2 = winPos;
                    manualMode = 3;
                }
                else
                {
                    winPos = cv::Point2f(-FLT_MAX, -FLT_MAX);
                }

                if(skey == POS1)
                {
                    noIterRef = false;
                }

                if(skey == ESCAPE)
                {
                    int answere;
#if __linux__
                    QMessageBox msgBox;
                    msgBox.setText("Exit?");
                    msgBox.setInformativeText("Are you sure? Reviewed matches of current image pair will be lost!");
                    msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
                    answere = msgBox.exec();
                    if(answere == QMessageBox::Yes)
                    {
                        cv::destroyWindow("GT match");
                        return false;
                    }
                    else
                    {
                        skey = NONE;
                        c = -1;
                    }
#else
                    answere = MessageBox(NULL, "Are you sure? Reviewed matches of current image pair will be lost!", "Exit?", MB_YESNO | MB_DEFBUTTON1);
                    if(answere == IDYES)
                    {
                        cv::destroyWindow("GT match");
                        return false;
                    }
                    else
                    {
                        skey = NONE;
                        c = -1;
                    }
#endif
                }

                if(c == 'm')//Deactivate automatic minimum search in manual mode forever
                {
                    string mbtext;
                    std::ostringstream strstr;
                    noIterRefProg = !noIterRefProg;
                    strstr << "The automatic minimum search after manual match point selection is now " << (noIterRefProg ? "deactivated.":"activated.");
                    mbtext = strstr.str();
#if __linux__
                    QMessageBox msgBox;
                    msgBox.setText("Activation of automatic minimum search");
                    msgBox.setInformativeText(QString::fromStdString(mbtext));
                    msgBox.exec();
#else
                    MessageBox(NULL, mbtext.c_str(), "Activation of automatic minimum search", MB_OK | MB_ICONINFORMATION);
#endif
                    skey = NONE;
                    c = -1;
                }

                if(c == 'k')//Deactivate automatic homography estimation based on SIFT keypoints forever for speedup
                {
                    string mbtext;
                    std::ostringstream strstr;
                    autoHestiSift = !autoHestiSift;
                    strstr << "The automatic homography estimation based on SIFT keypoints is now " << (autoHestiSift ? "activated.":"deactivated.");
                    mbtext = strstr.str();
#if __linux__
                    QMessageBox msgBox;
                    msgBox.setText("Activation of automatic homography estimation");
                    msgBox.setInformativeText(QString::fromStdString(mbtext));
                    msgBox.exec();
#else
                    MessageBox(NULL, mbtext.c_str(), "Activation of automatic homography estimation", MB_OK | MB_ICONINFORMATION);
#endif
                    skey = NONE;
                    c = -1;
                }

                if(c == '0')
                {
                    manualMode = 0;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                }
                else if(c == '1')
                {
                    manualMode = 1;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                }
                else if(c == '2')
                {
                    manualMode = 2;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                }
                else if(c == '3')
                {
                    manualMode = 3;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                }
                else if(c == '+')
                {
                    manualMode = 4;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                    manualScale *= 1.05;
                    Hmanual2 *= 1.05;
                }
                else if(c == '-')
                {
                    manualMode = 4;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                    manualScale /= 1.05;
                    Hmanual2 /= 1.05;
                }
                else if(c == 'r')
                {
                    manualMode = 4;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                    manualRot += 3.14159265 / 64;
                    Mat Rrot = (Mat_<double>(2,2) << std::cos(manualRot), (-1. * std::sin(manualRot)), std::sin(manualRot), std::cos(manualRot)); //generate the new rotation matrix
                    Hmanual2.colRange(0,2) = manualScale * Rrot;
                }
                else if(c == 'l')
                {
                    manualMode = 4;
                    if(manualHomoDone)
                        manualHomoDone = false;
                    skey = NONE;
                    c = -1;
                    manualRot -= 3.14159265 / 64;
                    Mat Rrot = (Mat_<double>(2,2) << std::cos(manualRot), (-1. * std::sin(manualRot)), std::sin(manualRot), std::cos(manualRot)); //generate the new rotation matrix
                    Hmanual2.colRange(0,2) = manualScale * Rrot;
                }


                if(c == 'a')
                {
                    int waitvalue = 1500;
                    do
                    {
                        c = cv::waitKey(waitvalue);
                        waitvalue = 5000;
                        if(c != -1)
                            skey = getSpecialKeyCode(c);
                        else
                        {
                            skey = NONE;
#if __linux__
                            QMessageBox msgBox;
                            msgBox.setText("Specify patch size");
                            msgBox.setInformativeText("Specify the minimal patch size factor (the optimal patch size is "
                                                      "automatically chosen but is restricted to be larger than the "
                                                      "given value) for finding the optimal matching position:\n 1 - "
                                                      "Original patch size\n 2 - Half the size\n 3 - Size dived by 4\n 4 - "
                                                      "Size dived by 8\n 5 - Size dived by 16\n Please press one of the "
                                                      "above mentioned buttons after clicking 'OK' within the next 5 "
                                                      "seconds or abort by pressing 'ESC'. Otherwise this message is shown again.");
                            msgBox.exec();
#else
                            MessageBox(NULL, "Specify the minimal patch size factor (the optimal patch size is automatically "
                                             "chosen but is restricted to be larger than the given value) for finding the "
                                             "optimal matching position:\n 1 - Original patch size\n 2 - Half the size\n 3 - "
                                             "Size dived by 4\n 4 - Size dived by 8\n 5 - Size dived by 16\n Please press "
                                             "one of the above mentioned buttons after clicking 'OK' within the next 5 "
                                             "seconds or abort by pressing 'ESC'. Otherwise this message is shown again.",
                                             "Specify patch size", MB_OK | MB_ICONINFORMATION);
#endif
                        }
                        if(skey == ESCAPE)
                        {
                            skey = NONE;
                            c = -1;
                            break;
                        }
                    }
                    while((c != '1') && (c != '2') && (c != '3') && (c != '4') && (c != '5'));

                    if((c == '1') || (c == '2') || (c == '3') || (c == '4') || (c == '5'))
                    {
                        autocalc2 = atoi((char*)(&c));
                    }
                    skey = NONE;
                    c = -1;
                }

                if(c == 'h')
                {
#if LEFT_TO_RIGHT
#if __linux__
                    QMessageBox msgBox;
                    msgBox.setText("Help");
                    msgBox.setInformativeText("If a new matching position is selected inside the area of "
                                              "'left equal hist - select pt', an local refinement is automatically "
                                              "started. If you want to cancel the local minimum search after manual "
                                              "selection of the matching position, press 'ESC' to go back to the "
                                              "selected position or select a new position by clicking inside the "
                                              "area of 'left equal hist - select pt'. After aborting the local "
                                              "refinement or after it has finished, it is deactivated for further "
                                              "manual selections. If you want the local refinement to start again "
                                              "from the current position, press 'Pos1'. To deactivate or activate "
                                              "local refinement for the whole program runtime, press 'm'.\n\n "
                                              "The patch 'left equal hist - select pt' can be processed by a "
                                              "homography to better align to the right patch. Thus, the following "
                                              "options are possible:\n '0' - Use the ground truth homography "
                                              "(if available) or the one generated by matches and properties of "
                                              "SIFT keypoints within the left and right patches.\n '1' - Use the "
                                              "homography generated by the found matches only\n '2' - Manually "
                                              "select 4 matching positions in the left and right patch to estimate "
                                              "a homography (local refinement (see above) is started afterwards)"
                                              "\n '3' - Use the original patch\n\n Global refinement using 'left "
                                              "equal hist - select pt' can be started by pressing 'a' followed "
                                              "by '1' to '5'. The number specifies the minimal patch size for "
                                              "optimization (For more details press 'a' and wait for a short "
                                              "amount of time).\n\n The automatic homography estimation based on "
                                              "SIFT keypoints can be activated/deactivated for all remaining image "
                                              "pairs using 'k'.\n\n To scale the original left patch and display "
                                              "the result within 'left equal hist - select pt', press '+' or '-'. "
                                              "To rotate the original left patch, use the keys 'r' and 'l'.");
                    msgBox.exec();
#else
                    MessageBox(NULL, "If a new matching position is selected inside the area of 'left equal "
                                     "hist - select pt', an local refinement is automatically started. If you want "
                                     "to cancel the local minimum search after manual selection of the matching "
                                     "position, press 'ESC' to go back to the selected position or select a new "
                                     "position by clicking inside the area of 'left equal hist - select pt'. "
                                     "After aborting the local refinement or after it has finished, it is "
                                     "deactivated for further manual selections. If you want the local refinement "
                                     "to start again from the current position, press 'Pos1'. To deactivate or "
                                     "activate local refinement for the whole program runtime, press 'm'.\n\n "
                                     "The patch 'left equal hist - select pt' can be processed by a homography "
                                     "to better align to the right patch. Thus, the following options are possible:"
                                     "\n '0' - Use the ground truth homography (if available) or the one generated "
                                     "by matches and properties of SIFT keypoints within the left and right patches."
                                     "\n '1' - Use the homography generated by the found matches only\n '2' - "
                                     "Manually select 4 matching positions in the left and right patch to estimate a "
                                     "homography (local refinement (see above) is started afterwards)\n '3' - "
                                     "Use the original patch\n\n Global refinement using 'left equal hist - select "
                                     "pt' can be started by pressing 'a' followed by '1' to '5'. The number "
                                     "specifies the minimal patch size for optimization (For more details press 'a' "
                                     "and wait for a short amount of time).\n\n The automatic homography estimation "
                                     "based on SIFT keypoints can be activated/deactivated for all remaining image "
                                     "pairs using 'k'.\n\n To scale the original left patch and display the result "
                                     "within 'left equal hist - select pt', press '+' or '-'. To rotate the original "
                                     "left patch, use the keys 'r' and 'l'.", "Help", MB_OK | MB_ICONINFORMATION);
#endif
#else
#if __linux__
                    QMessageBox msgBox;
                    msgBox.setText("Help");
                    msgBox.setInformativeText("If a new matching position is selected inside the area of 'right equal hist - select pt', "
                                              "an local refinement is automatically started. If you want to cancel the local minimum search after "
                                              "manual selection of the matching position, press 'ESC' to go back to the selected position or "
                                              "select a new position by clicking inside the area of 'right equal hist - select pt'. After "
                                              "aborting the local refinement or after it has finished, it is deactivated for further manual "
                                              "selections. If you want the local refinement to start again from the current position, press 'Pos1'. "
                                              "To deactivate or activate local refinement for the whole program runtime, press 'm'.\n\n The patch "
                                              "'right equal hist - select pt' can be processed by a homography to better align to the right patch. "
                                              "Thus, the following options are possible:\n '0' - Use the ground truth homography (if available) or "
                                              "the one generated by matches and properties of SIFT keypoints within the left and right patches."
                                              "\n '1' - Use the homography generated by the found matches only\n '2' - Manually select "
                                              "4 matching positions in the left and right patch to estimate a homography (local refinement "
                                              "(see above) is started afterwards)\n '3' - Use the original patch\n\n Global refinement using "
                                              "'right equal hist - select pt' can be started by pressing 'a' followed by '1' to '5'. "
                                              "The number specifies the minimal patch size for optimization (For more details press 'a' "
                                              "and wait for a short amount of time).\n\n The automatic homography estimation based on "
                                              "SIFT keypoints can be activated/deactivated for all remaining image pairs using 'k'.\n\n "
                                              "To scale the original left patch and display the result within 'right equal hist - select pt', "
                                              "press '+' or '-'. To rotate the original left patch, use the keys 'r' and 'l'.");
                    msgBox.exec();
#else
                    MessageBox(NULL, "If a new matching position is selected inside the area of 'right equal hist - select pt', "
                         "an local refinement is automatically started. If you want to cancel the local minimum search after "
                         "manual selection of the matching position, press 'ESC' to go back to the selected position or "
                         "select a new position by clicking inside the area of 'right equal hist - select pt'. After "
                         "aborting the local refinement or after it has finished, it is deactivated for further manual "
                         "selections. If you want the local refinement to start again from the current position, press 'Pos1'. "
                         "To deactivate or activate local refinement for the whole program runtime, press 'm'.\n\n The patch "
                         "'right equal hist - select pt' can be processed by a homography to better align to the right patch. "
                         "Thus, the following options are possible:\n '0' - Use the ground truth homography (if available) or "
                         "the one generated by matches and properties of SIFT keypoints within the left and right patches."
                         "\n '1' - Use the homography generated by the found matches only\n '2' - Manually select "
                         "4 matching positions in the left and right patch to estimate a homography (local refinement "
                         "(see above) is started afterwards)\n '3' - Use the original patch\n\n Global refinement using "
                         "'right equal hist - select pt' can be started by pressing 'a' followed by '1' to '5'. "
                         "The number specifies the minimal patch size for optimization (For more details press 'a' "
                         "and wait for a short amount of time).\n\n The automatic homography estimation based on "
                         "SIFT keypoints can be activated/deactivated for all remaining image pairs using 'k'.\n\n "
                         "To scale the original left patch and display the result within 'right equal hist - select pt', "
                         "press '+' or '-'. To rotate the original left patch, use the keys 'r' and 'l'.", "Help", MB_OK | MB_ICONINFORMATION);
#endif
#endif
                    c = -1;
                    skey = NONE;
                }
            }
            while(((c == -1) || (skey == ARROW_UP) || (skey == ARROW_DOWN) || (skey == ARROW_LEFT) || (skey == ARROW_RIGHT) || (skey == POS1) || (c == 'm') || (c == 'h')) ||
                  ((c != 's') && (c != 'i') && (c != 'e') && (c != 'n') && (skey != SPACE)));

            if(c == 's')
            {
                autoManualAnno.pop_back();
                skipped.push_back(i + static_cast<int>(skipped.size()));
                i--;
            }
            else if(c == 'i')
            {
                distanceHisto[0].second++;
                distances.push_back(0);
                errvecs.emplace_back(cv::Point2f(0,0));
                matchesGT_idx.push_back(idx);
#if LEFT_TO_RIGHT
                perfectMatches.emplace_back(std::make_pair(lkp, rkp));
#else
                perfectMatches.emplace_back(std::make_pair(rkp, lkp));
#endif
                calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
                distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
            }
            else if(c == 'e')
            {
                int k = 1;
                distances.push_back(diffdist);
                while((k < static_cast<int>(distanceHisto.size())) && (distanceHisto[k].first < diffdist))
                    k++;
                distanceHisto[k - 1].second++;
                errvecs.push_back(singleErrVec);
                matchesGT_idx.push_back(idx);
#if LEFT_TO_RIGHT
                perfectMatches.emplace_back(std::make_pair(lkp + singleErrVec, rkp));
#else
                perfectMatches.emplace_back(std::make_pair(rkp, lkp + singleErrVec));
#endif
                calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
                distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
                if(diffdist >= usedMatchTH)
                {
#if LEFT_TO_RIGHT
                    falseGT.emplace_back(std::make_pair(lkp, rkp));
#else
                    falseGT.emplace_back(std::make_pair(rkp, lkp));
#endif
                    wrongGTidxDist.emplace_back(std::make_pair(idx, diffdist));
                    if(flowGtIsUsed)
                    {
#if LEFT_TO_RIGHT
                        validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(lkp.y + 0.5f)), static_cast<int>(floor(lkp.x + 0.5f))));
#else
                        validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(rkp.y + 0.5f)), static_cast<int>(floor(rkp.x + 0.5f))));
#endif
                    }
                    else
                    {
                        validityValFalseGT.push_back(1);
                    }
                }
            }
            else if(c == 'n')
            {
#if LEFT_TO_RIGHT
                falseGT.emplace_back(std::make_pair(lkp, rkp));
#else
                falseGT.emplace_back(std::make_pair(rkp, lkp));
#endif
                notMatchable++;
                autoManualAnno.pop_back();
                wrongGTidxDist.emplace_back(std::make_pair(idx, -1.0));
                if(flowGtIsUsed)
                {
#if LEFT_TO_RIGHT
                    validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(lkp.y + 0.5f)), static_cast<int>(floor(lkp.x + 0.5f))));
#else
                    validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(rkp.y + 0.5f)), static_cast<int>(floor(rkp.x + 0.5f))));
#endif
                }
                else
                {
                    validityValFalseGT.push_back(1);
                }
            }
            else if((winPos2.x >= 0) && (winPos2.y >= 0))
            {
                int k = 1;
                distances.push_back(diffdist2);
                while((k < static_cast<int>(distanceHisto.size())) && (distanceHisto[k].first < diffdist2))
                    k++;
                distanceHisto[k - 1].second++;
                errvecs.push_back(singleErrVec2);
                matchesGT_idx.push_back(idx);
#if LEFT_TO_RIGHT
                perfectMatches.emplace_back(std::make_pair(lkp + singleErrVec2, rkp));
#else
                perfectMatches.emplace_back(std::make_pair(rkp, lkp + singleErrVec2));
#endif
                calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
                distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
                if(diffdist2 >= usedMatchTH)
                {
#if LEFT_TO_RIGHT
                    falseGT.emplace_back(std::make_pair(lkp, rkp));
#else
                    falseGT.emplace_back(std::make_pair(rkp, lkp));
#endif
                    wrongGTidxDist.emplace_back(std::make_pair(idx, diffdist2));
                    if(flowGtIsUsed)
                    {
#if LEFT_TO_RIGHT
                        validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(lkp.y + 0.5f)), static_cast<int>(floor(lkp.x + 0.5f))));
#else
                        validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(rkp.y + 0.5f)), static_cast<int>(floor(rkp.x + 0.5f))));
#endif
                    }
                    else
                    {
                        validityValFalseGT.push_back(1);
                    }
                }
            }
            else
            {
                int k = 1;
                distances.push_back(diffdist);
                while((k < static_cast<int>(distanceHisto.size())) && (distanceHisto[k].first < diffdist))
                    k++;
                distanceHisto[k - 1].second++;
                errvecs.push_back(singleErrVec);
                matchesGT_idx.push_back(idx);
#if LEFT_TO_RIGHT
                perfectMatches.emplace_back(std::make_pair(lkp + singleErrVec, rkp));
#else
                perfectMatches.emplace_back(std::make_pair(rkp, lkp + singleErrVec));
#endif
                calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed, errvecsGT, validityValGT, homoGT, lkp, rkp);
                distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
                if(diffdist >= usedMatchTH)
                {
#if LEFT_TO_RIGHT
                    falseGT.emplace_back(std::make_pair(lkp, rkp));
#else
                    falseGT.emplace_back(std::make_pair(rkp, lkp));
#endif
                    wrongGTidxDist.emplace_back(std::make_pair(idx, diffdist));
                    if(flowGtIsUsed)
                    {
#if LEFT_TO_RIGHT
                        validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(lkp.y + 0.5f)), static_cast<int>(floor(lkp.x + 0.5f))));
#else
                        validityValFalseGT.push_back(channelsFlow[2].at<float>(static_cast<int>(floor(rkp.y + 0.5f)), static_cast<int>(floor(rkp.x + 0.5f))));
#endif
                    }
                    else
                    {
                        validityValFalseGT.push_back(1);
                    }
                }
            }
            cv::destroyWindow("GT match");

            //Reestimate number of samples
            if(!falseGT.empty() && fullN && fullSamples && fullFails && (maxSampleSize < GTsi))
            {
                double newFullSamples = (double)(*fullSamples + i + 1);
                newP = (double)(*fullFails + static_cast<int>(falseGT.size())) / newFullSamples;
                if((abs(oldP) <= 10 * DBL_MIN) || (abs(newP - oldP) / oldP > 0.1))
                {
                    double e;
                    double minSampleSize, sampleRatio;
                    int newSamples;
                    getMinSampleSize(*fullN, newP, e, minSampleSize);

                    if(newFullSamples / minSampleSize > 0.85)
                    {
                        sampleRatio = minSampleSize / (double)(*fullN);
                    }
                    else
                    {
                        double sasi[2];
                        getMinSampleSize(*fullN, 2.0 * newP, e, sasi[0]);
                        getMinSampleSize(*fullN, 0.5 * newP, e, sasi[1]);
                        minSampleSize = max(sasi[0], max(minSampleSize, sasi[1]));
                        sampleRatio = minSampleSize / (double)(*fullN);
                    }

                    newSamples = static_cast<int>(ceil(static_cast<double>(GTsi) * sampleRatio));
                    newSamples = newSamples < 10 ? 10:newSamples;
                    newSamples = newSamples > GTsi ? GTsi:newSamples;
                    if(abs(maxSampleSize - newSamples) > 10)
                    {
                        if(newSamples < (i + 1))
                        {
                            samples = usedSamples = maxSampleSize = i + 1;
                        }
                        else
                        {
                            samples = usedSamples = maxSampleSize = newSamples;
                        }
                    }
                }
            }
#else
            stringstream ss;
            distances.push_back(0);
            autoManualAnno.push_back('U');
            errvecs.emplace_back(0, 0);
            matchesGT_idx.push_back(idx);
            perfectMatches.emplace_back(rkp, lkp);
            std::vector<int> validityValGT_tmp;
            calcErrorToSpatialGT(perfectMatches.back().first, perfectMatches.back().second, channelsFlow, flowGtIsUsed,
                                 errvecsGT, validityValGT_tmp, homoGT, lkp, rkp);
            distancesGT.push_back(std::sqrt(errvecsGT.back().x * errvecsGT.back().x + errvecsGT.back().y * errvecsGT.back().y));
            validityValGT.push_back(-1);
#endif
        }
	}

	//Generate homography or essential matrix from selected correspondences
	vector<Point2f> leftPs, rightPs;
	for(auto & perfectMatche : perfectMatches)
	{
		leftPs.push_back(perfectMatche.first);
		rightPs.push_back(perfectMatche.second);
	}
	if(flowGtIsUsed)
	{		
		HE = cv::findFundamentalMat(leftPs, rightPs, FM_RANSAC, 0.75);
		if(!HE.empty())
		{
			HE.convertTo(HE, CV_64FC1);
			Mat Et = HE.t();
			for (int i = 0; i < static_cast<int>(perfectMatches.size()); i++)
			{
				Mat x1 = (Mat_<double>(3, 1) << leftPs[i].x, leftPs[i].y, 1.0); 
				Mat x2 = (Mat_<double>(3, 1) << rightPs[i].x, rightPs[i].y, 1.0); 
				double x2tEx1 = x2.dot(HE * x1); 
				Mat Ex1 = HE * x1; 
				Mat Etx2 = Et * x2; 
				double a = Ex1.at<double>(0) * Ex1.at<double>(0); 
				double b = Ex1.at<double>(1) * Ex1.at<double>(1); 
				double c = Etx2.at<double>(0) * Etx2.at<double>(0); 
				double d = Etx2.at<double>(1) * Etx2.at<double>(1); 

				distancesEstModel.push_back(x2tEx1 * x2tEx1 / (a + b + c + d));
			}
		}
	}
	else
	{
		HE = cv::findHomography(leftPs, rightPs, LMEDS);
		if(!HE.empty())
		{
			HE.convertTo(HE, CV_64FC1);
			for (int i = 0; i < static_cast<int>(perfectMatches.size()); i++)
			{
				Mat x1 = (Mat_<double>(3, 1) << leftPs[i].x, leftPs[i].y, 1.0); 
				Mat x2 = (Mat_<double>(3, 1) << rightPs[i].x, rightPs[i].y, 1.0);
				Mat Hx1 = HE * x1;
				Hx1 /= Hx1.at<double>(2);
				Hx1 = x2 - Hx1;
				distancesEstModel.push_back(std::sqrt(Hx1.at<double>(0) * Hx1.at<double>(0) + Hx1.at<double>(1) * Hx1.at<double>(1)));
			}
		}
	}

	return true;
}

/* This function calculates the subpixel-difference in the position of two patches.
   *
   * Mat patch	                Input  -> First patch that will be dynamic in its position
   * Mat image                  Input  -> Second patch (or image) that will be static in its position
   * Point2f diff				Output -> The position diffenrence between the patches
   *									  (Position (upper left corner of patch1) in patch2 for which patch1 fits perfectly)
   *
   * Return value:              none
   */
void getSubPixPatchDiff(const cv::Mat& patch, const cv::Mat& image, cv::Point2f &diff)
{
	cv::Mat results;
	float featuresize_2, nx, ny, valPxy, valPxp, valPxn, valPyp, valPyn;
	cv::Point minLoc;

    	
     
	cv::matchTemplate(image, patch, results, TM_SQDIFF);
	cv::minMaxLoc(results,(double *)nullptr,(double *)nullptr,&minLoc);

	diff = cv::Point2f(static_cast<float>(minLoc.x), static_cast<float>(minLoc.y));
	if((minLoc.x >= results.cols - 1) || (minLoc.x <= 0) || (minLoc.y >= results.rows - 1) || (minLoc.y <= 0))
		return;

	valPxy = results.at<float>(minLoc.y,minLoc.x);
	valPxp = results.at<float>(minLoc.y,minLoc.x+1);
	valPxn = results.at<float>(minLoc.y,minLoc.x-1);
	valPyp = results.at<float>(minLoc.y+1,minLoc.x);
	valPyn = results.at<float>(minLoc.y-1,minLoc.x);

	nx = 2*(2*valPxy-valPxn-valPxp);
	ny = 2*(2*valPxy-valPyn-valPyp);

	if((nx != 0) && (ny != 0))
	{
		nx = (valPxp-valPxn)/nx;
		ny = (valPyp-valPyn)/ny;
		diff += cv::Point2f(nx, ny);
	}
}

void iterativeTemplMatch(cv::InputArray patch, cv::InputArray img, int maxPatchSize, cv::Point2f & minDiff, int maxiters)
{
	Mat _patch, _img;
	_patch = patch.getMat();
	_img = img.getMat();
	CV_Assert((_patch.rows >= maxPatchSize) && (_patch.cols == _patch.rows));
	CV_Assert((_img.rows == _img.cols) && (_img.rows > maxPatchSize));
	int paImDi = _img.rows - maxPatchSize;
	CV_Assert(!(paImDi % 2));
	vector<int> sizeDivs;

	paImDi /= 2;
	sizeDivs.push_back(maxPatchSize);
	while(!(sizeDivs.back() % 2))
	{
		sizeDivs.push_back(sizeDivs.back() / 2);
	}
	int actualborder = paImDi;
	int actPatchborder = 0;
	float minDist = FLT_MAX;
	for(int i = 0; i < static_cast<int>(sizeDivs.size()) - 1; i++)
	{
		float dist_tmp;
		cv::Point2f minDiff_tmp;
		getSubPixPatchDiff(_patch(Rect(actPatchborder, actPatchborder, sizeDivs[i], sizeDivs[i])), _img, minDiff_tmp);//Template matching with subpixel accuracy
		minDiff_tmp -= Point2f(static_cast<float>(actualborder), static_cast<float>(actualborder));//Compensate for the large patch size
		dist_tmp = minDiff_tmp.x * minDiff_tmp.x + minDiff_tmp.y * minDiff_tmp.y;
		if(dist_tmp < minDist)
		{
			minDist = dist_tmp;
			minDiff = minDiff_tmp;
		}
		if((sizeDivs.size() > 1) && (i < static_cast<int>(sizeDivs.size()) - 2))
		{
			actPatchborder += sizeDivs[i+2];
			actualborder += sizeDivs[i+2];
		}
		maxiters--;
		if(maxiters <= 0)
			break;
	}
}

/* Estimates the minimum sample size for a given population
 *
 * int N						Input  -> Size of the whole population (dataset)
 * double p						Input  -> Expected frequency of occurance
 * double e						Output -> error range
 * double minSampleSize			Output -> Minimum sample size that should be used
 * 
 *
 * Return value:				none
 */
void getMinSampleSize(int N, double p, double & e, double & minSampleSize)
{
	double q;
	//double e; //error range
	const double z = 1.96; //a z-value of 1.96 corresponds to a confidence level of 95%
	p = p >= 1.0 ? 1.0:p;
	q = 1.0 - p;
	if(p < 0.02)
		e = p / 2.0;
	else
		e = 0.01;
	minSampleSize = z * z * p * q / (e * e);
	minSampleSize = floor(minSampleSize / (1.0 + minSampleSize / static_cast<double>(N)) + 0.5);
	minSampleSize = minSampleSize > 15000 ? 15000:minSampleSize;
}

/* Bilinear interpolation function for interpolating the value of a coordinate between 4 pixel locations.
 * The function was copied from https://helloacm.com/cc-function-to-compute-the-bilinear-interpolation/
 *
 * float q11					Input  -> Value (e.g. intensity) at first coordinate (e.g. bottom left)
 * float q12					Input  -> Value (e.g. intensity) at second coordinate (e.g. top left)
 * float q22					Input  -> Value (e.g. intensity) at third coordinate (e.g. top right)
 * float q21					Input  -> Value (e.g. intensity) at fourth coordinate (e.g. bottom right)
 * float x1						Input  -> x-coordinate of q11 and q12
 * float x2						Input  -> x-coordinate of q21 and q22
 * float y1						Input  -> y-coordinate of q11 and q21
 * float y1						Input  -> y-coordinate of q12 and q22
 * float x						Input  -> x-coordinate of the position for which the interpolation is needed
 * float y						Input  -> y-coordinate of the position for which the interpolation is needed
 * 
 *
 * Return value:				The interpolated value
 */
float BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y)
{
    float x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.f / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}


void calcErrorToSpatialGT(const cv::Point2f& perfectMatchesFirst, const cv::Point2f& perfectMatchesSecond,
						  std::vector<cv::Mat> channelsFlow, bool flowGtIsUsed, std::vector<cv::Point2f> & errvecsGT, 
						  std::vector<int> & validityValGT, const cv::Mat& homoGT, const cv::Point2f& lkp, const cv::Point2f& rkp)
{
#if LEFT_TO_RIGHT
	if(flowGtIsUsed)
	{
		Point2f newFlow;
		Point2f surroundingPts[4];
		bool validSurrounding[4];
		surroundingPts[0] = Point2f(ceil(perfectMatchesFirst.x - 1.f), ceil(perfectMatchesFirst.y - 1.f));
		surroundingPts[1] = Point2f(ceil(perfectMatchesFirst.x - 1.f), floor(perfectMatchesFirst.y + 1.f));
		surroundingPts[2] = Point2f(floor(perfectMatchesFirst.x + 1.f), floor(perfectMatchesFirst.y + 1.f));
		surroundingPts[3] = Point2f(floor(perfectMatchesFirst.x + 1.f), ceil(perfectMatchesFirst.y - 1.f));
		validSurrounding[0] = channelsFlow[2].at<float>(surroundingPts[0].y, surroundingPts[0].x) > 0;
		validSurrounding[1] = channelsFlow[2].at<float>(surroundingPts[1].y, surroundingPts[1].x) > 0;
		validSurrounding[2] = channelsFlow[2].at<float>(surroundingPts[2].y, surroundingPts[2].x) > 0;
		validSurrounding[3] = channelsFlow[2].at<float>(surroundingPts[3].y, surroundingPts[3].x) > 0;
		if(validSurrounding[0] && validSurrounding[1] && validSurrounding[2] && validSurrounding[3])
		{
			newFlow.x = BilinearInterpolation(channelsFlow[0].at<float>(surroundingPts[0].y, surroundingPts[0].x), channelsFlow[0].at<float>(surroundingPts[1].y, surroundingPts[1].x),
												channelsFlow[0].at<float>(surroundingPts[3].y, surroundingPts[3].x), channelsFlow[0].at<float>(surroundingPts[2].y, surroundingPts[2].x),
												surroundingPts[0].x, surroundingPts[2].x, surroundingPts[0].y, surroundingPts[2].y,
												perfectMatchesFirst.x, perfectMatchesFirst.y);
			newFlow.y = BilinearInterpolation(channelsFlow[1].at<float>(surroundingPts[0].y, surroundingPts[0].x), channelsFlow[1].at<float>(surroundingPts[1].y, surroundingPts[1].x),
												channelsFlow[1].at<float>(surroundingPts[3].y, surroundingPts[3].x), channelsFlow[1].at<float>(surroundingPts[2].y, surroundingPts[2].x),
												surroundingPts[0].x, surroundingPts[2].x, surroundingPts[0].y, surroundingPts[2].y,
												perfectMatchesFirst.x, perfectMatchesFirst.y);
			newFlow += perfectMatchesFirst;
		}
		else
		{
			Point2i intP = Point2i(static_cast<int>(floor(perfectMatchesFirst.x + 0.5f)), static_cast<int>(floor(perfectMatchesFirst.y + 0.5f)));
			if(channelsFlow[2].at<float>(intP.y, intP.x) > 0)
			{
				newFlow = Point2f(channelsFlow[0].at<float>(intP.y, intP.x), channelsFlow[1].at<float>(intP.y, intP.x));
				newFlow += perfectMatchesFirst;
			}
			else
			{
				int i1;
				for(i1 = 0; i1 < 4; i1++)
				{
					if(channelsFlow[2].at<float>(surroundingPts[i1].y, surroundingPts[i1].x) > 0)
						break;
				}
				if(i1 < 4)
				{
					newFlow = Point2f(channelsFlow[0].at<float>(surroundingPts[i1].y, surroundingPts[i1].x), channelsFlow[1].at<float>(surroundingPts[i1].y, surroundingPts[i1].x));
					newFlow += perfectMatchesFirst;
				}
				else
					newFlow = Point2f(FLT_MAX, FLT_MAX);
			}
		}
		errvecsGT.push_back(perfectMatchesSecond - newFlow);
		validityValGT.push_back(channelsFlow[2].at<float>(lkp.y, lkp.x));
	}
	else
	{
		Mat lpt_tmp;
		lpt_tmp = (Mat_<double>(3,1) << static_cast<double>(perfectMatchesSecond.x), static_cast<double>(perfectMatchesSecond.y), 1.0);
		lpt_tmp = homoGT.inv() * lpt_tmp;
		lpt_tmp /= lpt_tmp.at<double>(2);
		errvecsGT.push_back(perfectMatchesFirst - Point2f(lpt_tmp.at<double>(0),lpt_tmp.at<double>(1)));
		validityValGT.push_back(1);
	}
#else
	if(flowGtIsUsed)
	{
		Point2i intP = Point2i(static_cast<int>(floor(perfectMatchesFirst.x + 0.5f)), static_cast<int>(floor(perfectMatchesFirst.y + 0.5f)));
		Point2f newFlow;
		newFlow = Point2f(channelsFlow[0].at<float>(intP.y, intP.x), channelsFlow[1].at<float>(intP.y, intP.x));
		newFlow += perfectMatchesFirst;
		errvecsGT.push_back(perfectMatchesSecond - newFlow);
		validityValGT.push_back(channelsFlow[2].at<float>(rkp.y, rkp.x));
	}
	else
	{
		Mat lpt_tmp;
		lpt_tmp = (Mat_<double>(3,1) << static_cast<double>(rkp.x), static_cast<double>(rkp.y), 1.0);
		lpt_tmp = homoGT * lpt_tmp;
		lpt_tmp /= lpt_tmp.at<double>(2);
		errvecsGT.push_back(perfectMatchesSecond - Point2f(lpt_tmp.at<double>(0),lpt_tmp.at<double>(1)));
		validityValGT.push_back(1);
	}
#endif
}

void findLocalMin(const Mat& patchL, const Mat& patchR, float quarterpatch, float eigthpatch, cv::Point2f &winPos, float patchsizeOrig)
{
	double errlevels[3] = {0, 0, 0};
	int autseait = 0;
	int direction = 0;//0=left, 1=right, 2=up, 3=down
	cv::Point2f addWinPos = cv::Point2f(0,0);
	cv::Point2f winPosOld, winPos2;
	int oldDir[4] = {0,0,0,0};
	int horVerCnt[2] = {0,0};
	int atMinimum = 0;
	Mat patch_equal1, shiftedpatch;
	float halfpatchsize = patchsizeOrig / 2.f;

	winPos2 = winPos;
	do
	{
		Mat compDiff, tdiff;
		compDiff = Mat::eye(2,2,CV_64F);
		tdiff = (Mat_<double>(2,1) << static_cast<double>(addWinPos.x), static_cast<double>(addWinPos.y));
		cv::hconcat(compDiff, tdiff, compDiff);
		warpAffine(patchL, shiftedpatch, compDiff, patchL.size(), INTER_LANCZOS4); //Shift the left patch according to the value from template matching
		absdiff(shiftedpatch, patchR, patch_equal1);
		//get sum of difference near match
		if(autseait == 0)
		{
			errlevels[0] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
			errlevels[2] = errlevels[0];
		}
		else
		{
			errlevels[1] = cv::sum(patch_equal1(Rect(quarterpatch + eigthpatch, quarterpatch + eigthpatch, quarterpatch, quarterpatch)))[0];
		}

		if(autseait == 0)
		{
			winPosOld = winPos;
		}
		else //local refinement of the manual selected match position
		{
			if(errlevels[1] >= errlevels[0])
			{
				switch(direction)
				{
				case 0:
					addWinPos.x += 0.0625f;
					if((oldDir[0] > 0) || (horVerCnt[0] >= 2))
					{
						direction = 2;
						if(oldDir[0] > 0)
							atMinimum = 0;
					}
					else
					{
						direction = 1;
						atMinimum++;
						horVerCnt[0]++;
					}
					oldDir[0] = 0;
					horVerCnt[1] = 0;
					break;
				case 1:
					addWinPos.x -= 0.0625f;
					if((oldDir[1] > 0) || (horVerCnt[0] >= 2))
					{
						direction = 2;
						if(oldDir[1] > 0)
							atMinimum = 0;
					}
					else
					{
						direction = 0;
						atMinimum++;
						horVerCnt[0]++;
					}
					oldDir[1] = 0;
					horVerCnt[1] = 0;
					break;
				case 2:
					addWinPos.y += 0.0625f;
					if((oldDir[2] > 0) || (horVerCnt[1] >= 2))
					{
						direction = 0;
						if(oldDir[2] > 0)
							atMinimum = 0;
					}
					else
					{
						direction = 3;
						atMinimum++;
						horVerCnt[1]++;
					}
					oldDir[2] = 0;
					horVerCnt[0] = 0;
					break;
				case 3:
					addWinPos.y -= 0.0625f;
					if((oldDir[3] > 0) || (horVerCnt[1] >= 2))
					{
						direction = 0;
						if(oldDir[3] > 0)
							atMinimum = 0;
					}
					else
					{
						direction = 2;
						atMinimum++;
						horVerCnt[1]++;
					}
					oldDir[3] = 0;
					horVerCnt[0] = 0;
					break;
				default:
				    break;
				}
			}
			else
			{
				oldDir[direction]++;
				errlevels[0] = errlevels[1];
				winPosOld = winPos;
			}
		}
		switch(direction)
		{
		case 0:
			addWinPos.x -= 0.0625f;
			break;
		case 1:
			addWinPos.x += 0.0625f;
			break;
		case 2:
			addWinPos.y -= 0.0625f;
			break;
		case 3:
			addWinPos.y += 0.0625f;
			break;
        default:
            break;
		}
		if((abs(winPos.x - winPosOld.x) <= 0.075f) && (abs(winPos.y - winPosOld.y) <= 0.075f) && (atMinimum < 5))
		{
			winPos = winPos2 + addWinPos;
		}
		autseait++;
	}
	while((winPos.x > -halfpatchsize) && (winPos.x < halfpatchsize) && (winPos.y > -halfpatchsize) && (winPos.y < halfpatchsize)
		  && (addWinPos.x < 2.f) && (addWinPos.y < 2.f) && (atMinimum < 5));

	winPos = winPosOld;

	if((errlevels[2] < errlevels[0]) || (addWinPos.x >= 2.f) || (addWinPos.y >= 2.f) || (winPos.x <= -halfpatchsize) || (winPos.x >= halfpatchsize) || (winPos.y <= -halfpatchsize) || (winPos.y >= halfpatchsize))
		winPos = winPos2;
}

//Prepare GTM from Oxford dataset
bool baseMatcher::calcGTM_KITTI(size_t &min_nrTP){
    string path = concatPath(imgsPath, "KITTI");
    if(!checkPathExists(path)){
        cerr << "No folder named KITTI found in " << imgsPath << endl;
        cerr << "Skipping GTM for KITTI" << endl;
        return false;
    }
    gtmdata.sum_TP_KITTI = 0;
    gtmdata.sum_TN_KITTI = 0;
    flowGtIsUsed = true;
    for(auto &i: GetKITTISubDirs()){
        string img1_path = concatPath(path, i.img1.sub_folder);
        if(!checkPathExists(img1_path)){
            cerr << "No folder " << img1_path << " found" << endl;
            continue;
        }
        string img2_path = concatPath(path, i.img2.sub_folder);
        if(!checkPathExists(img2_path)){
            cerr << "No folder " << img2_path << " found" << endl;
            continue;
        }
        string gt_path = concatPath(path, i.gt12.sub_folder);
        if(!checkPathExists(gt_path)){
            cerr << "No folder " << gt_path << " found" << endl;
            continue;
        }
        std::vector<std::tuple<std::string, std::string, std::string>> fileNames;
        if(!loadKittiImageGtFnames(path, i, fileNames)){
            continue;
        }
        const string gtm_path = concatPath(gt_path, gtm_sub_folder);
        size_t nrGts = fileNames.size();
        std::vector<std::pair<std::string, std::string>> imgNames_tmp;
        if(checkPathExists(gtm_path)){
            for (size_t j = 0; j < nrGts; ++j){
                std::pair<std::string, std::string> imageNames = make_pair(get<0>(fileNames[j]), get<1>(fileNames[j]));
                if(!loadGTM(gtm_path, imageNames)){
                    if(!calcRefineStoreGTM_KITTI(fileNames[j], i.isFlow, gtm_path, i.gt12.sub_folder,
                                                 static_cast<int>(nrGts - j), true)){
                        continue;
                    }
                }
                addGTMdataToPool();
                gtmdata.sum_TP_KITTI += positivesGT;
                gtmdata.sum_TN_KITTI += negativesGT;
                imgNames_tmp.emplace_back(move(imageNames));
            }
            if(imgNames_tmp.empty()){
                cerr << "Unable to load/calculate GTM for KITTI subset " << i.gt12.sub_folder << endl;
                return false;
            }
            gtmdata.imgNamesAll.emplace_back(std::move(imgNames_tmp));
            gtmdata.sourceGT.emplace_back('K');
        }else{
            bool save_it = true;
            if(!createDirectory(gtm_path)){
                cerr << "Unable to create GTM directory" << endl;
                save_it = false;
            }
            for (size_t j = 0; j < nrGts; ++j) {
                std::pair<std::string, std::string> imageNames = make_pair(get<0>(fileNames[j]), get<1>(fileNames[j]));
                if(!calcRefineStoreGTM_KITTI(fileNames[j], i.isFlow, gtm_path, i.gt12.sub_folder,
                                             static_cast<int>(nrGts - j), save_it)){
                    continue;
                }
                addGTMdataToPool();
                gtmdata.sum_TP_KITTI += positivesGT;
                gtmdata.sum_TN_KITTI += negativesGT;
                imgNames_tmp.emplace_back(move(imageNames));
            }
            if(imgNames_tmp.empty()){
                cerr << "Unable to calculate GTM for KITTI subset " << i.gt12.sub_folder << endl;
                return false;
            }
            gtmdata.imgNamesAll.emplace_back(std::move(imgNames_tmp));
            gtmdata.sourceGT.emplace_back('K');
        }
        if(gtmdata.sum_TP_KITTI >= min_nrTP){
            break;
        }
    }
    return true;
}

bool baseMatcher::getKittiGTM(const std::string &img1f, const std::string &img2f, const std::string &gt, bool is_flow){
    imgs[0] = imread(img1f, cv::IMREAD_GRAYSCALE);
    imgs[1] = imread(img2f, cv::IMREAD_GRAYSCALE);
    vector<cv::Point2f> pts1, pts2;
    if(is_flow){
        if(!convertImageFlowFile(gt, pts1, pts2)){
            return false;
        }
    }else{
        if(!convertImageDisparityFile(gt, pts1, pts2)){
            return false;
        }
    }
    //Interpolate missing values
    Mat dense_flow(imgs[0].rows, imgs[0].cols, CV_32FC2);
    Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
    gd->setK(128);
    gd->setSigma(0.05f);
    gd->setLambda(999.f);
    gd->setFGSLambda(500.0f);
    gd->setFGSSigma(1.5f);
    gd->setUsePostProcessing(false);
    gd->interpolate(imgs[0], pts1, imgs[1], pts2, dense_flow);
    CV_Assert(dense_flow.type() == CV_32FC2);
    std::vector<Mat> vecMats;
    cv::split(dense_flow, vecMats);
    cv::Mat validity(vecMats.back().size(), vecMats.back().type(), 2.f);
    for(auto &pos: pts1){
        validity.at<float>(static_cast<int>(pos.y), static_cast<int>(pos.x)) = 1.f;
    }
    vecMats.emplace_back(move(validity));
#if 1
    namedWindow( "Channel 1", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Channel 1", vecMats[0] );
    namedWindow( "Channel 2", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Channel 2", vecMats[1] );
    namedWindow( "Channel 3", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Channel 3", vecMats[2] );
    cv::waitKey(0);
    cv::destroyAllWindows();
#endif
    cv::merge(vecMats, flowGT);
    if(!detectFeatures()){
        return false;
    }
    int err = filterInitFeaturesGT();
    return err == 0;
}

bool baseMatcher::loadKittiImageGtFnames(const std::string &mainPath, kittiFolders &info,
                                         std::vector<std::tuple<std::string, std::string, std::string>> &fileNames){
    string fileprefl = concatPath(info.img1.sub_folder, "*" + info.img1.postfix);
    string fileprefr = concatPath(info.img2.sub_folder, "*" + info.img2.postfix);
    std::vector<std::string> filenamesl, filenamesr, filenamesgt;
    int res = loadStereoSequenceNew(mainPath, fileprefl, fileprefr, filenamesl, filenamesr);
    if(res){
        return false;
    }
    size_t nr_files = filenamesl.size();
    if(nr_files != filenamesr.size()){
        cerr << "Number of corresponding image files does not match" << endl;
        return false;
    }
    string gt_path = concatPath(mainPath, info.gt12.sub_folder);
    fileprefl = "*" + info.gt12.postfix;
    res = loadImageSequenceNew(gt_path, fileprefl, filenamesgt);
    if(res){
        return false;
    }
    if(nr_files != filenamesgt.size()){
        cerr << "Number of corresponding image and GT files does not match" << endl;
        return false;
    }
    for (size_t i = 0; i < nr_files; ++i) {
        fileNames.emplace_back(move(filenamesl[i]), move(filenamesr[i]), move(filenamesgt[i]));
    }
    return true;
}

//Prepare GTM from Oxford dataset
bool baseMatcher::calcGTM_Oxford(size_t &min_nrTP) {
    string path = concatPath(imgsPath, "Oxford");
    if(checkPathExists(path)){
        if(!getOxfordDatasets(path)){
            return false;
        }
    }else{
        if(!createDirectory(path)){
            return false;
        }
        if(!getOxfordDatasets(path)){
            return false;
        }
    }
    return getOxfordGTM(path, min_nrTP);
}

bool baseMatcher::getOxfordGTM(const std::string &path, size_t &min_nrTP){
    gtmdata.sum_TP_Oxford = 0;
    gtmdata.sum_TN_Oxford = 0;
    for(auto &sub: GetOxfordSubDirs()){
        const string sub_path = concatPath(path, sub);
        const string gtm_path = concatPath(sub_path, gtm_sub_folder);
        std::vector<std::pair<std::string, std::string>> imgNames, imgNames_tmp;
        std::vector<cv::Mat> homographies;
        flowGtIsUsed = false;
        if(!loadOxfordImagesHomographies(sub_path, imgNames, homographies)){
            return false;
        }
        size_t nrGts = imgNames.size();
        if(checkPathExists(gtm_path)){
            for (size_t i = 0; i < nrGts; ++i){
                if(!loadGTM(gtm_path, imgNames[i])){
                    if(!calcRefineStoreGTM_Oxford(imgNames[i], homographies[i], gtm_path, sub, static_cast<int>(nrGts - i), true)){
                        continue;
                    }
                }
                addGTMdataToPool();
                gtmdata.sum_TP_Oxford += positivesGT;
                gtmdata.sum_TN_Oxford += negativesGT;
                imgNames_tmp.push_back(imgNames[i]);
            }
            if(imgNames_tmp.empty()){
                cerr << "Unable to load/calculate GTM for Oxford subset " << sub << endl;
                return false;
            }
            gtmdata.imgNamesAll.emplace_back(std::move(imgNames_tmp));
            gtmdata.sourceGT.emplace_back('O');
        }else{
            bool save_it = true;
            if(!createDirectory(gtm_path)){
                cerr << "Unable to create GTM directory" << endl;
                save_it = false;
            }
            for (size_t i = 0; i < nrGts; ++i) {
                if(!calcRefineStoreGTM_Oxford(imgNames[i], homographies[i], gtm_path, sub, static_cast<int>(nrGts - i), save_it)){
                    continue;
                }
                addGTMdataToPool();
                gtmdata.sum_TP_Oxford += positivesGT;
                gtmdata.sum_TN_Oxford += negativesGT;
                imgNames_tmp.push_back(imgNames[i]);
            }
            if(imgNames_tmp.empty()){
                cerr << "Unable to calculate GTM for Oxford subset " << sub << endl;
                return false;
            }
            gtmdata.imgNamesAll.emplace_back(std::move(imgNames_tmp));
            gtmdata.sourceGT.emplace_back('O');
        }
        if(gtmdata.sum_TP_Oxford >= min_nrTP){
            break;
        }
    }
    return true;
}

void baseMatcher::addGTMdataToPool(){
    gtmdata.sum_TP += positivesGT;
    gtmdata.sum_TN += negativesGT;
    gtmdata.keypLAll.emplace_back(std::move(keypL));
    gtmdata.keypRAll.emplace_back(std::move(keypR));
    gtmdata.leftInlierAll.emplace_back(std::move(leftInlier));
    gtmdata.rightInlierAll.emplace_back(std::move(rightInlier));
    gtmdata.matchesGTAll.emplace_back(std::move(matchesGT));
}

bool baseMatcher::loadGTM(const std::string &gtm_path, const std::pair<std::string, std::string> &imageNames){
    string GT_filename = prepareFileNameGT(imageNames, gtm_path);
    if(!checkFileExists(GT_filename)){
        return false;
    }
    if(!readGTMatchesDisk(GT_filename)){
        return false;
    }
    if(refinedGTMAvailable && refineGTM){
        getRefinedGTM();
        switchToRefinedGTM();
    }
    return true;
}

bool baseMatcher::calcRefineStoreGTM_Oxford(const std::pair<std::string, std::string> &imageNames, const cv::Mat &H,
                                            const std::string &gtm_path, const std::string &sub,
                                            const int &remainingImgs, bool save_it) {
    if(!calculateGTM_Oxford(imageNames, H)){
        cout << "Unable to calculate GTM for images " << imageNames.first << " --> " << imageNames.second << endl;
        return false;
    }
    bool is_refined = false;
    if(refineGTM){
        if(refineFoundGTM(remainingImgs)) {
            quality.id = "Oxford-" + sub + "-" + concatImgNames(imageNames);
            is_refined = true;
        }else{
            cout << "Unable to refine GTM" << endl;
        }
    }
    if(save_it) {
        string GT_filename = prepareFileNameGT(imageNames, gtm_path);
        if (!writeGTMatchesDisk(GT_filename, is_refined)) {
            cerr << "Unable to store GTM for images " << imageNames.first << " --> " << imageNames.second
                 << endl;
        }
    }
    if(is_refined){
        switchToRefinedGTM();
    }
    return true;
}

bool baseMatcher::calcRefineStoreGTM_KITTI(const std::tuple<std::string, std::string, std::string> &fileNames,
                                           bool is_flow, const std::string &gtm_path, const std::string &sub,
                                           const int &remainingImgs, bool save_it) {
    if(!getKittiGTM(get<0>(fileNames), get<1>(fileNames), get<2>(fileNames), is_flow)){
        cout << "Unable to calculate GTM for images " << get<0>(fileNames) << " --> " << get<1>(fileNames) << endl;
        return false;
    }
    std::pair<std::string, std::string> imageNames = make_pair(get<0>(fileNames), get<1>(fileNames));
    bool is_refined = false;
    if(refineGTM){
        if(refineFoundGTM(remainingImgs)) {
            string sub1 = sub;
            std::replace(sub1.begin(), sub1.end(), '/', '_');
            quality.id = "KITTI-" + sub1 + "-" + concatImgNames(imageNames);
            is_refined = true;
        }else{
            cout << "Unable to refine GTM" << endl;
        }
    }
    if(save_it) {
        string GT_filename = prepareFileNameGT(imageNames, gtm_path);
        if (!writeGTMatchesDisk(GT_filename, is_refined)) {
            cerr << "Unable to store GTM for images " << imageNames.first << " --> " << imageNames.second
                 << endl;
        }
    }
    if(is_refined){
        switchToRefinedGTM();
    }
    return true;
}

bool baseMatcher::refineFoundGTM(int remainingImgs){
    int nr_GTMs = static_cast<int>(matchesGT.size()), usedSamples = 0;
    quality.clear();
    bool res = testGTmatches(nr_GTMs, quality.falseGT, usedSamples, quality.distanceHisto,
                             quality.distances, remainingImgs, quality.notMatchable,
                             quality.errvecs, quality.perfectMatches, quality.matchesGT_idx,
                             quality.HE, quality.validityValFalseGT, quality.errvecsGT, quality.distancesGT,
                             quality.validityValGT, quality.distancesEstModel, quality.autoManualAnnot, "SIFT");
    if(!res){
        return false;
    }
    getRefinedGTM();
    return true;
}

void baseMatcher::getRefinedGTM(){
    matchesGT_refined.clear();
    keypL_refined.clear();
    keypR_refined.clear();
    leftInlier_refined.clear();
    rightInlier_refined.clear();
    size_t idx = 0, idx1 = 0;
    for(auto &i: quality.matchesGT_idx){
        if(quality.distancesEstModel[idx1] > 0.75){
            idx1++;
            continue;
        }
        DMatch m = matchesGT[i];
        KeyPoint p1 = keypL[m.queryIdx];
        KeyPoint p2 = keypR[m.trainIdx];
        p1.pt = quality.perfectMatches[idx1].first;
        p2.pt = quality.perfectMatches[idx1].second;
        m.queryIdx = m.trainIdx = idx;
        matchesGT_refined.emplace_back(m);
        keypL_refined.emplace_back(move(p1));
        keypR_refined.emplace_back(move(p2));
        idx1++;
        idx++;
    }
    leftInlier_refined = vector<bool>(keypL_refined.size(), true);
    rightInlier_refined = vector<bool>(keypR_refined.size(), true);
    positivesGT_refined = matchesGT_refined.size();
    CV_Assert(leftInlier.size() == rightInlier.size());
    for (size_t i = 0; i < leftInlier.size(); ++i) {
        if(!leftInlier[i]){
            keypL_refined.push_back(keypL[i]);
            leftInlier_refined.emplace_back(false);
        }
        if(!rightInlier[i]){
            keypR_refined.push_back(keypR[i]);
            rightInlier_refined.emplace_back(false);
        }
    }
    negativesGT_refined = rightInlier_refined.size() - positivesGT_refined;
    inlRatio_refined = static_cast<double>(positivesGT_refined) / static_cast<double>(positivesGT_refined + negativesGT_refined);
}

void baseMatcher::switchToRefinedGTM(){
    matchesGT = move(matchesGT_refined);
    keypL = move(keypL_refined);
    keypR = move(keypR_refined);
    leftInlier = move(leftInlier_refined);
    rightInlier = move(rightInlier_refined);
    positivesGT = positivesGT_refined;
    negativesGT = negativesGT_refined;
    inlRatio = inlRatio_refined;
}

bool baseMatcher::calculateGTM_Oxford(const std::pair<std::string, std::string> &imageNames, const cv::Mat &H){
    imgs[0] = cv::imread(imageNames.first, cv::IMREAD_GRAYSCALE);
    imgs[1] = cv::imread(imageNames.second, cv::IMREAD_GRAYSCALE);
    homoGT = H;
    if(!detectFeatures()){
        return false;
    }
    int err = filterInitFeaturesGT();
    return err == 0;
}

bool baseMatcher::loadOxfordImagesHomographies(const std::string &path,
        std::vector<std::pair<std::string, std::string>> &imgNames,
        std::vector<cv::Mat> &homographies){
    imgNames.clear();
    homographies.clear();
    vector<string> filenamesi, hnames;
    cv::Mat H;
    bool err = loadImageSequence(path, "img", filenamesi);
    if (!err || filenamesi.empty())
    {
        cerr << "Could not find Oxford images!" << endl;
        return false;
    }
    err = readHomographyFiles(path, "H1to", hnames);
    size_t nr_hs = hnames.size();
    size_t nr_is = filenamesi.size();
    if (!err || hnames.empty() || ((nr_hs + 1) != nr_is))
    {
        cerr << "Could not find homography files or number of provided homography files is wrong!" << endl;
        return false;
    }
    std::vector<cv::Mat> Hs(nr_hs);
    for (size_t idx1 = 0; idx1 < nr_hs; idx1++)
    {
        err = readHomographyFromFile(path, hnames[idx1], Hs[idx1]);
        if (!err)
        {
            cerr << "Error opening homography file with index " << idx1 << endl;
            return false;
        }
    }

    string iname0 = concatPath(path, filenamesi[0]);
    string iname1;
    for (size_t idx1 = 0; idx1 < nr_is; idx1++)
    {
        homographies.emplace_back(Hs[idx1].clone());
        iname1 = concatPath(path, filenamesi[idx1 + 1]);
        imgNames.emplace_back(iname0, iname1);
    }
    //Generate new homographies to evaluate all other possible configurations of the images to each other
    for (size_t idx1 = 0; idx1 < nr_hs - 1; idx1++)
    {
        for (size_t idx2 = idx1 + 1; idx2 < nr_hs; idx2++)
        {
            homographies.emplace_back(Hs[idx2] * Hs[idx1].inv());
            iname0 = concatPath(path, filenamesi[idx1 + 1]);
            iname1 = concatPath(path, filenamesi[idx2 + 1]);
            imgNames.emplace_back(iname0, iname1);
        }
    }
    return true;
}

//Check if GT data is available for the Oxford dataset. If not, download it
bool baseMatcher::getOxfordDatasets(const std::string &path){
    for(auto &sub: GetOxfordSubDirs()){
        if(!getOxfordDataset(path, sub)){
            return false;
        }
    }
}

//Check if GT data is available for an Oxford sub-set. If not, download it
bool baseMatcher::getOxfordDataset(const std::string &path, const std::string &datasetName){
    const static string ext = ".tar.gz";
    const string sub_path = concatPath(path, datasetName);
    if(!checkPathExists(sub_path)){
        if(!createDirectory(sub_path)){
            cerr << "Unable to create directory " << sub_path << endl;
            cerr << "Disabling use of GTM for Oxford dataset!" << endl;
            return false;
        }
#if __linux__
        string command = "cd " + sub_path + " && curl -O " + base_url_oxford + datasetName + ext;
        int ret = system(command.c_str());
        if(ret){
            cerr << "Unable to download Oxford dataset using command " << command << endl;
            cerr << "Disabling use of GTM for Oxford dataset!" << endl;
            return false;
        }
        command = "cd " + sub_path + " && tar -xvzf " + datasetName + ext + "-C ./";
        ret = system(command.c_str());
        if(ret){
            cerr << "Unable to extract Oxford dataset using command " << command << endl;
            cerr << "Disabling use of GTM for Oxford dataset!" << endl;
            return false;
        }
        if(!deleteFile(concatPath(sub_path, datasetName + ext))){
            cerr << "Unable to delete downloaded archive." << endl;
        }
#else
        cerr << "Unable to download Oxford dataset. Please download it manually at http://www.robots.ox.ac.uk/~vgg/research/affine/" << endl;
                cerr << "Disabling use of GTM for Oxford dataset!" << endl;
                return false;
#endif
    }else {
        if (!checkOxfordSubDataset(sub_path)) {
            if(!deleteDirectory(sub_path)){
                return false;
            }
            if(!getOxfordDataset(path, datasetName)){
                return false;
            }
        }
    }
    return true;
}

//Check if an Oxford dataset folder is complete
bool baseMatcher::checkOxfordSubDataset(const std::string &path){
    std::vector<std::string> filenames;
    if(!readHomographyFiles(path, "H1to", filenames)){
        return false;
    }
    if(filenames.size() != 6){
        return false;
    }
    filenames.clear();
    if(!loadImageSequence(path, "img", filenames)){
        return false;
    }
    return filenames.size() == 6;
}


std::vector<std::string> baseMatcher::GetOxfordSubDirs()
{
    int const nrSupportedTypes = 8;

    static std::string types [] = {"bikes",
                                   "trees",
                                   "graf",
                                   "wall",
                                   "bark",
                                   "boat",
                                   "leuven",
                                   "ubc"
    };
    return std::vector<std::string>(types, types + nrSupportedTypes);
}

std::vector<kittiFolders> baseMatcher::GetKITTISubDirs(){
    static kittiFolders dispflow [] = {{{"2012/image_0", "_10"}, {"2012/image_1", "_10"}, {"2012/disp_occ", "_10"}, false},
                                       {{"2012/image_0", "_10"}, {"2012/image_0", "_11"}, {"2012/flow_occ", "_10"}, true},
                                       {{"2015/image_2", "_10"}, {"2015/image_3", "_10"}, {"2015/disp_occ_0", "_10"}, false},
                                       {{"2015/image_2", "_10"}, {"2015/image_2", "_11"}, {"2015/flow_occ", "_10"}, true}};
    return std::vector<kittiFolders>(dispflow, dispflow + 4);
}


#if defined(USE_MANUAL_ANNOTATION)
void on_mouse_click(int event, int x, int y, int flags, void* param){
	if(event == cv::EVENT_LBUTTONDOWN){
		if(flags == (cv::EVENT_FLAG_CTRLKEY | cv::EVENT_FLAG_LBUTTON)){
			*((cv::Point2f*)param) = cv::Point2f((float)(-x), (float)(-y));
		}else{
			*((cv::Point2f*)param) = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
		}
	}
}

SpecialKeyCode getSpecialKeyCode(int & val){
    int sk = NONE;
#ifdef __linux__
    //see X11/keysymdef.h and X11/XF86keysym.h for more mappings
    switch(val & 0x0000ffff){
    case (0xffb0):{ val = '0';      break;}
    case (0xffb1):{ val = '1';      break;}
    case (0xffb2):{ val = '2';      break;}
    case (0xffb3):{ val = '3';      break;}
    case (0xffb4):{ val = '4';      break;}
    case (0xffb5):{ val = '5';      break;}
    case (0xffb6):{ val = '6';      break;}
    case (0xffb7):{ val = '7';      break;}
    case (0xffb8):{ val = '8';      break;}
    case (0xffb9):{ val = '9';      break;}
    case (0xffab):{ val = '+';      break;}
    case (0xffad):{ val = '-';      break;}
    case (0xffaa):{ val = '*';      break;}
    case (0xffaf):{ val = '/';      break;}
    case (0xff09): case (0x0009): { val = '\t'; break;} //TAB
    case (0xffae): case (0xffac): { val = '.';      break;}
    case (0x0020): case (0xff80): { sk = SPACE;     break;}
    case (0xff08): { sk = BACKSPACE; break;}
    case (0xff8d) : { sk = CARRIAGE_RETURN; break;}
    case (0x001b):{ sk = ESCAPE;    break;}
    case (0x5B41): case (0xff52): case (0xff97): { sk = ARROW_UP;    break;}
    case (0x5B43): case (0xff53): case (0xff98): { sk = ARROW_RIGHT; break;}
    case (0x5B42): case (0xff54): case (0xff99): { sk = ARROW_DOWN;  break;}
    case (0x5B44): case (0xff51): case (0xff96): { sk = ARROW_LEFT;  break;}
    case (0x5B35): case (0xff55): case (0xff9a): { sk = PAGE_UP  ;   break;}
    case (0x5B36): case (0xff56): case (0xff9b): { sk = PAGE_DOWN;   break;}
    case (0x4F48): case (0xff50): case (0xff95): case (0xff58): { sk = POS1         ;break;}
    case (0x4F46): case (0xff57): case (0xff9c): { sk = END_KEY      ;break;}
    case (0xff63): case (0xff9e): { sk = INSERT       ;break;} //TODO: missing for kbhit case
    case (0xffff): case (0xff9f): { sk = DELETE_KEY   ;break;} //TODO: missing for kbhit case

#else
    switch(val & 0x00ffffff){
    case (0x090000): { val = '\t'; break;} //TAB
    case (0x000020 ):{ sk = SPACE;break;}
    case (0x000008 ):{ sk = BACKSPACE;break;}
    case (0x00001b ):{ sk = ESCAPE;break;}
    case (0xe048): case (0x260000 ):{ sk = ARROW_UP;break;}
    case (0xe04D): case (0x270000 ):{ sk = ARROW_RIGHT;break;}
    case (0xe050): case (0x280000 ):{ sk = ARROW_DOWN;break;}
    case (0xe04B): case (0x250000 ):{ sk = ARROW_LEFT; break;}
    case (0xe049): case (0x210000 ):{ sk = PAGE_UP  ;break;}
    case (0xe051): case (0x220000 ):{ sk = PAGE_DOWN;break;}
    case (0xe047): case (0x240000 ):{ sk = POS1     ;break;}
    case (0xe04F): case (0x230000 ):{ sk = END_KEY  ;break;}
    case (0xe052): case (0x2d0000 ):{ sk = INSERT   ;break;}
    case (0xe053): case (0x2e0000 ):{ sk = DELETE_KEY  ;break;}
#endif
    default:
        sk=NONE;
    }

    if (sk == NONE)
      val = (val&0xFF);
    else
      val = 0x00;

    return (SpecialKeyCode)sk;
}
#endif