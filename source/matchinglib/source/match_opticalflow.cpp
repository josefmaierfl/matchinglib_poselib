/******************************************************************************
* FILENAME:     match_opticalflow
* PURPOSE:      %{Cpp:License:ClassName}
* AUTHOR:       jungr - Roland Jung
* MAIL:         Roland.Jung@ait.ac.at
* VERSION:      v1.0.0
*
*  Copyright (C) 2016 Austrian Institute of Technologies GmbH - AIT
*  All rights reserved. See the LICENSE file for details.
******************************************************************************/

#include "matchinglib_matchers.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <PointCloudOperation.hpp>
#include <utils.hpp>

using namespace PointCloud;

namespace matchinglib
{
  typedef PointCloud::Point2D<float> ptType;
  typedef std::vector<ptType> vecPoint2D;


  vecPoint2D toVecPoint2D(std::vector<cv::Point2f> const& pts_in)
  {
    vecPoint2D pts;
    int idx = 0;

    for(cv::Point2f p : pts_in)
    {
      pts.push_back(ptType(p.x, p.y, idx));
      idx++;
    }

    return pts;
  }

  vecPoint2D toVecPoint2D(std::vector<cv::Point2f> const& pts_in, std::vector<uchar> status)
  {
    vecPoint2D pts;
    assert(pts_in.size() == status.size());

    int idx = 0;

    for(cv::Point2f p : pts_in)
    {
      if(status[idx])
      {
        pts.push_back(ptType(p.x, p.y, idx));
      }

      idx++;
    }

    return pts;
  }

  vecPoint2D toVecPoint2D(std::vector<cv::KeyPoint> const& keypts)
  {
    vecPoint2D pts;
    int idx = 0;

    for(cv::KeyPoint i : keypts)
    {
      pts.push_back(ptType(i.pt.x, i.pt.y, idx));
      idx++;
    }

    return pts;
  }

  std::vector<cv::Point2f> toPoints(std::vector<cv::KeyPoint> const& keypts)
  {
    std::vector<cv::Point2f> pts;

    for(cv::KeyPoint i : keypts)
    {
      pts.push_back(i.pt);
    }

    return pts;
  }

  //int match_PredictedPts_and_NextKeypoints(std::vector<cv::Point2f> const& pts_predict_next, const std::vector<cv::KeyPoint> &keypoints_next, )

  int MATCHINGLIB_API getMatches_OpticalFlow(const std::vector<cv::KeyPoint> &keypoints_prev, const std::vector<cv::KeyPoint> &keypoints_next,
      cv::Mat& img_prev, cv::Mat const& img_next, std::vector<cv::DMatch> & finalMatches,
      bool const buildpyr, bool drawRes, cv::Size winSize, float searchRadius_px)
  {
    return getMatches_OpticalFlowAdvanced(keypoints_prev, keypoints_next,
                                          cv::Mat(), cv::Mat(),
                                          img_prev, img_next,
                                          finalMatches,
                                          "LKOF",
                                          buildpyr,
                                          drawRes,
                                          winSize,
                                          searchRadius_px, 1);
  }

  int MATCHINGLIB_API getMatches_OpticalFlowAdvanced(const std::vector<cv::KeyPoint> &keypoints_prev,
      const std::vector<cv::KeyPoint> &keypoints_next,
      cv::Mat const& descriptors1, cv::Mat const& descriptors2,
      cv::Mat &img_prev, cv::Mat const& img_next, std::vector<cv::DMatch> & finalMatches, std::string const& matcher_name,
      bool const buildpyr, bool drawRes, cv::Size winSize, float searchRadius_px,
      unsigned const numNeighbors)
  {
    size_t const maxNumNeighbors = 10;
    assert(numNeighbors <= maxNumNeighbors);
    assert(img_next.size() == img_prev.size());
    assert(matcher_name == "LKOF" || matcher_name == "ALKOF");
    assert(img_next.type() == CV_8UC1 && img_prev.type() == CV_8UC1);
    assert(descriptors1.type() == CV_8U && descriptors2.type() == descriptors1.type());


    static cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.013);
    std::vector<uchar> vStatus;
    std::vector<float> err;
    cv::Mat iprev, inext;

    if(buildpyr)
    {
      int levels;
      int const max_levels = 3; // TODO: add as function argument.
      levels = cv::buildOpticalFlowPyramid(img_prev, iprev, winSize, max_levels);
      levels = cv::buildOpticalFlowPyramid(img_next, inext, winSize, max_levels);
    }
    else
    {
      iprev = img_prev;
      inext = img_next;
    }

    std::vector<cv::Point2f> pts_predict_next, pts_prev = toPoints(keypoints_prev);

    cv::calcOpticalFlowPyrLK(iprev, inext,
                             pts_prev, pts_predict_next,
                             vStatus, err,
                             winSize,
                             3,
                             termcrit,
                             0, //cv::OPTFLOW_USE_INITIAL_FLOW,
                             0.05);

    if(drawRes)
    {
      /// Create Windows
      ///
      std::string img_name = "getMatches_OpticalFlow + blend + optical flow";
      cv::namedWindow(img_name, 1);
      double alpha = 0.3;
      double beta;
      cv::Mat dst;
      beta = ( 1.0 - alpha );
      cv::addWeighted( img_prev, alpha, img_next, beta, 0.0, dst);

      cv::Mat cimg;
      cvtColor(dst, cimg, CV_GRAY2RGB);

      // draw keypoints:
      for(unsigned i = 0; i < keypoints_prev.size(); i++)
      {
        cv::circle(cimg, keypoints_prev[i].pt, 1, cv::Scalar(192, 0,0));
      }

      for(unsigned i = 0; i < keypoints_next.size(); i++)
      {
        cv::circle(cimg, keypoints_next[i].pt, 1, cv::Scalar(0, 192,0));
      }

      // draw the prediction
      for(unsigned i = 0; i < pts_prev.size(); i++)
      {
        if(vStatus[i])
        {
          cv::line(cimg, pts_prev[i], pts_predict_next[i], cv::Scalar(0, 0, 255));
        }
      }


      cv::imshow( img_name, cimg );
      cv::waitKey(1);
    }

    // now check if the pts_predict_next are in the keypoints_next visible!
    std::vector<Point2D<float>> vec2d_predict_next = matchinglib::toVecPoint2D(pts_predict_next, vStatus);
    std::vector<Point2D<float>> vec2d_keypts_next = matchinglib::toVecPoint2D(keypoints_next);

    SmartPointCloud<float, Cloud2D_Adaptor<float>, Point2D<float>, 2> smartCloud2d(vec2d_keypts_next);

    float const maxDist_sqr = searchRadius_px*searchRadius_px;

    finalMatches.clear();

    size_t arr_ret_index[maxNumNeighbors];
    float arr_out_dist_sqr[maxNumNeighbors];

    // TODO: check if there are enough pts to track! or to do a nearest neighbor search!

    // without descriptos
    if(matcher_name == "LKOF")
    {
      for(unsigned i = 0; i <vec2d_predict_next.size(); i++)
      {
        smartCloud2d.findKnn(vec2d_predict_next[i], 1, arr_ret_index, arr_out_dist_sqr);

        if(arr_out_dist_sqr[0] < maxDist_sqr)
        {
          int idx_pred = vec2d_predict_next[i].id;
          int idx_next = vec2d_keypts_next[arr_ret_index[0]].id;
          finalMatches.push_back(cv::DMatch(idx_pred, idx_next, arr_out_dist_sqr[0]));
        }
      }

    }
    // with descriptors and more neighbors are considered to be possible matches -> hamming distance
    else if(matcher_name == "ALKOF")
    {
      for(unsigned i = 0; i <vec2d_predict_next.size(); i++)
      {

        smartCloud2d.findKnn(vec2d_predict_next[i], numNeighbors, arr_ret_index, arr_out_dist_sqr);

        if(numNeighbors == 1)
        {
          if(arr_out_dist_sqr[0] < maxDist_sqr)
          {
            int idx_pred = vec2d_predict_next[i].id;
            int idx_next = vec2d_keypts_next[arr_ret_index[0]].id;
            finalMatches.push_back(cv::DMatch(idx_pred, idx_next, arr_out_dist_sqr[0]));
          }
        }
        else // more neighbors have to be considered! calculate the hamming distance
        {
          float const max_ham_dist = 50.0f;
          float min = 10E8;
          float dist = min;
          int idx_min = -1;
          int idx_pred = vec2d_predict_next[i].id;
          // TODO:
          cv::Mat desc_predict = descriptors1.row(idx_pred);
          cv::Mat desc_posible_next;
          bool isPopCnt = IsPopCntAvailable();
          unsigned char byte8width = (unsigned char)descriptors1.cols >> 3;

          for(unsigned k = 0; k < numNeighbors; k++)   // iterate over the neigh.
          {
            if(arr_out_dist_sqr[k] < maxDist_sqr)
            {
              int idx_next = vec2d_keypts_next[arr_ret_index[k]].id;
              desc_posible_next = descriptors2.row(idx_next);
              float res;

              if(isPopCnt)
              {
                res = (float)getHammingL1PopCnt(desc_predict, desc_posible_next, byte8width);
              }
              else
              {
                res = (float)getHammingL1(desc_predict, desc_posible_next);
              }

              // better match found: store its data:
              if(res < max_ham_dist && res < min)
              {
                min = res;
                idx_min = idx_next;
                dist = arr_out_dist_sqr[k];
              }
            }
          }

          // add the result if valid!
          if(idx_min >= 0)
          {
            finalMatches.push_back(cv::DMatch(idx_pred, idx_min, dist));
          }
        }
      }
    }

    return 0;
  }

} // namespace matchinglib

