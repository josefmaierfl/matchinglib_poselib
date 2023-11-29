/******************************************************************************
* FILENAME:     utils_opticalflow
* PURPOSE:      %{Cpp:License:ClassName}
* AUTHOR:       jungr - Roland Jung
* MAIL:         Roland.Jung@ait.ac.at
* VERSION:      v1.0.0
*
*  Copyright (C) 2016 Austrian Institute of Technologies GmbH - AIT
*  All rights reserved. See the LICENSE file for details.
******************************************************************************/

#ifndef UTILS_OPTICALFLOW_HPP
#define UTILS_OPTICALFLOW_HPP
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <PointCloudOperation.hpp>

namespace matchinglib
{
  typedef PointCloud::Point2D<float> ptType;
  typedef std::vector<ptType> vecPoint2D;


  static vecPoint2D toVecPoint2D(std::vector<cv::Point2f> const& pts_in)
  {
    vecPoint2D pts;
    //int idx = 0;
    pts.reserve(pts_in.size());

    for(int i = 0; i < static_cast<int>(pts_in.size()); i++)//(cv::Point2f p : pts_in)
    {
		pts.push_back(ptType(pts_in[i].x, pts_in[i].y, i));
      /*pts.push_back(ptType(p.x, p.y, idx));
      idx++;*/
    }

    return pts;
  }


  static vecPoint2D toVecPoint2D(std::vector<cv::Point2f> const& pts_in, std::vector<uchar> status)
  {
    vecPoint2D pts;
    assert(pts_in.size() == status.size());
    pts.reserve(pts_in.size());
    //int idx = 0;

    for (int i = 0; i < static_cast<int>(pts_in.size()); i++) //(cv::Point2f p : pts_in)
    {
      if(status[i])
      {
		  pts.push_back(ptType(pts_in[i].x, pts_in[i].y, i));
        //pts.push_back(ptType(p.x, p.y, idx));
      }

      //idx++;
    }

    return pts;
  }

  static vecPoint2D toVecPoint2D(std::vector<cv::KeyPoint> const& keypts)
  {
    vecPoint2D pts;
    //int idx = 0;
    pts.reserve(keypts.size());

    for (int i = 0; i < static_cast<int>(keypts.size()); i++) //(cv::KeyPoint i : keypts)
    {
      pts.push_back(ptType(keypts[i].pt.x, keypts[i].pt.y, i));
      //idx++;
    }

    return pts;
  }

  static std::vector<cv::KeyPoint> toVecKeypoint(std::vector<cv::Point2f> const& pts)
  {
    std::vector<cv::KeyPoint> kpts;
    kpts.reserve(pts.size());

    //int idx = 0;

    for (int i = 0; i < static_cast<int>(pts.size()); i++) //(auto i : pts)
    {
		kpts.push_back(cv::KeyPoint(pts[i], 0.0f, -1, 0, 0, i));
     // kpts.push_back(cv::KeyPoint(i, 0.0f, -1, 0, 0, idx));
      //idx++;
    }

    return kpts;
  }

  static std::vector<cv::Point2f> toPoints(std::vector<cv::KeyPoint> const& keypts)
  {
    std::vector<cv::Point2f> pts;
    pts.reserve(keypts.size());

    for(size_t i = 0; i < keypts.size(); i++)//(cv::KeyPoint i : keypts)
    {
		pts.push_back(keypts[i].pt);
      //pts.push_back(i.pt);
    }

    return pts;
  }

  ///
  /// \brief PrintOpticalFlowRes is printing the prev. and next keypoints and the prediction of the optical flow.
  ///        the img_prev and img_next are added weighted so they are overlayed for better debugging
  /// \param img_prev
  /// \param img_next
  /// \param outImg_rgb
  /// \param keypoints_prev
  /// \param keypoints_next
  /// \param pts_prev
  /// \param pts_predict_next
  /// \param vStatus
  /// \param showImg
  /// \param waitTime
  /// \param winName
  ///
  static void PrintOpticalFlowRes(cv::InputArray img_prev, cv::InputArray img_next, cv::OutputArray outImg_rgb,
                                  const std::vector<cv::KeyPoint> &keypoints_prev,
                                  const std::vector<cv::KeyPoint> &keypoints_next,
                                  const std::vector<cv::Point2f>& pts_prev,
                                  const std::vector<cv::Point2f>& pts_predict_next,
                                  const std::vector<uchar>& vStatus,
                                  const bool showImg = true,
                                  const int waitTime = 1,
                                  const std::string winName="getMatches_OpticalFlow + blend + optical flow" )
  {
    std::string img_name = winName;

    if(showImg)
    {

      cv::namedWindow(img_name, 1);
    }

    double alpha = 0.3;
    double beta;
    cv::Mat dst;
    beta = ( 1.0 - alpha );
    cv::addWeighted( img_prev, alpha, img_next, beta, 0.0, dst);

    cv::Mat cimg;
    cvtColor(dst, cimg, cv::COLOR_GRAY2RGB);

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

    if(showImg)
    {
      cv::imshow( img_name, cimg );
      cv::waitKey(waitTime);
    }

    outImg_rgb.assign(cimg);
  }



} // namespace matchinglib


#endif // UTILS_OPTICALFLOW_HPP
