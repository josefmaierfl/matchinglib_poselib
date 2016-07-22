/******************************************************************************
* FILENAME:     utils
* PURPOSE:      %{Cpp:License:ClassName}
* AUTHOR:       jungr - Roland Jung
* MAIL:         Roland.Jung@ait.ac.at
* VERSION:      v1.0.0
*
*  Copyright (C) 2016 Austrian Institute of Technologies GmbH - AIT
*  All rights reserved. See the LICENSE file for details.
******************************************************************************/

#ifndef UTILS_HPP
#define UTILS_HPP
#include <opencv2/core.hpp>


namespace matchinglib
{

  inline bool IsPopCntAvailable();
  //This function calculates the L1-norm of the hamming distance between two column vectors using a LUT.
  inline unsigned getHammingL1(cv::Mat const& vec1, cv::Mat const& vec2);
  //This function calculates the L1-norm of the hamming distance between two column vectors using the CPU popcnt instruction.
  inline unsigned getHammingL1PopCnt(cv::Mat const& vec1, cv::Mat const& vec2, unsigned char byte8width);
  //This function calculates the L2-norm of two descriptors
  inline float getL2Distance(cv::Mat const& vec1, cv::Mat const& vec2);

  //
  inline float calcMatchingCost(cv::Mat const& vDescriptors1, cv::Mat const& vDescriptors2, unsigned const idx1, unsigned const idx2);


} // namespace matchinglib


// hidden definition:
#include <utils.inl>

#endif // UTILS_HPP
