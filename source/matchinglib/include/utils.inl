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

#include <bitset>


#if defined(__x86_64__) || defined(_WIN32)
#include <nmmintrin.h>
#define _USE_HW_POPCNT_ 1
#endif

#if defined(__linux__)
#include <inttypes.h>
#include <cpuid.h>
#endif

#if defined(__arm__) || defined(_ARM)
#define _USE_HW_POPCNT_ 0
#endif

namespace matchinglib
{


  inline bool IsPopCntAvailable()
  {
#if !(_USE_HW_POPCNT_)
    return false;
#else
#if defined(__linux__)
    unsigned level = 1, eax = 1, ebx, ecx, edx;

    if(!__get_cpuid(level, &eax, &ebx, &ecx, &edx))
    {
      return false;
    }

    std::bitset<32> f_1_ECX_ = ecx;
#else
    int cpuInfo[4];
    __cpuid(cpuInfo,0x00000001);
    std::bitset<32> f_1_ECX_ = cpuInfo[2];  // ECX
#endif

    return (bool)f_1_ECX_[23];
#endif
  }

  /* This function calculates the L1-norm of the hamming distance between two column vectors
   * using a LUT.
   *
   * Mat vec1           Input  -> First vector which must be of type uchar or CV_8U
   * Mat vec1           Input  -> Second vector which must be the same size as vec1
   *                      and of type uchar or CV_8U
   *
   * Return value:        L1-norm of the hamming distance
   */
  inline unsigned getHammingL1(const cv::Mat &vec1, const cv::Mat &vec2)
  {
    static const unsigned char BitsSetTable256[256] =
    {
#   define B2(n) n,     n+1,     n+1,     n+2
#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
      B6(0), B6(1), B6(1), B6(2)
    };
    unsigned int l1norm = 0;

    for(int i = 0; i<vec1.size().width; i++)
    {
      l1norm += (unsigned int)BitsSetTable256[(vec1.at<uchar>(i)^vec2.at<uchar>(i))];
    }

    return l1norm;
  }

  /* This function calculates the L1-norm of the hamming distance between two column vectors using
   * the popcnt CPU-instruction
   *
   * Mat vec1           Input  -> First vector which must be of type uchar or CV_8U
   * Mat vec1           Input  -> Second vector which must be the same size as vec1
   *                      and of type uchar or CV_8U
   * unsigned char byte8width   Input  -> Number of bytes devided by 8 (64 bit) for one descriptor
   *
   * Return value:        L1-norm of the hamming distance
   */
  inline unsigned getHammingL1PopCnt(cv::Mat const& vec1, cv::Mat const &vec2, unsigned char byte8width)
  {
#if _USE_HW_POPCNT_
#ifdef __linux__
    __uint64_t hamsum1 = 0;
    const __uint64_t *inputarr1 = reinterpret_cast<const __uint64_t*>(vec1.data);
    const __uint64_t *inputarr2 = reinterpret_cast<const __uint64_t*>(vec2.data);
#else
    unsigned __int64 hamsum1 = 0;
    const unsigned __int64 *inputarr1 = reinterpret_cast<const unsigned __int64*>(vec1.data);
    const unsigned __int64 *inputarr2 = reinterpret_cast<const unsigned __int64*>(vec2.data);
#endif

    for(unsigned char i = 0; i < byte8width; i++)
    {
      hamsum1 += _mm_popcnt_u64(*(inputarr1 + i) ^ *(inputarr2 + i));
    }

    return (unsigned int)hamsum1;
#else
    return 0;
#endif

  }

  /* This function calculates the L2-norm between two column vectors of desriptors
   *
   * Mat vec1           Input  -> First vector which must be of type uchar or CV_32F
   * Mat vec1           Input  -> Second vector which must be the same size as vec1
   *                      and of type uchar or CV_8U
   *
   * Return value:        L1-norm of the hamming distance
   */
  inline float getL2Distance(cv::Mat const& vec1, cv::Mat const& vec2)
  {
    static int descrCols = vec1.cols;
    float vecsum = 0;

    for(int i = 0; i < descrCols; i++)
    {
      float hlp = vec1.at<float>(i) - vec2.at<float>(i);
      vecsum += hlp * hlp;
    }

    return vecsum;
  }

} // namespace matchinglib
