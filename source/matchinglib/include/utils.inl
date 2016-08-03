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
#include <iostream>

#if defined(__x86_64__) || defined(_WIN32)
#include <nmmintrin.h>
#define _USE_HW_POPCNT_ 1
#define USE_SSE 1
#endif

#if defined(__linux__)
#include <inttypes.h>
#include <cpuid.h>
#endif

#if defined(__arm__) || defined(_ARM)
#include <SSE2NEON.hpp>

#define _USE_HW_POPCNT_ 1
#define USE_SSE 1
#define USE_DESCRIPTOR_DISTANCE 1

#endif

namespace matchinglib
{
  int descriptorDistance(const cv::Mat &a, const cv::Mat &b, unsigned char byte8width);

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

#if USE_DESCRIPTOR_DISTANCE
    return descriptorDistance(vec1, vec2, byte8width);
#else

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
#endif
#else
    return 0;
#endif

  }

  // Bit set count operation from
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
  inline int descriptorDistance(const cv::Mat &a, const cv::Mat &b, unsigned char byte8width)
  {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
#if USE_SSE
    // old reference code, which is the hotspot:
    __m128i dist_ = _mm_set1_epi32(0);
    __m128i const cx55555555_= _mm_set1_epi32(0x55555555);
    __m128i const cx33333333_= _mm_set1_epi32(0x33333333);
    __m128i const cxF0F0F0F_= _mm_set1_epi32(0x0F0F0F0F);
    __m128i const cx3F_= _mm_set1_epi32(0x0000003F);

    for (int i = 0; i < byte8width/2; i++, pa += 4, pb += 4)
    {
      // unsigned int v = *pa ^*pb;
      __m128i a_ = _mm_load_si128((__m128i const*)pa); // use unaligned load
      __m128i b_ = _mm_load_si128((__m128i const*)pb); // use unaligned load
      __m128i v_ = _mm_xor_si128(a_, b_);

      // v = v - ((v >> 1) & 0x55555555);
      __m128i v__ = _mm_srli_epi32(v_, 1);
      v__ = _mm_and_si128(v__, cx55555555_);
      v_ =  _mm_sub_epi32(v_, v__);

      // v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      __m128i v___ =  _mm_and_si128(v_, cx33333333_);
      v__ = _mm_srli_epi32(v_, 2);
      v__ =  _mm_and_si128(v__, cx33333333_);
      v_ = _mm_add_epi32(v___, v__);

      // v = ((v + (v >> 4)) & 0xF0F0F0F);
      v__ = _mm_srli_epi32(v_, 4);
      v__ = _mm_add_epi32(v_, v__);
      v_ = _mm_and_si128(v__, cxF0F0F0F_);

      // v = v + (v >> 8);
      v__ = _mm_srli_epi32(v_, 8);
      v_ = _mm_add_epi32(v_, v__);


      // v = (v + (v >> 16)) & 0x0000003F;
      v__ = _mm_srli_epi32(v_, 16);
      v__ = _mm_add_epi32(v_, v__);
      v_ = _mm_and_si128(v__, cx3F_);

      // dist += v;
      dist_ = _mm_add_epi32(v_, dist_);
    }

    // horizontal add of four 32 bit partial sums and return result
    dist_ = _mm_add_epi32(dist_, _mm_srli_si128(dist_, 8));
    dist_ = _mm_add_epi32(dist_, _mm_srli_si128(dist_, 4));
    dist = _mm_cvtsi128_si32(dist_);
#else

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
      unsigned int v = *pa ^*pb;
      v = v - ((v >> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

#endif
    return dist;
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


  inline float calcMatchingCost(cv::Mat const& vDescriptors1, cv::Mat const& vDescriptors2, unsigned const idx1, const unsigned idx2)
  {
    //Calculate the matching costs for the initial keypoints
    if(vDescriptors1.type() == CV_8U)
    {
      if(IsPopCntAvailable())
      {
        unsigned char byte8width = (unsigned char)vDescriptors1.cols >> 3;
        return (float)getHammingL1PopCnt(vDescriptors1.row(idx1), vDescriptors2.row(idx2), byte8width);
      }
      else
      {
        return (float)getHammingL1(vDescriptors1.row(idx1), vDescriptors2.row(idx2));
      }
    }
    else if(vDescriptors1.type() == CV_32F)
    {
      return getL2Distance(vDescriptors1.row(idx1), vDescriptors2.row(idx2));
    }
    else
    {
      std::cout << "-- ERROR: descriptor type not supported!" << std::endl;
      return 0.0f; //Descriptor type not supported
    }
  }

} // namespace matchinglib
