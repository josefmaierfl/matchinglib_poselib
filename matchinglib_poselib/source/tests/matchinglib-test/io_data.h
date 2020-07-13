//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2019 AIT Austrian Institute of Technology GmbH
//
//Permission is hereby granted, free of charge, to any person obtaining
//a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation
//the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included
//in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

#pragma once

#include "glob_includes.h"
//#include "opencv2/highgui/highgui.hpp"

//#include "PfeImgFileIO.h"

//This function reads all stereo or 2 subsequent images from a given directory and stores their names into two vectors.
int loadStereoSequence(std::string filepath, std::string fileprefl, std::string fileprefr,
					   std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr);
//This function reads all images from a given directory and stores their names into a vector.
int loadImageSequence(std::string filepath, std::string fileprefl, std::vector<std::string> & filenamesl);
//This function takes an 16Bit RGB integer image and converts it to a 3-channel float flow matrix.
//int convertImageFlowFile(std::string filepath, std::string filename, cv::Mat* flow3, const float precision = 64.0f, 
//						 bool useBoolValidity = true, const float validityPrecision = 64.0f, const float minConfidence = 1.0f);
////This function takes an 16Bit 1-channel integer image (grey values) and converts it to a 3-channel float flow matrix
//int convertImageDisparityFile(std::string filepath, std::string filename, cv::Mat* flow3, const bool useFLowStyle = true, const float precision = 256.0f, const bool use0Invalid = true);
////This function reads all homography file names from a given directory and stores their names into a vector.
//int readHomographyFiles(std::string filepath, std::string fileprefl, std::vector<std::string> & filenamesl);
////This function reads a homography from a given file.
//int readHomographyFromFile(std::string filepath, std::string filename, cv::Mat* H);

/*!***************************************************************************
* PURPOSE: returns cv::Mat datastructure with PfePixImgStruct data
*
* DESCRIPTION:
*
* RETURN:
*     cv::Mat
****************************************************************************!*/
//static cv::Mat PfeConvToMat
//  ( PfePixImgStruct *   pPfe
//  )
//{
//  int type = CV_8UC1;
//
//  if ( pPfe->u16BitsPerPixel == 8 )
//  {
//    type = CV_8UC1;
//  }
//  else if ( pPfe->u16BitsPerPixel == 16 )
//  {
//    type = CV_16UC1;
//  }
//  else if ( pPfe->u16BitsPerPixel == 24 )
//  {
//    type = CV_8UC3;
//  }
//  else if ( pPfe->u16BitsPerPixel == 32 )
//  {
//    type = CV_32FC1;
//  }
//  else if ( pPfe->u16BitsPerPixel == 48 )
//  {
//	  type = CV_16UC3;
//  }
//
//  return cv::Mat
//      ( pPfe->u32Height         // _rows
//      , pPfe->u32Width          // _cols
//      , type                    // _type
//      , pPfe->pBits             // _data
//      , pPfe->u32BytesPerLine   // _step
//      );
//}