/**********************************************************************************************************
 FILE: io_data.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: September 2015

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file holds functionalities for loading image data, flow data from the KITTI dataset,
 flow data from O. Zendel's generated virtuel images and homographys from files of the test data
 from K. Mikolajczyk. Moreover, functions are provided to generate the different quality measurements
 for the matching algorithms.
**********************************************************************************************************/

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