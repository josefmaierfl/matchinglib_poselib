/**********************************************************************************************************
 FILE: io_data.cpp

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

//#include "..\include\glob_includes.h"
#include "io_data.h"
#include <stdint.h>
#include <fstream>
#ifdef _LINUX
#include "sys/dir.h"
#else
#include "atlstr.h"
#include "dirent.h"
#endif
#include <algorithm>

//#include "PfeImgFileIO.h"
//#include "PfeConv.h"


using namespace std;
//using namespace cv;

/* --------------------- Function prototypes --------------------- */


/* --------------------- Functions --------------------- */

/* This function reads all stereo or 2 subsequent images from a given directory and stores their names into two vectors.
 *
 * string filepath				Input  -> Path to the directory
 * string fileprefl				Input  -> File prefix for the left or first images (last character must be "_" which is the
 *										  last character in the filename before the image number)
 * string fileprefr				Input  -> File prefix for the right or second images (last character must be "_" which is the
 *										  last character in the filename before the image number)
 * vector<string> filenamesl	Output -> Vector with sorted filenames of the left or first images in the given directory 
 *										  that correspond to the image numbers of filenamesr
 * vector<string> filenamesr	Output -> Vector with sorted filenames of the right or second images in the given directory 
 *										  that correspond to the image numbers of filenamesl
 *
 * Return value:				 0:		  Everything ok
 *								-1:		  Could not open directory
 *								-2:		  No corresponding images available
 *								-3:		  No images available
 */
int loadStereoSequence(std::string filepath, std::string fileprefl, std::string fileprefr,
					   std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr)
{
	DIR *dir;
	struct dirent *ent;
	if((dir = opendir(filepath.c_str())) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
			string filename;
			filename = string(ent->d_name);
			if(filename.compare(0,fileprefl.size(),fileprefl) == 0)
				filenamesl.push_back(filename);
			else if(filename.compare(0,fileprefr.size(),fileprefr) == 0)
				filenamesr.push_back(filename);
		}
		closedir(dir);

		if(filenamesl.empty())
		{
			perror("No left images available");
			return -3;
		}

		if(filenamesr.empty())
		{
			perror("No right images available");
			return -3;
		}

		sort(filenamesl.begin(),filenamesl.end(),
			 [](string const &first, string const &second){return atoi(first.substr(first.find_last_of("_")+1).c_str()) <
			 atoi(second.substr(second.find_last_of("_")+1).c_str());});

		sort(filenamesr.begin(),filenamesr.end(),
			 [](string const &first, string const &second){return atoi(first.substr(first.find_last_of("_")+1).c_str()) <
			 atoi(second.substr(second.find_last_of("_")+1).c_str());});

		size_t i = 0;
		while((i < filenamesr.size()) && (i < filenamesl.size()))
		{
			if(atoi(filenamesl[i].substr(filenamesl[i].find_last_of("_")+1).c_str()) <
			   atoi(filenamesr[i].substr(filenamesr[i].find_last_of("_")+1).c_str()))
			   filenamesl.erase(filenamesl.begin()+i,filenamesl.begin()+i+1);
			else if(atoi(filenamesl[i].substr(filenamesl[i].find_last_of("_")+1).c_str()) >
					atoi(filenamesr[i].substr(filenamesr[i].find_last_of("_")+1).c_str()))
					filenamesr.erase(filenamesr.begin()+i,filenamesr.begin()+i+1);
			else
				i++;
		}

		while(filenamesl.size() < filenamesr.size())
			filenamesr.pop_back();

		while(filenamesl.size() > filenamesr.size())
			filenamesl.pop_back();

		if(filenamesl.empty())
		{
			perror("No corresponding images available");
			return -2;
		}
	}
	else
	{
		perror("Could not open directory");
		return -1;
	}

	return 0;
}


/* This function reads all images from a given directory and stores their names into a vector.
 *
 * string filepath				Input  -> Path to the directory
 * string fileprefl				Input  -> File prefix for the left or first images (last character must be "_" which is the
 *										  last character in the filename before the image number)
 * vector<string> filenamesl	Output -> Vector with sorted filenames of the images in the given directory
 *
 * Return value:				 0:		  Everything ok
 *								-1:		  Could not open directory
 *								-2:		  No images available
 */
int loadImageSequence(std::string filepath, std::string fileprefl, std::vector<std::string> & filenamesl)
{
	DIR *dir;
	struct dirent *ent;
	if((dir = opendir(filepath.c_str())) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
			string filename;
			filename = string(ent->d_name);
			if(filename.compare(0,fileprefl.size(),fileprefl) == 0)
				filenamesl.push_back(filename);
		}
		closedir(dir);

		if(filenamesl.empty())
		{
			perror("No images available");
			return -2;
		}

		sort(filenamesl.begin(),filenamesl.end(),
			 [](string const &first, string const &second){return atoi(first.substr(first.find_last_of("_")+1).c_str()) <
			 atoi(second.substr(second.find_last_of("_")+1).c_str());});
	}
	else
	{
		perror("Could not open directory");
		return -1;
	}

	return 0;
}


/* This function takes an 16Bit RGB integer image and converts it to a 3-channel float flow matrix where R specifies the 
 * flow in u, G the flow in v and B if the flow is valid.
 *
 * string filepath				Input  -> Path to the directory
 * string filename				Input  -> File name of the flow file with extension (e.g. *.png)
 * Mat* flow3					Output -> Pointer to the resulting 3-channel flow matrix (floats) where channel 1 specifies 
 *										  the flow in u, channel 2 the flow in v and channel 3 if the flow is valid (=1).
 * float precision				Input  -> Used precision in the given flow image file after the decimal point
 *										  (e.g. a precision of 64 yields a resolution of 1/64). [Default = 64]
 * bool useBoolValidity			Input  -> If true, it is asumed that the given validity in B is boolean (0 or 1) as e.g.
 *										  used within the KITTI database. Otherwise it can be a float number with
 *										  with precision validityPrecision (validity = B/validityPrecision). [Default = true]
 * float validityPrecision		Input  -> If useBoolValidity = false, this value specifies the precision of the
 *										  validity B (validity = B/validityPrecision). [Default = 64]
 * float minConfidence			Input  -> If useBoolValidity = false, minConfidence specifies the treshold used to decide
 *										  if a flow value is marked as valid or invalid (for validities between 0 and 1).
 *										  [Default = 1.0]
 *
 * Return value:				 0:		  Everything ok
 *								-1:		  Error reading flow file
 */
//int convertImageFlowFile(std::string filepath, std::string filename, cv::Mat* flow3, const float precision, 
//						 bool useBoolValidity, const float validityPrecision, const float minConfidence)
//{
//	Mat intflow;
//	PfePixImgStruct imgPfe = {0};
//	//intflow = imread(filepath + "\\" + filename,CV_LOAD_IMAGE_COLOR);
//	string pathfile = filepath + "\\" + filename;
//	PfeStatus stat;
//	PfeChar *pfepathfile = (PfeChar*)pathfile.c_str();
//	stat = PfeReadFileUIC(pfepathfile, &imgPfe, NULL, NULL);
//
//	if(stat != pfeOK)
//	{
//		perror("Error reading flow file");
//		return -1;
//	}
//	intflow = PfeConvToMat(&imgPfe);
//	if(intflow.data  == NULL)
//	{
//		perror("Error reading flow file");
//		return -1;
//	}
//	intflow = intflow.clone();
//	PfeFreeImgBuf(&imgPfe, 0);
//
//	//flow3->create(intflow.rows, intflow.cols, CV_32FC3);
//
//	vector<Mat> channels(3), channels_fin;
//	channels_fin.push_back(Mat(intflow.rows, intflow.cols, CV_32FC1));
//	channels_fin.push_back(Mat(intflow.rows, intflow.cols, CV_32FC1));
//	channels_fin.push_back(Mat(intflow.rows, intflow.cols, CV_32FC1));
//	cv::split(intflow, channels);
//	if(useBoolValidity)
//	{
//		for(size_t u = 0; u < intflow.rows; u++)
//		{
//			for( size_t v = 0; v < intflow.cols; v++)
//			{
//				if(channels[2].at<uint16_t>(u,v) > 0)
//				{
//					channels_fin[0].at<float>(u,v) = ((float)channels[0].at<uint16_t>(u,v) - 32768.0f) / precision;
//					channels_fin[1].at<float>(u,v) = ((float)channels[1].at<uint16_t>(u,v) - 32768.0f) / precision;
//					channels_fin[2].at<float>(u,v) = (float)channels[2].at<uint16_t>(u,v);
//				}
//				else
//				{
//					channels_fin[0].at<float>(u,v) = 0.0f;
//					channels_fin[1].at<float>(u,v) = 0.0f;
//					channels_fin[2].at<float>(u,v) = 0.0f;
//				}
//			}
//		}
//	}
//	else
//	{
//		for(size_t u = 0; u < intflow.rows; u++)
//		{
//			for( size_t v = 0; v < intflow.cols; v++)
//			{
//				if(channels[2].at<uint16_t>(u,v) > 0)
//				{
//					float conf = (float)channels[2].at<uint16_t>(u,v) / validityPrecision;
//					if(conf >= minConfidence)
//					{
//						channels_fin[0].at<float>(u,v) = ((float)channels[0].at<uint16_t>(u,v) - 32768.0f) / precision;
//						channels_fin[1].at<float>(u,v) = ((float)channels[1].at<uint16_t>(u,v) - 32768.0f) / precision;
//						channels_fin[2].at<float>(u,v) = 1.0f;
//					}
//					else
//					{
//						channels_fin[0].at<float>(u,v) = 0.0f;
//						channels_fin[1].at<float>(u,v) = 0.0f;
//						channels_fin[2].at<float>(u,v) = 0.0f;
//					}
//				}
//				else
//				{
//					channels_fin[0].at<float>(u,v) = 0.0f;
//					channels_fin[1].at<float>(u,v) = 0.0f;
//					channels_fin[2].at<float>(u,v) = 0.0f;
//				}
//			}
//		}
//	}
//
//	cv::merge(channels_fin,*flow3); 
//
//	return 0;
//}
//
///* This function takes an 16Bit 1-channel integer image (grey values) and converts it to a 3-channel (RGB) float flow matrix 
// * where R specifies the disparity, G is always 0 (as the disparity only represents the flow in x-direction) and B specifies 
// * if the flow/disparity is valid (0 or 1).
// *
// * string filepath				Input  -> Path to the directory
// * string filename				Input  -> File name of the disparity file with extension (e.g. *.png)
// * Mat* flow3					Output -> Pointer to the resulting 3-channel flow matrix (floats) where channel 1 specifies 
// *										  the the disparity, channel 2 is always 0 (as the disparity only represents the 
// *										  flow in x-direction) and channel 3 specifies if the disparity is valid (=1).
// * bool useFLowStyle			Input  -> If true [Default], the input file is expected to be a 3-channel 16bit file,
// *										  where the first channel includes the disparity values, the second channel is useless
// *										  and the third channel specifies if a disparity value is valid (valid >0, invalid 0)
// * float precision				Input  -> Used precision in the given disparity image file after the decimal point
// *										  (e.g. a precision of 64 yields a resolution of 1/64). [Default = 256]
// * bool use0Invalid				Input  -> If true, it is asumed that the given disparity is valid if the disparity is >0
// *										  (0 = invalid) as e.g. used within the KITTI database. Otherwise it is asumed that
// *										  invalid disparities have the value 0xFFFF. [Default = true]
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Error reading disparity file
// */
//int convertImageDisparityFile(std::string filepath, std::string filename, cv::Mat* flow3, const bool useFLowStyle, const float precision, const bool use0Invalid)
//{
//	Mat intflow;
//	PfePixImgStruct imgPfe = {0};
//	string pathfile = filepath + "\\" + filename;
//	PfeStatus stat;
//	PfeChar *pfepathfile = (PfeChar*)pathfile.c_str();
//	stat = PfeReadFileUIC(pfepathfile, &imgPfe, NULL, NULL);
//
//	if(stat != pfeOK)
//	{
//		perror("Error reading flow file");
//		return -1;
//	}
//	intflow = PfeConvToMat(&imgPfe);
//	if(intflow.data  == NULL)
//	{
//		perror("Error reading flow file");
//		return -1;
//	}
//	//intflow = intflow.rowRange(0,375).colRange(0,1242).clone();
//	intflow = intflow.clone();
//	PfeFreeImgBuf(&imgPfe, 0);
//
//
//	vector<Mat> channels(3), channels_fin;
//	channels_fin.push_back(Mat(intflow.rows, intflow.cols, CV_32FC1));
//	channels_fin.push_back(Mat(intflow.rows, intflow.cols, CV_32FC1));
//	channels_fin.push_back(Mat(intflow.rows, intflow.cols, CV_32FC1));
//	if(useFLowStyle)
//	{
//		cv::split(intflow, channels);
//		//namedWindow( "Channel 1", WINDOW_AUTOSIZE );// Create a window for display.
//		//imshow( "Channel 1", channels[0] );
//		//namedWindow( "Channel 2", WINDOW_AUTOSIZE );// Create a window for display.
//		//imshow( "Channel 2", channels[1] );
//		//namedWindow( "Channel 3", WINDOW_AUTOSIZE );// Create a window for display.
//		//imshow( "Channel 3", channels[2] );
//		//cv::waitKey(0);
//	}
//	if(intflow.data  == NULL)
//	{
//		perror("Error reading disparity file");
//		return -1;
//	}
//
//	if(useFLowStyle)
//	{
//		for(size_t u = 0; u < intflow.rows; u++)
//		{
//			for( size_t v = 0; v < intflow.cols; v++)
//			{
//				if(channels[2].at<uint16_t>(u,v) > 0)
//				{
//					channels_fin[0].at<float>(u,v) = -1.0f * (float)channels[0].at<uint16_t>(u,v) / precision;
//					channels_fin[1].at<float>(u,v) = 0.0f;
//					channels_fin[2].at<float>(u,v) = (float)channels[2].at<uint16_t>(u,v);
//				}
//				else
//				{
//					channels_fin[0].at<float>(u,v) = 0.0f;
//					channels_fin[1].at<float>(u,v) = 0.0f;
//					channels_fin[2].at<float>(u,v) = 0.0f;
//				}
//			}
//		}
//	}
//	else
//	{
//		if(use0Invalid)
//		{
//			for(size_t u = 0; u < intflow.rows; u++)
//			{
//				for( size_t v = 0; v < intflow.cols; v++)
//				{
//					if(intflow.at<uint16_t>(u,v) > 0)
//					{
//						channels_fin[0].at<float>(u,v) = -1.0f * (float)intflow.at<uint16_t>(u,v) / precision;
//						channels_fin[1].at<float>(u,v) = 0.0f;
//						channels_fin[2].at<float>(u,v) = 1.0f;
//					}
//					else
//					{
//						channels_fin[0].at<float>(u,v) = 0.0f;
//						channels_fin[1].at<float>(u,v) = 0.0f;
//						channels_fin[2].at<float>(u,v) = 0.0f;
//					}
//				}
//			}
//		}
//		else
//		{
//			for(size_t u = 0; u < intflow.rows; u++)
//			{
//				for( size_t v = 0; v < intflow.cols; v++)
//				{
//					if(intflow.at<uint16_t>(u,v) == 0xFFFF)
//					{
//						channels_fin[0].at<float>(u,v) = 0.0f;
//						channels_fin[1].at<float>(u,v) = 0.0f;
//						channels_fin[2].at<float>(u,v) = 0.0f;
//					}
//					else
//					{
//						channels_fin[0].at<float>(u,v) = -1.0f * (float)intflow.at<uint16_t>(u,v) / precision;
//						channels_fin[1].at<float>(u,v) = 0.0f;
//						channels_fin[2].at<float>(u,v) = 1.0f;
//					}
//				}
//			}
//		}
//	}
//
//	cv::merge(channels_fin,*flow3);
//
//	return 0;
//}
//
///* This function reads all homography file names from a given directory and stores their names into a vector.
// *
// * string filepath				Input  -> Path to the directory
// * string fileprefl				Input  -> File prefix for the left or first images (for the dataset of 
// *										  www.robots.ox.ac.uk/~vgg/research/affine/ this must be H1to because
// *										  the file names look like H1to2, H1to3, ...)
// * vector<string> filenamesl	Output -> Vector with sorted filenames of the images in the given directory
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Could not open directory
// *								-2:		  No homography files available
// */
//int readHomographyFiles(std::string filepath, std::string fileprefl, std::vector<std::string> & filenamesl)
//{
//	DIR *dir;
//	struct dirent *ent;
//	if((dir = opendir(filepath.c_str())) != NULL)
//	{
//		while ((ent = readdir(dir)) != NULL)
//		{
//			string filename;
//			filename = string(ent->d_name);
//			if(filename.compare(0,fileprefl.size(),fileprefl) == 0)
//				filenamesl.push_back(filename);
//		}
//		closedir(dir);
//
//		if(filenamesl.empty())
//		{
//			perror("No homography files available");
//			return -2;
//		}
//
//		sort(filenamesl.begin(),filenamesl.end(),
//			 [](string const &first, string const &second){return atoi(first.substr(first.find_last_of("o")+1).c_str()) <
//			 atoi(second.substr(second.find_last_of("o")+1).c_str());});
//	}
//	else
//	{
//		perror("Could not open directory");
//		return -1;
//	}
//
//	return 0;
//}
//
///* This function reads a homography from a given file.
// *
// * string filepath				Input  -> Path to the directory
// * string filename				Input  -> Filename of a stored homography (from www.robots.ox.ac.uk/~vgg/research/affine/ )
// * Mat* H						Output -> Pointer to the homography
// *
// * Return value:				 0:		  Everything ok
// *								-1:		  Reading homography failed
// */
//int readHomographyFromFile(std::string filepath, std::string filename, cv::Mat* H)
//{
//	ifstream ifs;
//	char stringline[100];
//	char* pEnd;
//	H->create(3,3,CV_64FC1);
//	size_t i = 0, j;
//	ifs.open(filepath + "\\" + filename, ifstream::in);
//
//	while(ifs.getline(stringline,100) && (i < 3))
//	{
//		H->at<double>(i,0) = strtod(stringline, &pEnd);
//		for(j = 1; j < 3; j++)
//		{
//			H->at<double>(i,j) = strtod(pEnd, &pEnd);
//		}
//		i++;
//	}
//	ifs.close();
//
//	if((i < 3) || (j < 3))
//		return -1; //Reading homography failed
//
//	return 0;
//}
