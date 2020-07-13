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

//#include "..\include\glob_includes.h"
#include "io_data.h"
#include <stdint.h>
#include <fstream>
#ifdef __linux__
#include "sys/dir.h"
#else
#include "atlstr.h"
#include "dirent.h"
#endif
#include <algorithm>
#include <functional>
#include "alphanum.hpp"

//#include "PfeImgFileIO.h"
//#include "PfeConv.h"


using namespace std;
//using namespace cv;

/* --------------------- Function prototypes --------------------- */

int makeFrameIdConsistent(std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr, std::size_t prefxPos1, std::size_t prefxPos2, bool bVerbose = false);

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
//int loadStereoSequence(std::string filepath, std::string fileprefl, std::string fileprefr,
//					   std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr)
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
//			else if(filename.compare(0,fileprefr.size(),fileprefr) == 0)
//				filenamesr.push_back(filename);
//		}
//		closedir(dir);

//		if(filenamesl.empty())
//		{
//			perror("No left images available");
//			return -3;
//		}

//		if(filenamesr.empty())
//		{
//			perror("No right images available");
//			return -3;
//		}

//		sort(filenamesl.begin(),filenamesl.end(),
//			 [](string const &first, string const &second){return atoi(first.substr(first.find_last_of("_")+1).c_str()) <
//			 atoi(second.substr(second.find_last_of("_")+1).c_str());});

//		sort(filenamesr.begin(),filenamesr.end(),
//			 [](string const &first, string const &second){return atoi(first.substr(first.find_last_of("_")+1).c_str()) <
//			 atoi(second.substr(second.find_last_of("_")+1).c_str());});

//		size_t i = 0;
//		while((i < filenamesr.size()) && (i < filenamesl.size()))
//		{
//			if(atoi(filenamesl[i].substr(filenamesl[i].find_last_of("_")+1).c_str()) <
//			   atoi(filenamesr[i].substr(filenamesr[i].find_last_of("_")+1).c_str()))
//			   filenamesl.erase(filenamesl.begin()+i,filenamesl.begin()+i+1);
//			else if(atoi(filenamesl[i].substr(filenamesl[i].find_last_of("_")+1).c_str()) >
//					atoi(filenamesr[i].substr(filenamesr[i].find_last_of("_")+1).c_str()))
//					filenamesr.erase(filenamesr.begin()+i,filenamesr.begin()+i+1);
//			else
//				i++;
//		}

//		while(filenamesl.size() < filenamesr.size())
//			filenamesr.pop_back();

//		while(filenamesl.size() > filenamesr.size())
//			filenamesl.pop_back();

//		if(filenamesl.empty())
//		{
//			perror("No corresponding images available");
//			return -2;
//		}
//	}
//	else
//	{
//		perror("Could not open directory");
//		return -1;
//	}

//	return 0;
//}


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
//int loadImageSequence(std::string filepath, std::string fileprefl, std::vector<std::string> & filenamesl)
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

//		if(filenamesl.empty())
//		{
//			perror("No images available");
//			return -2;
//		}

//		sort(filenamesl.begin(),filenamesl.end(),
//			 [](string const &first, string const &second){return atoi(first.substr(first.find_last_of("_")+1).c_str()) <
//			 atoi(second.substr(second.find_last_of("_")+1).c_str());});
//	}
//	else
//	{
//		perror("Could not open directory");
//		return -1;
//	}

//	return 0;
//}

/* This function reads all stereo or 2 subsequent images from a given directory and stores their names into two vectors.
*
* string filepath				Input  -> Path to the directory
* string fileprefl				Input  -> Prefix for the left or first images. It can include a folder structure that follows after the
*										  filepath, a file prefix, a '*' indicating the position of the number and a postfix. If it is empty,
*										  all files from the folder filepath are used (also if fileprefl only contains a folder ending with '/',
*										  every file within this folder is used). It is possible to specify only a prefix with or
*										  without '*' at the end. If a prefix is used, all characters until the first number (excluding) must
*										  be provided. For a postfix, '*' must be placed before the postfix.
*										  Valid examples: folder/pre_*post, *post, pre_*, pre_, folder/*post, folder/pre_*, folder/pre_, folder/,
*										  folder/folder/, folder/folder/pre_*post, ...
*										  For non stereo images (consecutive images), fileprefl must be equal to fileprefr.
* string fileprefr				Input  -> Prefix for the right or second images. It can include a folder structure that follows after the
*										  filepath, a file prefix, a '*' indicating the position of the number and a postfix. If it is empty,
*										  all files from the folder filepath are used (also if fileprefl only contains a folder ending with '/',
*										  every file within this folder is used). It is possible to specify only a prefix with or
*										  without '*' at the end. If a prefix is used, all characters until the first number (excluding) must
*										  be provided. For a postfix, '*' must be placed before the postfix.
*										  Valid examples: folder/pre_*post, *post, pre_*, pre_, folder/*post, folder/pre_*, folder/pre_, folder/,
*										  folder/folder/, folder/folder/pre_*post, ...
*										  For non stereo images (consecutive images), fileprefl must be equal to fileprefr.
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
    std::string l_frameUsed = "";
    size_t prefxPos1 = 0, prefxPos2 = 0;
    bool bInputIdent = false;
    if (filepath.find("\\") != std::string::npos)
        std::replace(filepath.begin(), filepath.end(), '\\', '/');
    if (fileprefl.find("\\") != std::string::npos)
        std::replace(fileprefl.begin(), fileprefl.end(), '\\', '/');
    if (fileprefr.find("\\") != std::string::npos)
        std::replace(fileprefr.begin(), fileprefr.end(), '\\', '/');
    if (filepath.rfind("/") == filepath.size() - 1)
        filepath = filepath.substr(0,filepath.size() - 1);
    for (int i = 0; i<2; ++i)
    {
        std::string fileprefl_use = (i == 0) ? fileprefl : fileprefr, filedir_use = filepath;
        std::string filepostfx;
        int posLastSl = (int)fileprefl_use.rfind("/");
        if (posLastSl >= 0)
        {
            if(fileprefl_use.find("/") == 0)
                filedir_use += fileprefl_use.substr(0, posLastSl);
            else
                filedir_use += "/" + fileprefl_use.substr(0, posLastSl);
            fileprefl_use = fileprefl_use.substr(posLastSl + 1);
        }
        std::string cmp_is_ident = filedir_use + "/" + fileprefl_use;
        if (i == 0)
            l_frameUsed = cmp_is_ident;
        else if ((cmp_is_ident == l_frameUsed) && filenamesl.size() > 1)
        {
            bInputIdent = true;
            break;
        }


        int nFuzzyPos = (int)fileprefl_use.find("*");
        bool bCmpFuzzy = (nFuzzyPos >= 0);
        if (bCmpFuzzy)
        {
            //fileprefl_use = fileprefl_use.substr(0, nFuzzyPos) + fileprefl_use.substr(nFuzzyPos + 1);
            std::string fileprefl_use_tmp = fileprefl_use;
            fileprefl_use = fileprefl_use.substr(0, nFuzzyPos);
            filepostfx = fileprefl_use_tmp.substr(nFuzzyPos + 1);
        }
        if ((dir = opendir(filedir_use.c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                string filename;
                filename = string(ent->d_name);
                if ((fileprefl_use.empty() && filepostfx.empty() && filename.size() > 2 && filename.find(".db") == std::string::npos)
                    || (bCmpFuzzy && !fileprefl_use.empty() && !filepostfx.empty()
                        && (filename.size() >= fileprefl_use.size()) && (filename.find(fileprefl_use) != std::string::npos)
                        && (filename.size() >= filepostfx.size()) && (filename.find(filepostfx) != std::string::npos))
                    || (bCmpFuzzy && !fileprefl_use.empty() && filepostfx.empty()
                        && (filename.size() >= fileprefl_use.size()) && filename.find(fileprefl_use) != std::string::npos)
                    || (bCmpFuzzy && fileprefl_use.empty() && !filepostfx.empty()
                        && (filename.size() >= filepostfx.size()) && filename.find(filepostfx) != std::string::npos)
                    || (!fileprefl_use.empty() && filename.compare(0, fileprefl_use.size(), fileprefl_use) == 0))
                    if (i == 0)
                        filenamesl.push_back(filedir_use + "/" + filename);
                    else
                        filenamesr.push_back(filedir_use + "/" + filename);
            }
            closedir(dir);
        }
        else
        {
            perror("Could not open directory");
            return -1;
        }

        if (!fileprefl_use.empty() && i == 0)
        {
            prefxPos1 = filenamesl.back().rfind(fileprefl_use);
            prefxPos1 += fileprefl_use.size();
        }
        else if (i == 0)
        {
            prefxPos1 = filedir_use.size() + 1;
            size_t nrpos = std::string::npos, nrpos1 = prefxPos1;
            bool firstchar = false;
            for (int i = 0; i < 10; i++)
            {
                nrpos = filenamesl.back().find_first_of(std::to_string(i), prefxPos1);

                if (!firstchar && nrpos != std::string::npos)
                    firstchar = true;
                if (firstchar && nrpos == prefxPos1)
                    break;

                if (nrpos != std::string::npos && ((nrpos < nrpos1) || ((nrpos >= prefxPos1) && (nrpos1 == prefxPos1))))
                    nrpos1 = nrpos;
            }
            prefxPos1 = nrpos1;
        }

        if (!fileprefl_use.empty() && i == 1)
        {
            prefxPos2 = filenamesr.back().rfind(fileprefl_use);
            prefxPos2 += fileprefl_use.size();
        }
        else if (i == 1)
        {
            prefxPos2 = filedir_use.size() + 1;
            size_t nrpos = std::string::npos, nrpos1 = prefxPos2;
            bool firstchar = false;
            for (int i = 0; i < 10; i++)
            {
                nrpos = filenamesr.back().find_first_of(std::to_string(i), prefxPos2);

                if (!firstchar && nrpos != std::string::npos)
                    firstchar = true;
                if (firstchar && nrpos == prefxPos2)
                    break;

                if (nrpos != std::string::npos && ((nrpos < nrpos1) || ((nrpos >= prefxPos2) && (nrpos1 == prefxPos2))))
                    nrpos1 = nrpos;
            }
            prefxPos2 = nrpos1;
        }
    }

    {
        if (filenamesl.empty())
        {
            perror("No left images available");
            return -3;
        }

        if (bInputIdent)
        {
            sort(filenamesl.begin(), filenamesl.end(), doj::alphanum_less<std::string>());
            filenamesr = filenamesl;   //r==l (remove first/last frame)
            filenamesl.pop_back();
            filenamesr.erase(filenamesr.begin());
        }
        else
        {
            if (filenamesr.empty())
            {
                perror("No right images available");
                return -3;
            }

            using namespace std::placeholders;
            sort(filenamesl.begin(), filenamesl.end(),
                std::bind([](string const &first, string const &second, std::size_t prefxPos) {return atoi(first.substr(prefxPos).c_str()) <
                atoi(second.substr(prefxPos).c_str()); }, _1, _2, prefxPos1));

            sort(filenamesr.begin(), filenamesr.end(),
                std::bind([](string const &first, string const &second, std::size_t prefxPos) {return atoi(first.substr(prefxPos).c_str()) <
                atoi(second.substr(prefxPos).c_str()); }, _1, _2, prefxPos2));

            makeFrameIdConsistent(filenamesl, filenamesr, prefxPos1, prefxPos2);
        }

        if (filenamesl.empty())
        {
            perror("No corresponding images available");
            return -2;
        }
    }

    return 0;
}

int makeFrameIdConsistent(std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr, std::size_t prefxPos1, std::size_t prefxPos2, bool bVerbose)
{
    int num_rem = 0;
    size_t i = 0;
    while ((i < filenamesr.size()) && (i < filenamesl.size()))
    {
        if (atoi(filenamesl[i].substr(prefxPos1).c_str()) <
            atoi(filenamesr[i].substr(prefxPos2).c_str()))
        {
            if (bVerbose)
                cout << "Warning: removing inconsistent frame " << filenamesl[i] << " vs. " << filenamesr[i] << endl;
            num_rem++;
            filenamesl.erase(filenamesl.begin() + i, filenamesl.begin() + i + 1);
        }
        else if (atoi(filenamesl[i].substr(prefxPos1).c_str()) >
            atoi(filenamesr[i].substr(prefxPos2).c_str()))
        {
            num_rem++;
            filenamesr.erase(filenamesr.begin() + i, filenamesr.begin() + i + 1);
        }
        else
            i++;
    }

    while (filenamesl.size() < filenamesr.size())
    {
        num_rem++;
        filenamesr.pop_back();
    }

    while (filenamesl.size() > filenamesr.size())
    {
        if (bVerbose)
            cout << "Warning: removing inconsistent frame " << filenamesl[filenamesl.size() - 1] << " at end (numbers mismatch)" << endl;
        num_rem++;
        filenamesl.pop_back();
    }

    return num_rem;
}

/* This function reads all images from a given directory and stores their names into a vector.
*
* string filepath				Input  -> Path to the directory
* string fileprefl				Input  -> Prefix images. It can include a folder structure that follows after the
*										  filepath, a file prefix, a '*' indicating the position of the number and a postfix. If it is empty,
*										  all files from the folder filepath are used (also if fileprefl only contains a folder ending with '/',
*										  every file within this folder is used). It is possible to specify only a prefix with or
*										  without '*' at the end. If a prefix is used, all characters until the first number (excluding) must
*										  be provided. For a postfix, '*' must be placed before the postfix.
*										  Valid examples: folder/pre_*post, *post, pre_*, pre_, folder/*post, folder/pre_*, folder/pre_, folder/,
*										  folder/folder/, folder/folder/pre_*post, ...
* vector<string> filenamesl		Output -> Vector with sorted filenames of the images in the given directory
*
* Return value:					 0:		  Everything ok
*								-1:		  Could not open directory
*								-2:		  No images available
*/
int loadImageSequence(std::string filepath, std::string fileprefl, std::vector<std::string> & filenamesl)
{
    DIR *dir;
    struct dirent *ent;
    std::string fileprefl_use = fileprefl, filedir_use = filepath;
    std::string filepostfx;
    if (filedir_use.find("\\") != std::string::npos)
        std::replace(filedir_use.begin(), filedir_use.end(), '\\', '/');
    if (fileprefl_use.find("\\") != std::string::npos)
        std::replace(fileprefl_use.begin(), fileprefl_use.end(), '\\', '/');
    if (filedir_use.rfind("/") == filedir_use.size() - 1)
        filedir_use = filedir_use.substr(0, filedir_use.size() - 1);
    int posLastSl = (int)fileprefl_use.rfind("/");
    if (posLastSl >= 0)
    {
        if (fileprefl_use.find("/") == 0)
            filedir_use += fileprefl_use.substr(0, posLastSl);
        else
            filedir_use += "/" + fileprefl_use.substr(0, posLastSl);
        fileprefl_use = fileprefl_use.substr(posLastSl + 1);
    }
    int nFuzzyPos = (int)fileprefl_use.find("*");
    bool bCmpFuzzy = (nFuzzyPos >= 0);
    if (bCmpFuzzy)
    {
        //fileprefl_use = fileprefl_use.substr(0, nFuzzyPos) + fileprefl_use.substr(nFuzzyPos + 1);
        std::string fileprefl_use_tmp = fileprefl_use;
        fileprefl_use = fileprefl_use.substr(0, nFuzzyPos);
        filepostfx = fileprefl_use_tmp.substr(nFuzzyPos + 1);
    }
    if ((dir = opendir(filedir_use.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            string filename;
            filename = string(ent->d_name);
            if ((fileprefl_use.empty() && filepostfx.empty() && filename.size() > 2 && filename.find(".db") == std::string::npos)
                || (bCmpFuzzy && !fileprefl_use.empty() && !filepostfx.empty()
                    && (filename.size() >= fileprefl_use.size()) && (filename.find(fileprefl_use) != std::string::npos)
                    && (filename.size() >= filepostfx.size()) && (filename.find(filepostfx) != std::string::npos))
                || (bCmpFuzzy && !fileprefl_use.empty() && filepostfx.empty()
                    && (filename.size() >= fileprefl_use.size()) && filename.find(fileprefl_use) != std::string::npos)
                || (bCmpFuzzy && fileprefl_use.empty() && !filepostfx.empty()
                    && (filename.size() >= filepostfx.size()) && filename.find(filepostfx) != std::string::npos)
                || (!fileprefl_use.empty() && filename.compare(0, fileprefl_use.size(), fileprefl_use) == 0))
                filenamesl.push_back(filedir_use + "/" + filename);
        }
        closedir(dir);
        std::sort(filenamesl.begin(), filenamesl.end(), doj::alphanum_less<std::string>());
    }
    else
    {
        perror("Could not open directory");
        return -1;
    }

    if (filenamesl.empty())
        return -2;

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
