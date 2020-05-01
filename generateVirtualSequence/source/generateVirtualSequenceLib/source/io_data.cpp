/**********************************************************************************************************
 FILE: io_data.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: September 2015

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file holds functionalities for loading image data, flow data from the KITTI dataset,
 flow data from O. Zendel's generated virtuel images and homographys from files of the test data
 from K. Mikolajczyk. Moreover, functions are provided to generate the different quality measurements
 for the matching algorithms.
**********************************************************************************************************/

//#include "..\include\glob_includes.h"
#include "io_data.h"
//#include "atlstr.h"
#include <stdint.h>
#include <fstream>
#include "dirent.h"
#include <algorithm>
#include <functional>

#define OLDVERSIONCHECKDIR 0
#if OLDVERSIONCHECKDIR
#include <sys/types.h>
#include <sys/stat.h>
#include <cstring>
#include <string>
#endif

#include <boost/filesystem.hpp>
#include "alphanum.hpp"

//#include "PfeImgFileIO.h"
//#include "PfeConv.h"

using namespace std;
using namespace cv;

/* --------------------- Function prototypes --------------------- */

int makeFrameIdConsistent1(std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr, std::size_t prefxPos1, std::size_t prefxPos2, bool bVerbose = false);

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
	if((dir = opendir(filepath.c_str())) != nullptr)
	{
		while ((ent = readdir(dir)) != nullptr)
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
			 [](string const &first, string const &second){return stoi(first.substr(first.find_last_of('_')+1)) <
			 stoi(second.substr(second.find_last_of('_')+1));});

		sort(filenamesr.begin(),filenamesr.end(),
			 [](string const &first, string const &second){return stoi(first.substr(first.find_last_of('_')+1)) <
			 stoi(second.substr(second.find_last_of('_')+1));});

		size_t i = 0;
		while((i < filenamesr.size()) && (i < filenamesl.size()))
		{
			if(stoi(filenamesl[i].substr(filenamesl[i].find_last_of('_')+1)) <
			   stoi(filenamesr[i].substr(filenamesr[i].find_last_of('_')+1)))
			   filenamesl.erase(filenamesl.begin()+i,filenamesl.begin()+i+1);
			else if(stoi(filenamesl[i].substr(filenamesl[i].find_last_of('_')+1)) >
					stoi(filenamesr[i].substr(filenamesr[i].find_last_of('_')+1)))
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
	if((dir = opendir(filepath.c_str())) != nullptr)
	{
		while ((ent = readdir(dir)) != nullptr)
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
			 [](string const &first, string const &second){return stoi(first.substr(first.find_last_of('_')+1)) <
			 stoi(second.substr(second.find_last_of('_')+1));});
	}
	else
	{
		perror("Could not open directory");
		return -1;
	}

	return 0;
}

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
int loadStereoSequenceNew(std::string filepath, std::string fileprefl, std::string fileprefr,
	std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr)
{
	DIR *dir;
	struct dirent *ent;
	std::string l_frameUsed;
	size_t prefxPos1 = 0, prefxPos2 = 0;
	bool bInputIdent = false;
	if (filepath.find('\\') != std::string::npos)
		std::replace(filepath.begin(), filepath.end(), '\\', '/');
	if (fileprefl.find('\\') != std::string::npos)
		std::replace(fileprefl.begin(), fileprefl.end(), '\\', '/');
	if (fileprefr.find('\\') != std::string::npos)
		std::replace(fileprefr.begin(), fileprefr.end(), '\\', '/');
	if (filepath.rfind('/') == filepath.size() - 1)
		filepath = filepath.substr(0, filepath.size() - 1);
	for (int i = 0; i<2; ++i)
	{
		std::string fileprefl_use = (i == 0) ? fileprefl : fileprefr, filedir_use = filepath;
		std::string filepostfx;
		int posLastSl = (int)fileprefl_use.rfind('/');
		if (posLastSl >= 0)
		{
			if (fileprefl_use.find('/') == 0)
				filedir_use += fileprefl_use.substr(0, (size_t)posLastSl);
			else
				filedir_use += "/" + fileprefl_use.substr(0, (size_t)posLastSl);
			fileprefl_use = fileprefl_use.substr((size_t)posLastSl + 1);
		}
		std::string cmp_is_ident = filedir_use + "/" + fileprefl_use;
		if (i == 0)
			l_frameUsed = cmp_is_ident;
		else if ((cmp_is_ident == l_frameUsed) && filenamesl.size() > 1)
		{
			bInputIdent = true;
			break;
		}


		int nFuzzyPos = (int)fileprefl_use.find('*');
		bool bCmpFuzzy = (nFuzzyPos >= 0);
		if (bCmpFuzzy)
		{
			//fileprefl_use = fileprefl_use.substr(0, nFuzzyPos) + fileprefl_use.substr(nFuzzyPos + 1);
			std::string fileprefl_use_tmp = fileprefl_use;
			fileprefl_use = fileprefl_use.substr(0, (size_t)nFuzzyPos);
			filepostfx = fileprefl_use_tmp.substr((size_t)nFuzzyPos + 1);
		}
		if ((dir = opendir(filedir_use.c_str())) != nullptr)
		{
			while ((ent = readdir(dir)) != nullptr)
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
					|| (!fileprefl_use.empty() && filename.compare(0, fileprefl_use.size(), fileprefl_use) == 0)) {
                    if (i == 0) {
                        filenamesl.push_back(filedir_use + "/" + filename);
                    } else {
                        filenamesr.push_back(filedir_use + "/" + filename);
                    }
                }
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
			for (int j = 0; j < 10; j++)
			{
				nrpos = filenamesl.back().find_first_of(std::to_string(j), prefxPos1);

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
			for (int j = 0; j < 10; j++)
			{
				nrpos = filenamesr.back().find_first_of(std::to_string(j), prefxPos2);

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
				std::bind([](string const &first, string const &second, std::size_t prefxPos)
				{return stoi(first.substr(prefxPos)) <
					stoi(second.substr(prefxPos)); }, _1, _2, prefxPos1));

			sort(filenamesr.begin(), filenamesr.end(),
				std::bind([](string const &first, string const &second, std::size_t prefxPos)
				{return stoi(first.substr(prefxPos)) <
					stoi(second.substr(prefxPos)); }, _1, _2, prefxPos2));

			makeFrameIdConsistent1(filenamesl, filenamesr, prefxPos1, prefxPos2);
		}

		if (filenamesl.empty())
		{
			perror("No corresponding images available");
			return -2;
		}
	}

	return 0;
}

int makeFrameIdConsistent1(std::vector<std::string> & filenamesl, std::vector<std::string> & filenamesr, std::size_t prefxPos1, std::size_t prefxPos2, bool bVerbose)
{
	int num_rem = 0;
	size_t i = 0;
	while ((i < filenamesr.size()) && (i < filenamesl.size()))
	{
		if (stoi(filenamesl[i].substr(prefxPos1)) <
			stoi(filenamesr[i].substr(prefxPos2)))
		{
			if (bVerbose)
				cout << "Warning: removing inconsistent frame " << filenamesl[i] << " vs. " << filenamesr[i] << endl;
			num_rem++;
			filenamesl.erase(filenamesl.begin() + i, filenamesl.begin() + i + 1);
		}
		else if (stoi(filenamesl[i].substr(prefxPos1)) >
			stoi(filenamesr[i].substr(prefxPos2)))
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
int loadImageSequenceNew(std::string filepath, std::string fileprefl, std::vector<std::string> & filenamesl)
{
	DIR *dir;
	struct dirent *ent;
	std::string fileprefl_use = std::move(fileprefl), filedir_use = move(filepath);
	std::string filepostfx;
	if (filedir_use.find('\\') != std::string::npos)
		std::replace(filedir_use.begin(), filedir_use.end(), '\\', '/');
	if (fileprefl_use.find('\\') != std::string::npos)
		std::replace(fileprefl_use.begin(), fileprefl_use.end(), '\\', '/');
	if (filedir_use.rfind('/') == filedir_use.size() - 1)
		filedir_use = filedir_use.substr(0, filedir_use.size() - 1);
	int posLastSl = (int)fileprefl_use.rfind('/');
	if (posLastSl >= 0)
	{
		if (fileprefl_use.find('/') == 0)
			filedir_use += fileprefl_use.substr(0, posLastSl);
		else
			filedir_use += "/" + fileprefl_use.substr(0, posLastSl);
		fileprefl_use = fileprefl_use.substr(posLastSl + 1);
	}
	int nFuzzyPos = (int)fileprefl_use.find('*');
	bool bCmpFuzzy = (nFuzzyPos >= 0);
	if (bCmpFuzzy)
	{
		//fileprefl_use = fileprefl_use.substr(0, nFuzzyPos) + fileprefl_use.substr(nFuzzyPos + 1);
		std::string fileprefl_use_tmp = fileprefl_use;
		fileprefl_use = fileprefl_use.substr(0, (size_t)nFuzzyPos);
		filepostfx = fileprefl_use_tmp.substr((size_t)nFuzzyPos + 1);
	}
	if ((dir = opendir(filedir_use.c_str())) != nullptr)
	{
		while ((ent = readdir(dir)) != nullptr)
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
				|| (!fileprefl_use.empty() && filename.compare(0, fileprefl_use.size(), fileprefl_use) == 0)) {
				if(IsImgTypeSupported(filename)) {
					filenamesl.push_back(filedir_use + "/" + filename);
				}
			}
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

bool IsImgTypeSupported(std::string const& filename){
	std::vector<std::string> vecSupportedTypes = GetSupportedImgTypes();

	size_t ptPos = filename.rfind('.');
	if(ptPos == string::npos){
		return false;
	}
	std::string ending = filename.substr(ptPos + 1);

	std::transform(ending.begin(), ending.end(), ending.begin(), ::tolower);

	if(std::find(vecSupportedTypes.begin(), vecSupportedTypes.end(), ending) != vecSupportedTypes.end())
	{
		return true;
	}

	return false;
}

std::vector<std::string> GetSupportedImgTypes(){
	int const nrSupportedTypes = 20;
	static std::string types [] = {"bmp",
								   "dib",
								   "jpeg",
								   "jpg",
								   "jpe",
								   "jp2",
								   "png",
								   "webp",
								   "pbm",
								   "pgm",
								   "ppm",
								   "pxm",
								   "pnm",
								   "sr",
								   "ras",
								   "tiff",
								   "tif",
								   "exr",
								   "hdr",
								   "pic"
	};
	return std::vector<std::string>(types, types + nrSupportedTypes);
}

#if OLDVERSIONCHECKDIR
bool checkPathExists(const std::string &path){
    struct stat info;
    if( stat( path.c_str() , &info ) != 0 )//Cannot access directory
        return false;
    else if( info.st_mode & S_IFDIR )  //It is a directory
        return true;

    return false;
}
#else
bool checkPathExists(const std::string &path){
	boost::filesystem::path path2 = path;
	return boost::filesystem::is_directory(path2);
}
#endif

bool checkFileExists(const std::string &filename){
	return boost::filesystem::exists(filename);
}

bool createDirectory(const std::string &path){
	return boost::filesystem::create_directories(path);
}

bool deleteFile(const std::string &filename){
	return boost::filesystem::remove(filename);
}

std::string concatPath(const std::string &mainPath, const std::string &subPath){
	std::string filedir1 = mainPath;
	std::string filedir2 = subPath;
	if (filedir1.find('\\') != std::string::npos)
		std::replace(filedir1.begin(), filedir1.end(), '\\', '/');
	if (filedir2.find('\\') != std::string::npos)
		std::replace(filedir2.begin(), filedir2.end(), '\\', '/');
	if (filedir1.rfind('/') == filedir1.size() - 1)
		filedir1 = filedir1.substr(0, filedir1.size() - 1);
	if (filedir2.find('/') == 0)
		filedir2 = filedir2.substr(1, filedir1.size());

	return filedir1 + '/' + filedir2;
}

/* This function reads all homography file names from a given directory and stores their names into a vector.
 *
 * string filepath				Input  -> Path to the directory
 * string fileprefl				Input  -> File prefix for the left or first images (for the dataset of
 *										  www.robots.ox.ac.uk/~vgg/research/affine/ this must be H1to because
 *										  the file names look like H1to2, H1to3, ...)
 * vector<string> filenamesl	Output -> Vector with sorted filenames of the images in the given directory
 */
bool readHomographyFiles(const std::string& filepath, const std::string& fileprefl, std::vector<std::string> & filenamesl)
{
    DIR *dir;
    struct dirent *ent;
    if((dir = opendir(filepath.c_str())) != nullptr)
    {
        while ((ent = readdir(dir)) != nullptr)
        {
            string filename;
            filename = string(ent->d_name);
            if(filename.compare(0,fileprefl.size(),fileprefl) == 0)
                filenamesl.push_back(filename);
        }
        closedir(dir);

        if(filenamesl.empty())
        {
            std::cerr << "No homography files available" << endl;
            return false;
        }

        sort(filenamesl.begin(),filenamesl.end(),
             [](string const &first, string const &second){return stoi(first.substr(first.find_last_of('o')+1)) <
                     stoi(second.substr(second.find_last_of('o')+1));});
    }
    else
    {
        std::cerr << "Could not open directory" << endl;
        return false;
    }

    return true;
}

/* This function reads a homography from a given file.
 *
 * string filepath				Input  -> Path to the directory
 * string filename				Input  -> Filename of a stored homography (from www.robots.ox.ac.uk/~vgg/research/affine/ )
 * Mat* H						Output -> Pointer to the homography
 */
bool readHomographyFromFile(const std::string& filepath, const std::string& filename, cv::OutputArray H)
{
    ifstream ifs;
    char stringline[100];
    char* pEnd;
    Mat H_ = Mat(3,3,CV_64FC1);
    size_t i = 0, j;
    ifs.open(filepath + "\\" + filename, ifstream::in);

    while(ifs.getline(stringline,100) && (i < 3))
    {
        H_.at<double>(i,0) = strtod(stringline, &pEnd);
        for(j = 1; j < 3; j++)
        {
            H_.at<double>(i,j) = strtod(pEnd, &pEnd);
        }
        i++;
    }
    ifs.close();

    if((i < 3) || (j < 3))
        return false; //Reading homography failed
    if (H.needed())
    {
        if (H.empty())
            H.create(3, 3, CV_64F);
        H_.copyTo(H.getMat());
    }else{
        return false;
    }

    return true;
}