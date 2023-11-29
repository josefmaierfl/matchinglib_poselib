//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2021 Josef Maier
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

#include <string>
#include <vector>
#include <fstream>
#include <opencv2/highgui.hpp>

#include "utilslib/utilslib_api.h"

namespace utilslib
{
  class UTILSLIB_API FileHelper
  {
  public:
    static bool fileExists(const std::string& path);
    static bool directoryExists(const std::string& directory);
    static bool deleteFile(const std::string &path);
    static bool deleteDirectory(const std::string &path);

    static void ensureFileExists(const std::string& path);
    static void ensureDirectoryExists(const std::string& directory);

    static bool filePathEndsWith(const std::string& path, const std::string& ending);
    static std::string joinPaths(const std::string& path1, const std::string& path2);

    static std::string getBaseName(const std::string& path);
    static std::string getDirName(const std::string &path);

    static std::vector<std::string> getDirectories(const std::string& path);
    static std::vector<std::string> getFiles(const std::string& path);

    static void matchToBinary(std::ofstream &resultsToFile, const cv::DMatch &m);
    static void keypointToBinary(std::ofstream &resultsToFile, const cv::KeyPoint &kp);
    static void keypointsToBinary(std::ofstream &resultsToFile, const std::vector<cv::KeyPoint> &kps);
    static void cvMatToBinary(std::ofstream &resultsToFile, const cv::Mat &mat);
    static cv::DMatch matchFromBinary(std::ifstream &resultsFromFile);
    static cv::KeyPoint keypointFromBinary(std::ifstream &resultsFromFile);
    static void keypointsFromBinary(std::ifstream &resultsFromFile, std::vector<cv::KeyPoint> &kps);
    static cv::Mat cvMatFromBinary(std::ifstream &resultsFromFile);

  private:
    FileHelper() = default;
    ~FileHelper() = default;
  };
}
