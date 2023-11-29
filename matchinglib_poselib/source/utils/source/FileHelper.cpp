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

#include <FileHelper.h>
#include <sstream>
#include <errno.h>
#include <filesystem>

#ifdef WIN32
#include <direntWindows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

namespace fs = std::filesystem;

namespace utilslib
{
  bool FileHelper::fileExists(const std::string& path)
  {
    std::ifstream f(path.c_str());
    if (f.good()) {
      f.close();
      return true;
    }

    return false;
  }

  bool FileHelper::directoryExists(const std::string& directory)
  {
    DIR *dirp;
    // struct dirent *direntp;

    dirp = opendir(directory.c_str());
    if (dirp != NULL) {
      closedir(dirp);
      return true;
    }

    return false;
  }

  bool FileHelper::deleteFile(const std::string &path){
    if (std::remove(path.c_str()) != 0){
      return false;
    }
    return true;
  }

  bool FileHelper::deleteDirectory(const std::string &path) 
  {
    if (std::filesystem::remove_all(path.c_str()) > 0)
    {
      return true;
    }
    return false;
  }

  void FileHelper::ensureFileExists(const std::string& path)
  {
    if (!fileExists(path)) {
      std::stringstream ss;
      ss << "file \"" << path << "\" does not exist!" << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void FileHelper::ensureDirectoryExists(const std::string& directory)
  {
    if (directoryExists(directory)) {
      return;
    }

    //if (!directoryExists(directory)) {
    //  std::stringstream ss;
    //  ss << "directory \"" << directory << "\" does not exist!" << std::endl;
    //  throw std::runtime_error(ss.str());
    //}

#if defined( WIN32 )
    std::filesystem::create_directory(directory);
#elif defined( __linux__ )
    // std::filesystem::create_directory(directory);
    int res = mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    if (res == -1)
      throw std::runtime_error("failed to create directory");
#else
#error "Unsupported Platform!"
#endif
  }

  bool FileHelper::filePathEndsWith(const std::string& path, const std::string& ending)
  {
    if (path.length() >= ending.length()) {
      return (0 == path.compare(path.length() - ending.length(), ending.length(), ending));
    }
    else {
      return false;
    }
  }

  std::string FileHelper::joinPaths(const std::string& path1, const std::string& path2)
  {
    if (filePathEndsWith(path1, "/") || filePathEndsWith(path1, "\\")) {
      return path1 + path2;
    }

    return path1 + "/" + path2;
  }

  std::string FileHelper::getBaseName(const std::string & path)
  {
    auto baseName = path;

    // remove path prefix
    auto lastIndexBSBS = baseName.find_last_of("\\");
    auto lastIndexSlash = baseName.find_last_of("/");

    size_t maxIndex = 0;

    if (lastIndexBSBS == std::string::npos) {
      maxIndex = lastIndexSlash;
    }
    else if (lastIndexSlash == std::string::npos) {
      maxIndex = lastIndexBSBS;
    }
    else {
      maxIndex = lastIndexBSBS > lastIndexSlash ? lastIndexBSBS : lastIndexSlash;
    }

    if (maxIndex != std::string::npos) {
      baseName = baseName.substr(maxIndex + 1, baseName.length() - maxIndex - 1);
    }

    // remove extension
    auto lastIndexDot = baseName.find_last_of(".");
    if (lastIndexDot == std::string::npos ||
      lastIndexDot == 0) {
      // no extension or . is at the front -> hidden file
      return baseName;
    }

    return baseName.substr(0, lastIndexDot);
  }

  std::string FileHelper::getDirName(const std::string &path)
  {
    auto baseName = path;

    // remove path prefix
    auto lastIndexBSBS = baseName.find_last_of("\\");
    auto lastIndexSlash = baseName.find_last_of("/");

    size_t maxIndex = 0;

    if (lastIndexBSBS == std::string::npos)
    {
      maxIndex = lastIndexSlash;
    }
    else if (lastIndexSlash == std::string::npos)
    {
      maxIndex = lastIndexBSBS;
    }
    else
    {
      maxIndex = lastIndexBSBS > lastIndexSlash ? lastIndexBSBS : lastIndexSlash;
    }

    if (maxIndex != std::string::npos)
    {
      baseName = baseName.substr(0, maxIndex);
    }
    return baseName;
  }

  std::vector<std::string> FileHelper::getDirectories(const std::string& path){
    std::vector<std::string> sub_dirs;
    for (const auto & entry : fs::directory_iterator(path)){
      if (entry.is_directory()){
        sub_dirs.emplace_back(entry.path().string());
      }
    }
    return sub_dirs;
  }

  std::vector<std::string> FileHelper::getFiles(const std::string& path){
    std::vector<std::string> files;
    for (const auto & entry : fs::directory_iterator(path)){
      if (entry.is_regular_file()){
        files.emplace_back(entry.path().string());
      }
    }
    return files;
  }

  void FileHelper::matchToBinary(std::ofstream &resultsToFile, const cv::DMatch &m)
  {
    resultsToFile.write((char *)&m.queryIdx, sizeof(m.queryIdx));
    resultsToFile.write((char *)&m.trainIdx, sizeof(m.trainIdx));
    resultsToFile.write((char *)&m.distance, sizeof(m.distance));
  }

  cv::DMatch FileHelper::matchFromBinary(std::ifstream &resultsFromFile)
  {
    cv::DMatch m;
    resultsFromFile.read((char *)&m.queryIdx, sizeof(int));
    resultsFromFile.read((char *)&m.trainIdx, sizeof(int));
    resultsFromFile.read((char *)&m.distance, sizeof(float));
    return m;
  }

  void FileHelper::keypointToBinary(std::ofstream &resultsToFile, const cv::KeyPoint &kp)
  {
    resultsToFile.write((char *)&kp.angle, sizeof(kp.angle));
    resultsToFile.write((char *)&kp.octave, sizeof(kp.octave));
    resultsToFile.write((char *)&kp.response, sizeof(kp.response));
    resultsToFile.write((char *)&kp.size, sizeof(kp.size));
    resultsToFile.write((char *)&kp.pt, sizeof(kp.pt));
  }

  cv::KeyPoint FileHelper::keypointFromBinary(std::ifstream &resultsFromFile)
  {
    cv::KeyPoint kp;
    resultsFromFile.read((char *)&kp.angle, sizeof(float));
    resultsFromFile.read((char *)&kp.octave, sizeof(int));
    resultsFromFile.read((char *)&kp.response, sizeof(float));
    resultsFromFile.read((char *)&kp.size, sizeof(float));
    resultsFromFile.read((char *)&kp.pt, sizeof(cv::Point2f));
    return kp;
  }

  void FileHelper::keypointsToBinary(std::ofstream &resultsToFile, const std::vector<cv::KeyPoint> &kps)
  {
    size_t nr_values = kps.size();
    resultsToFile.write((char *)&nr_values, sizeof(nr_values));
    for (const auto &k : kps)
    {
      keypointToBinary(resultsToFile, k);
    }
  }

  void FileHelper::keypointsFromBinary(std::ifstream &resultsFromFile, std::vector<cv::KeyPoint> &kps)
  {
    size_t nr_values;
    resultsFromFile.read((char *)&nr_values, sizeof(size_t));
    kps.resize(nr_values);
    for (size_t i = 0; i < nr_values; ++i)
    {
      kps[i] = keypointFromBinary(resultsFromFile);
    }
  }

  void FileHelper::cvMatToBinary(std::ofstream &resultsToFile, const cv::Mat &mat)
  {
    // Header
    const int type = mat.type();
    const int channels = mat.channels();
    resultsToFile.write((char *)&mat.rows, sizeof(int)); // rows
    resultsToFile.write((char *)&mat.cols, sizeof(int)); // cols
    resultsToFile.write((char *)&type, sizeof(int));     // type
    resultsToFile.write((char *)&channels, sizeof(int)); // channels

    // Data
    if (mat.isContinuous())
    {
      resultsToFile.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    }
    else
    {
      int rowsz = CV_ELEM_SIZE(type) * mat.cols;
      for (int r = 0; r < mat.rows; ++r)
      {
        resultsToFile.write(mat.ptr<char>(r), rowsz);
      }
    }
  }

  cv::Mat FileHelper::cvMatFromBinary(std::ifstream &resultsFromFile)
  {
    // Header
    int rows, cols, type, channels;
    resultsFromFile.read((char *)&rows, sizeof(int));     // rows
    resultsFromFile.read((char *)&cols, sizeof(int));     // cols
    resultsFromFile.read((char *)&type, sizeof(int));     // type
    resultsFromFile.read((char *)&channels, sizeof(int)); // channels

    // Data
    cv::Mat mat(rows, cols, type);
    resultsFromFile.read((char *)mat.data, CV_ELEM_SIZE(type) * rows * cols);

    return mat;
  }
}