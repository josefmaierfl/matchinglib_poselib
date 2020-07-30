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

#include "matchinglib/matchinglib.h"
// ---------------------

#include "argvparser.h"
#include "io_data.h"
#include "gtest/gtest.h"

using namespace std;
using namespace cv;
using namespace CommandLineProcessing;

bool test_supportedFeatureTypes()
{
  vector<string> var =  matchinglib::GetSupportedKeypointTypes();
  bool res = true;
  std::cout << __FUNCTION__ <<  " - supported key types: \n";

  for(auto & i : var)
  {
    std::cout << i << ",";
    res &= matchinglib::IsKeypointTypeSupported(i);
  }

  std::cout << std::endl << std::endl;
  return res;
}

bool test_supportedExtractorTypes()
{
  vector<string> var =  matchinglib::GetSupportedDescriptorTypes();
  bool res = true;
  std::cout << __FUNCTION__ <<  " - supported descriptor types: \n";

  for(auto & i : var)
  {
    std::cout << i << ",";
    res &= matchinglib::IsDescriptorTypeSupported(i);
  }

  std::cout << std::endl << std::endl;
  return res;
}

bool test_supportedMatcherTypes()
{
  vector<string> var =  matchinglib::GetSupportedMatcher();
  bool res = true;
  std::cout << __FUNCTION__ <<  " - supported matcher types: \n";

  for(auto & i : var)
  {
    std::cout << i << ",";
    res &= matchinglib::IsMatcherSupported(i);
  }

  std::cout << std::endl << std::endl;
  return res;
}

void showMatches(cv::Mat img1, cv::Mat img2,
                 std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2,
                 std::vector<cv::DMatch> matches,
                 int nrFeatures,
                 bool drawAllKps = false);

void SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
{
  testing::internal::FilePath program(argv[0]);
  testing::internal::FilePath program_dir = program.RemoveFileName();
  testing::internal::FilePath data_path = testing::internal::FilePath::ConcatPaths(program_dir,testing::internal::FilePath("imgs//flow"));

  cmd.setIntroductoryDescription("Interface for testing various keypoint detectors, descriptor extractors, and matching algorithms.\n Example of usage:\n"
                                 + std::string(argv[0]) + " --img_path=" + data_path.string() + " --l_img_pref=left_ --r_img_pref=right_ ");
  //define error codes
  cmd.addErrorCode(0, "Success");
  cmd.addErrorCode(1, "Error");

  cmd.setHelpOption("h", "help","<Shows this help message.>");
  cmd.defineOption("img_path",
                   "<Path to the images (all required in one folder). All images are loaded one after another for matching using the specified file prefixes for left and right images. If only the left prefix is specified, images with the same prefix flollowing after another are matched.>",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOption("l_img_pref",
                   "<The prefix of the left or first image. The whole prefix until the start of the number is needed (last character must be '_').>",
                   ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
  cmd.defineOption("r_img_pref",
                   "<The prefix of the right or second image. The whole prefix until the start of the number is needed (last character must be '_'). Can be empty for image series where one image is matched to the next image.>",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOption("f_detect",
                   "<The name of the feature detector in OpenCV 3.2 style (FAST, MSER, ORB, BRISK, KAZE, AKAZE, STAR, MSD)(For SIFT & SURF, the comments of the corresponding code functions must be removed). [Default=FAST]>",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOption("d_extr",
                   "<The name of the descriptor extractor in OpenCV 3.2 style (BRISK, ORB, KAZE, AKAZE, FREAK, DAISY, LATCH, BGM, BGM_HARD, BGM_BILINEAR, LBGM, BINBOOST_64, BINBOOST_128, BINBOOST_256, VGG_120, VGG_80, VGG_64, VGG_48)(For SIFT & SURF, the comments of the corresponding code functions must be removed). For the non-OpenCV descriptors use RIFF or BOLD. [Default=FREAK]>",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOption("matcher",
                   "<The short form of the matcher [Default=GMBSOF]:\n CASHASH:\t Cascade Hashing matcher\n GMBSOF:\t Guided Matching based on Statistical Optical Flow\n HIRCLUIDX:\t Hirarchical Clustering Index Matching from the FLANN library\n HIRKMEANS:\t hierarchical k-means tree matcher from the FLANN library\n LINEAR:\t Linear matching algorithm (Brute force) from the FLANN library\n LSHIDX:\t LSH Index Matching algorithm from the FLANN library (not stable (bug in FLANN lib) -> program may crash)\n RANDKDTREE:\t randomized KD-trees matcher from the FLANN library\n SWGRAPH:\t Small World Graph (SW-graph) from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n HNSW:\t Hiarchical Navigable Small World Graph. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n VPTREE:\t VP-tree or ball-tree from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n MVPTREE:\t Multi-Vantage Point Tree from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n GHTREE:\t GH-Tree from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n LISTCLU:\t List of clusters from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n SATREE:\t Spatial Approximation Tree from the NMSLIB.\n BRUTEFORCENMS:\t Brute-force (sequential) searching from the NMSLIB.\n ANNOY:\t Approximate Nearest Neighbors Matcher.>",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOption("noRatiot", "<If provided, ratio test is disabled for the matchers for which it is possible.>",
                   ArgvParser::NoOptionAttribute);
  cmd.defineOption("refineVFC", "<If provided, the result from the matching algorithm is refined with VFC>", ArgvParser::NoOptionAttribute);
  cmd.defineOption("refineSOF", "<If provided, the result from the matching algorithm is refined with SOF>", ArgvParser::NoOptionAttribute);
  cmd.defineOption("refineGMS", "<If provided, the result from the matching algorithm is refined with GMS>", ArgvParser::NoOptionAttribute);
  cmd.defineOption("DynKeyP",
                   "<If provided, the keypoints are detected dynamically to limit the number of keypoints approximately to the maximum number but are limited using response values. CURRENTLY NOT WORKING with OpenCV 3.0.>",
                   ArgvParser::NoOptionAttribute);
  cmd.defineOption("f_nr", "<The maximum number of keypoints per frame [Default=8000] that should be used for matching.>",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOption("subPixRef", "<If provided, the feature positions of the final matches are refined by either template matching or OpenCV's corner refinement (cv::cornerSubPix) to get sub-pixel accuracy. Be careful, if there are large rotations, changes in scale or other feature deformations between the matches, template matching option should not be set. The following options are possible:\n 0\t No refinement.\n 1\t Refinement using template matching.\n >1\t Refinement using the OpenCV function cv::cornerSubPix seperately for both images.>", ArgvParser::OptionRequiresValue);
  cmd.defineOption("showNr",
                   "<Specifies the number of matches that should be drawn [Default=50]. If the number is set to -1, all matches are drawn. If the number is set to -2, all matches in addition to all not matchable keypoints are drawn.>",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOption("v",
                   "<Verbose value [Default=3].\n 0\t no information\n 1\t Display matching time\n 2\t Display feature detection times and matching time\n 3\t Display number of features and matches in addition to all temporal values>",
                   ArgvParser::OptionRequiresValue);
  cmd.defineOption("nmsIdx",
				  "<Index parameters for matchers of the NMSLIB. See manual of NMSLIB for details. Instead of '=' in the string you have to use '+'. If you are using a NMSLIB matcher but no parameters are given, the default parameters are used which may leed to unsatisfactory results.>",
				  ArgvParser::OptionRequiresValue);
  cmd.defineOption("nmsQry",
	  "<Query-time parameters for matchers of the NMSLIB. See manual of NMSLIB for details. Instead of '=' in the string you have to use '+'. If you are using a NMSLIB matcher but no parameters are given, the default parameters are used which may leed to unsatisfactory results.>",
	  ArgvParser::OptionRequiresValue);

  /// finally parse and handle return codes (display help etc...)
  if(argc <= 1)
  {
    if(data_path.DirectoryExists())
    {
      char *newargs[4];
      string arg1str = "--img_path=" + data_path.string();

      if(!cmd.isDefinedOption("img_path") || !cmd.isDefinedOption("l_img_pref") || !cmd.isDefinedOption("r_img_pref"))
      {
        cout << "Option definitions changed in code!! Exiting." << endl;
        exit(1);
      }

      newargs[0] = argv[0];
      newargs[1] = (char*)arg1str.c_str();
      string tmp1 = "--l_img_pref=left_";
      string tmp2 = "--r_img_pref=right_";
      newargs[2] = (char*)tmp1.c_str();
      newargs[3] = (char*)tmp2.c_str();

      int result = -1;
      result = cmd.parse(4, newargs);

      if (result != ArgvParser::NoParserError)
      {
        cout << cmd.parseErrorDescription(result);
        exit(1);
      }

      cout << "Executing the following default command: " << endl;
      cout << argv[0] << " " << arg1str << " --l_img_pref=left_ --r_img_pref=right_" << endl << endl;
      cout << "For options see help with option -h" << endl;
    }
    else
    {
      cout << "Standard image path not available!" << endl << "Options necessary - see help below." << endl << endl;
      cout << cmd.usageDescription();
      exit(1);
    }
  }
  else
  {
    int result = -1;
    result = cmd.parse(argc, argv);

    if (result != ArgvParser::NoParserError)
    {
      cout << cmd.parseErrorDescription(result);
    }
  }
}

void startEvaluation(ArgvParser& cmd)
{
  string img_path, l_img_pref, r_img_pref, f_detect, d_extr, matcher, nmsIdx, nmsQry;
  string show_str;
  int showNr, f_nr;
  bool noRatiot, refineVFC, refineSOF, refineGMS, DynKeyP, drawSingleKps = false;
  int subPixRef = 0;
  bool oneCam = false;
  int err, verbose;
  vector<string> filenamesl, filenamesr;
  cv::Mat src[2];
  std::vector<cv::DMatch> finalMatches;
  std::vector<cv::KeyPoint> kp1;
  std::vector<cv::KeyPoint> kp2;

  noRatiot = cmd.foundOption("noRatiot");
  refineVFC = cmd.foundOption("refineVFC");
  refineSOF = cmd.foundOption("refineSOF");
  refineGMS = cmd.foundOption("refineGMS");
  DynKeyP = cmd.foundOption("DynKeyP");
  
  if (cmd.foundOption("subPixRef"))
  {
	  subPixRef = stoi(cmd.optionValue("subPixRef"));
  }

  if(cmd.foundOption("f_detect"))
  {
    f_detect = cmd.optionValue("f_detect");
  }
  else
  {
    f_detect = "FAST";
  }

  if(cmd.foundOption("d_extr"))
  {
    d_extr = cmd.optionValue("d_extr");
  }
  else
  {
    d_extr = "ORB";
  }

  if(cmd.foundOption("matcher"))
  {
    matcher = cmd.optionValue("matcher");
  }
  else
  {
    matcher = "GMBSOF";
  }

  if (cmd.foundOption("nmsIdx"))
  {
	  nmsIdx = cmd.optionValue("nmsIdx");
	  std::replace(nmsIdx.begin(), nmsIdx.end(), '+', '=');
  }
  else
  {
	  nmsIdx = "";
  }

  if (cmd.foundOption("nmsQry"))
  {
	  nmsQry = cmd.optionValue("nmsQry");
	  std::replace(nmsQry.begin(), nmsQry.end(), '+', '=');
  }
  else
  {
	  nmsQry = "";
  }


  if(cmd.foundOption("f_nr"))
  {
    f_nr = stoi(cmd.optionValue("f_nr"));

    if(f_nr <= 10)
    {
      cout << "The specified maximum number of keypoints is too low!" << endl;
      exit(1);
    }
  }
  else
  {
    f_nr = 8000;
  }

  if(cmd.foundOption("v"))
  {
    verbose = stoi(cmd.optionValue("v"));
  }
  else
  {
    verbose = 3;
  }

  if(cmd.foundOption("img_path") && cmd.foundOption("l_img_pref") && !cmd.foundOption("r_img_pref"))
  {
    oneCam = true;
  }
  else if(cmd.foundOption("img_path") && cmd.foundOption("l_img_pref") && cmd.foundOption("r_img_pref"))
  {
    oneCam = false;
    r_img_pref = cmd.optionValue("r_img_pref");
  }
  else
  {
    cout << "Image path or file prefixes missing!" << endl;
    exit(1);
  }

  img_path = cmd.optionValue("img_path");
  l_img_pref = cmd.optionValue("l_img_pref");

  if(oneCam)
  {
    err = loadImageSequence(img_path, l_img_pref, filenamesl);

    if(err || filenamesl.size() < 2)
    {
      cout << "Could not find sequence of images! Exiting." << endl;
      exit(0);
    }
  }
  else
  {
    err = loadStereoSequence(img_path, l_img_pref, r_img_pref, filenamesl, filenamesr);

    if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
    {
      cout << "Could not find stereo images! Exiting." << endl;
      exit(1);
    }
  }

  if(cmd.foundOption("showNr"))
  {
    show_str = cmd.optionValue("showNr");
  }

  if(!show_str.empty())
  {
    showNr = stoi(show_str);

    drawSingleKps = false;
    if(showNr == -2)
    {
      drawSingleKps = true;
    }
  }
  else
  {
    showNr = 50;
  }

  int failNr = 0;

  for(int i = 0; i < (oneCam ? ((int)filenamesl.size() - 1):(int)filenamesl.size()); i++)
  {
    if(oneCam)
    {
      src[0] = cv::imread(img_path + "/" + filenamesl[i],cv::IMREAD_GRAYSCALE);
      src[1] = cv::imread(img_path + "/" + filenamesl[i + 1],cv::IMREAD_GRAYSCALE);
    }
    else
    {
      src[0] = cv::imread(img_path + "/" + filenamesl[i],cv::IMREAD_GRAYSCALE);
      src[1] = cv::imread(img_path + "/" + filenamesr[i],cv::IMREAD_GRAYSCALE);
    }

    err = matchinglib::getCorrespondences(src[0],
            src[1],
            finalMatches,
            kp1,
            kp2,
            f_detect,
            d_extr,
            matcher,
            DynKeyP,
            f_nr,
            refineVFC,
            refineGMS,
            !noRatiot,
            refineSOF,
            subPixRef,
            verbose,
            nmsIdx,
            nmsQry);

    if(err)
    {
      if((err == -5) || (err == -6))
      {
        cout << "Exiting!" << endl;
        exit(1);
      }

      failNr++;

      if((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f)) || (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
      {
        cout << "Matching failed! Trying next pair." << endl;
        continue;
      }
      else
      {
        cout << "Matching failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
        exit(1);
      }
    }

    showMatches(src[0], src[1], kp1, kp2, finalMatches, showNr, drawSingleKps);
  }
}

void showMatches(cv::Mat img1, cv::Mat img2,
                 std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2,
                 std::vector<cv::DMatch> matches,
                 int nrFeatures,
                 bool drawAllKps)
{
  if(nrFeatures <= 0)
  {
    Mat drawImg;

    if(drawAllKps)
    {
      cv::drawMatches( img1, kp1, img2, kp2, matches, drawImg);
      //imwrite("C:\\work\\matches_final.jpg", drawImg);
    }
    else
    {
      cv::drawMatches( img1, kp1, img2, kp2, matches, drawImg, Scalar::all(-1), Scalar(43, 112, 175), vector<char>(),
                       cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    cv::imshow( "All Matches", drawImg );
  }
  else
  {
    //Show reduced set of matches
    {
      Mat img_match;
      std::vector<cv::KeyPoint> keypL_reduced;//Left keypoints
      std::vector<cv::KeyPoint> keypR_reduced;//Right keypoints
      std::vector<cv::DMatch> matches_reduced;
      std::vector<cv::KeyPoint> keypL_reduced1;//Left keypoints
      std::vector<cv::KeyPoint> keypR_reduced1;//Right keypoints
      std::vector<cv::DMatch> matches_reduced1;
      int j = 0;
      size_t keepNMatches = nrFeatures;

      if(matches.size() > keepNMatches)
      {
        size_t keepXthMatch = matches.size() / keepNMatches;

        for (unsigned int i = 0; i < matches.size(); i++)
        {
          int idx = matches[i].queryIdx;
          keypL_reduced.push_back(kp1[idx]);
          matches_reduced.push_back(matches[i]);
          matches_reduced.back().queryIdx = i;
          keypR_reduced.push_back(kp2[matches_reduced.back().trainIdx]);
          matches_reduced.back().trainIdx = i;
        }

        j = 0;

        for (unsigned int i = 0; i < matches_reduced.size(); i++)
        {
          if((i % (int)keepXthMatch) == 0)
          {
            keypL_reduced1.push_back(keypL_reduced[i]);
            matches_reduced1.push_back(matches_reduced[i]);
            matches_reduced1.back().queryIdx = j;
            keypR_reduced1.push_back(keypR_reduced[i]);
            matches_reduced1.back().trainIdx = j;
            j++;
          }
        }

        drawMatches(img1, keypL_reduced1, img2, keypR_reduced1, matches_reduced1, img_match);
        imshow("Reduced set of matches", img_match);
      }
    }
  }

  cv::waitKey(0);
}

/** @function main */
int main( int argc, char* argv[])
{
  if(!test_supportedMatcherTypes())
  {
    return -1;
  }

  if(!test_supportedFeatureTypes())
  {
    return -1;
  }

  if(!test_supportedExtractorTypes())
  {
    return -1;
  }

  ArgvParser cmd;
  SetupCommandlineParser(cmd, argc, argv);
  startEvaluation(cmd);

  return 0;
}
