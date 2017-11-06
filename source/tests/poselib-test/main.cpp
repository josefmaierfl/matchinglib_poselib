
#if 0

#include <gmock/gmock.h>

int main(int argc, char* argv[])
{
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}

#else

// ideal case
#include "matchinglib/matchinglib.h"
#include "matchinglib/vfcMatches.h"
#include "poselib/pose_estim.h"
#include "poselib/pose_helper.h"
#include "poselib/pose_homography.h"
#include "poselib/pose_linear_refinement.h"
#include "poselib/stereo_pose_refinement.h"
// ---------------------

#include "opencv2/imgproc/imgproc.hpp"

#include "argvparser.h"
#include "io_data.h"
#include "gtest/gtest.h"
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <memory>

using namespace std;
using namespace cv;
using namespace CommandLineProcessing;

void showMatches(cv::Mat img1, cv::Mat img2,
                 std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2,
                 std::vector<cv::DMatch> matches,
                 int nrFeatures,
                 bool drawAllKps = false);
bool readCalibMat(std::ifstream& calibFile, std::string label, cv::Size matsize, cv::Mat& calibMat);
int loadCalibFile(std::string filepath, std::string filename, cv::Mat& R0, cv::Mat& R1, cv::Mat& t0, cv::Mat& t1, cv::Mat& K0, cv::Mat& K1, cv::Mat& dist0, cv::Mat& dist1, int nrDist = 5);
void cinfigureUSAC(poselib::ConfigUSAC &cfg,
	int cfgUSACnr[6],
	double USACdegenTh,
	cv::Mat K0,
	cv::Mat K1,
	cv::Size imgSize,
	std::vector<cv::KeyPoint> *kp1,
	std::vector<cv::KeyPoint> *kp2,
	std::vector<cv::DMatch> *finalMatches,
	double th_pix_user);



int loadCalibFile(std::string filepath, std::string filename, cv::Mat& R0, cv::Mat& R1, cv::Mat& t0, cv::Mat& t1, cv::Mat& K0, cv::Mat& K1, cv::Mat& dist0, cv::Mat& dist1, int nrDist)
{
    if(filepath.empty() || filename.empty())
        return -1;

    string filenameGT;
    if(filepath.back() == '/' || filepath.back() == '\\')
        filenameGT = filepath + filename;
    else
        filenameGT = filepath + "/" + filename;
    std::ifstream calibFile(filenameGT.c_str());
    if(!calibFile.good())
    {
        calibFile.close();
        return -1;
    }

    if(!readCalibMat(calibFile, "K_00:", cv::Size(3, 3), K0))
    {
        calibFile.close();
        return -2;
    }
    if(!readCalibMat(calibFile, "R_00:", cv::Size(3, 3), R0))
    {
        calibFile.close();
        return -2;
    }
    if(!readCalibMat(calibFile, "T_00:", cv::Size(3, 1), t0))
    {
        calibFile.close();
        return -2;
    }
    if (nrDist == 5)
    {
        if (!readCalibMat(calibFile, "D_00:", cv::Size(5, 1), dist0))
        {
            calibFile.close();
            return -2;
        }
    }
    else if (nrDist == 8)
    {
        if (!readCalibMat(calibFile, "D_00:", cv::Size(8, 1), dist0))
        {
            calibFile.close();
            return -2;
        }
    }
    else
        return -2;
    if(!readCalibMat(calibFile, "K_01:", cv::Size(3, 3), K1))
    {
        calibFile.close();
        return -2;
    }
    if(!readCalibMat(calibFile, "R_01:", cv::Size(3, 3), R1))
    {
        calibFile.close();
        return -2;
    }
    if(!readCalibMat(calibFile, "T_01:", cv::Size(3, 1), t1))
    {
        calibFile.close();
        return -2;
    }
    if (nrDist == 5)
    {
        if (!readCalibMat(calibFile, "D_01:", cv::Size(5, 1), dist1))
        {
            calibFile.close();
            return -2;
        }
    }
    else if (nrDist == 8)
    {
        if (!readCalibMat(calibFile, "D_01:", cv::Size(8, 1), dist1))
        {
            calibFile.close();
            return -2;
        }
    }
    else
        return -2;

    calibFile.close();

    return 0;
}

bool readCalibMat(std::ifstream& calibFile, std::string label, cv::Size matsize, cv::Mat& calibMat)
{
    string line;

    calibMat = Mat(matsize, CV_64FC1);

    calibFile.seekg(0, calibFile.beg);
    std::getline(calibFile, line);
    while(!line.empty() && (line.compare(0, label.size(), label) != 0))
    {
        std::getline(calibFile, line);
    }
    if(line.empty())
    {
        return false;
    }
    else
    {
        std::istringstream is;
        string word;
        int cntx = 0, cnty = 0;
        double val;
        is.str(line);
        is >> word;
        while(is >> val)
        {
            if(cntx >= matsize.width)
            {
                cntx = 0;
                cnty++;
            }
            calibMat.at<double>(cnty, cntx) = val;
            cntx++;
        }
        if((cntx * (cnty + 1)) != (matsize.width * matsize.height))
        {
            return false;
        }
    }

    return true;
}



void SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
{
    testing::internal::FilePath program(argv[0]);
    testing::internal::FilePath program_dir = program.RemoveFileName();
    testing::internal::FilePath data_path = testing::internal::FilePath::ConcatPaths(program_dir,testing::internal::FilePath("imgs//stereo"));

    cmd.setIntroductoryDescription("Interface for testing various keypoint detectors, descriptor extractors, and matching algorithms.\n Example of usage:\n" + std::string(argv[0]) + " --img_path=" + data_path.string() + " --l_img_pref=left_ --r_img_pref=right_ --DynKeyP --subPixRef --c_file=calib_cam_to_cam.txt --autoTH --BART=1");
    //define error codes
    cmd.addErrorCode(0, "Success");
    cmd.addErrorCode(1, "Error");

    cmd.setHelpOption("h", "help","<Shows this help message.>");
    cmd.defineOption("img_path", "<Path to the images (all required in one folder) and the calibration file. All images are loaded one after another for matching and the pose estimation using the specified file prefixes for left and right images. If only the left prefix is specified, images with the same prefix flollowing after another are matched and used for pose estimation.>", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("l_img_pref", "<The prefix of the left or first image. The whole prefix until the start of the number is needed (last character must be '_').>", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("r_img_pref", "<The prefix of the right or second image. The whole prefix until the start of the number is needed (last character must be '_'). Can be empty for image series where one image is matched to the next image.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("f_detect", "<The name of the feature detector in OpenCV 2.4.9 style (FAST, MSER, ORB, BRISK, KAZE, AKAZE, STAR)(For SIFT & SURF, the comments of the corresponding code functions must be removed). [Default=FAST]>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("d_extr", "<The name of the descriptor extractor in OpenCV 2.4.9 style (BRISK, ORB, KAZE, AKAZE, FREAK, DAISY, LATCH)(For SIFT & SURF, the comments of the corresponding code functions must be removed). [Default=FREAK]>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("matcher", "<The short form of the matcher[Default = GMBSOF]:\n "
        "CASHASH : \t Cascade Hashing matcher\n "
        "GMBSOF : \t Guided Matching based on Statistical Optical Flow\n "
        "HIRCLUIDX : \t Hirarchical Clustering Index Matching from the FLANN library\n "
        "HIRKMEANS : \t hierarchical k - means tree matcher from the FLANN library\n "
        "LINEAR : \t Linear matching algorithm(Brute force) from the FLANN library\n "
        "LSHIDX : \t LSH Index Matching algorithm from the FLANN library(not stable(bug in FLANN lib)->program may crash)\n "
        "RANDKDTREE : \t randomized KD - trees matcher from the FLANN library\n "
        "SWGRAPH : \t Small World Graph(SW - graph) from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n "
        "HNSW : \t Hiarchical Navigable Small World Graph. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n "
        "VPTREE : \t VP - tree or ball - tree from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n "
        "MVPTREE : \t Multi - Vantage Point Tree from the NMSLIB. Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n "
        "GHTREE : \t GH - Tree from the NMSLIB.Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n "
        "LISTCLU : \t List of clusters from the NMSLIB.Parameters for the matcher should be specified with options 'nmsIdx' and 'nmsQry'.\n "
        "SATREE : \t Spatial Approximation Tree from the NMSLIB.\n "
        "BRUTEFORCENMS : \t Brute - force(sequential) searching from the NMSLIB.\n "
        "ANNOY : \t Approximate Nearest Neighbors Matcher.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("nmsIdx",
        "<Index parameters for matchers of the NMSLIB. See manual of NMSLIB for details. Instead of '=' in the string you have to use '+'. If you are using a NMSLIB matcher but no parameters are given, the default parameters are used which may leed to unsatisfactory results.>",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("nmsQry",
        "<Query-time parameters for matchers of the NMSLIB. See manual of NMSLIB for details. Instead of '=' in the string you have to use '+'. If you are using a NMSLIB matcher but no parameters are given, the default parameters are used which may leed to unsatisfactory results.>",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("noRatiot", "<If provided, ratio test is disabled for the matchers for which it is possible.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("refineVFC", "<If provided, the result from the matching algorithm is refined with VFC>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("refineSOF", "<If provided, the result from the matching algorithm is refined with SOF>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("DynKeyP", "<If provided, the keypoints are detected dynamically to limit the number of keypoints approximately to the maximum number but are limited using response values. CURRENTLY NOT WORKING with OpenCV 3.0.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("f_nr", "<The maximum number of keypoints per frame [Default=8000] that should be used for matching.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("subPixRef", "<If provided, the feature positions of the final matches are refined by either template matching or OpenCV's corner refinement (cv::cornerSubPix) to get sub-pixel accuracy. Be careful, if there are large rotations, changes in scale or other feature deformations between the matches, template matching option should not be set. The following options are possible:\n 0\t No refinement.\n 1\t Refinement using template matching.\n >1\t Refinement using the OpenCV function cv::cornerSubPix seperately for both images.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("showNr", "<Specifies the number of matches that should be drawn [Default=50]. If the number is set to -1, all matches are drawn. If the number is set to -2, all matches in addition to all not matchable keypoints are drawn.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("v", "<Verbose value [Default=7].\n 0\t Display only pose\n 1\t Display matching time\n 2\t Display feature detection times and matching time\n 3\t Display number of features and matches in addition to all temporal values\n 4\t Display pose & pose estimation time\n 5\t Display pose and pose estimation & refinement times\n 6\t Display all available information\n 7\t Display all available information & visualize the matches.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("c_file", "<Name of the calibration file with file extension. The format of the file corresponds to that provided by KITTI for raw data. For each of the two cameras, the inrinsic ('K_00:', 'K_01:', 'D_00:', 'D_01:') & extrinsic ('R_00:', 'R_01:', 'T_00:', 'T_01:') parameters have to be specified.>", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("noPoseDiff", "<If provided, the calculation of the difference to the given pose is disabled.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("autoTH", "<If provided, the threshold for estimating the pose is automatically adapted to the data. This methode always uses ARRSAC with subsequent refinement.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("refineRT", "<If provided, the pose (R, t) is linearly refined using one of the following options. It consists of a combination of 2 digits [Default=00]:\n "
        "1st digit - choose a refinement algorithm:"
        "\n 0\t no refinement"
        "\n 1\t 8 point algorithm with a pseudo-huber cost-function (old version). Here, the second digit has no effect."
        "\n 2\t 8 point algorithm"
        "\n 3\t Nister"
        "\n 4\t Stewenius"
        "\n 5\t Kneip's Eigen solver is applied on the result (the Essential matrix or for USAC: R,t) of RANSAC, ARRSAC, or USAC directly."
        "\n 6\t Kneip's Eigen solver is applied after extracting R & t and triangulation. This option can be seen as an alternative to bundle adjustment (BA)."
        "\n 2nd digit - choose a weighting function:"
        "\n 0\t Don't use weights"
        "\n 1\t Torr weights (ref: torr dissertation, eqn. 2.25)"
        "\n 2\t Pseudo-Huber weights"
        ">", ArgvParser::OptionRequiresValue);
    cmd.defineOption("BART", "<If provided, the pose (R, t) is refined using bundle adjustment (BA). Try using the option --refineRT in addition to BA. This can lead to a better solution (if --autoTH is enabled, --refineRT is always used). The following options are available:\n 1\t BA for extrinsics only (including structure)\n 2\t BA for extrinsics and intrinsics (including structure)>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("RobMethod", "<Specifies the method for the robust estimation of the essential matrix [Default=USAC]. The following options are available:\n USAC\n ARRSAC\n RANSAC\n LMEDS>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("absCoord", "<If provided, the provided pose is assumed to be related to a specific 3D coordinate orign. Thus, the provided poses are not relativ from camera to camera centre but absolute to a position given by the pose of the first camera.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("Halign", "<If provided, the pose is estimated using homography alignment. Thus, multiple homographies are estimated using ARRSAC. The following options are available:\n 1\t Estimate homographies without a variable threshold\n 2\t Estimate homographies with a variable threshold>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("showRect", "<If provided, the images are rectified and shown using the estimated pose.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("output_path", "<Path where rectified images are saved to.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("distcoeffNr", "<Number of used distortion coeffitients in the calibration file. Can be 5 or 8. If not specifyed a default value of 5 is used.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("histEqual", "<If provided, histogram equalization is applied to the source images.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("cfgUSAC", "<Specifies parameters for USAC. It consists of a combination of 6 digits [Default=311225]. "
        "In the following the options for every digit are explained:\n "
        "1st digit:\n 0\t Use default paramters for SPRT\n 1\t Automatic estimation of SPRT delta\n 2\t Automatic estimation of SPRT epsilon (only without option refineVFC)\n 3\t Automatic estimation of SPRT delta and epsilon\n "
        "2nd digit:\n 0\t Use default paramter for PROSAC beta\n 1\t Automatic estimation of PROSAC beta (uses SPRT delta)\n "
        "3rd digit:\n 0\t Disable prevalidation of samples\n 1\t Enable prevalidation of samples\n "
        "4th digit:\n 0\t Disable degeneracy check\n 1\t Use QDEGSAC for checking degeneracy\n 2\t Use USACs internal degeneracy check\n "
        "5th digit:\n 0\t Estimator: Nister\n 1\t Estimator: Kneip's Eigen solver\n 2\t Estimator: Stewenius\n "
        "6th digit:\n 0\t Inner refinement alg: 8pt with Torr weights\n 1\t Inner refinement alg: 8pt with pseudo-huber weights\n 2\t Inner refinement alg: Kneip's Eigen solver\n 3\t Inner refinement alg: Kneip's Eigen solver with Torr weights\n 4\t Inner refinement alg: Stewenius\n 5\t Inner refinement alg: Stewenius with pseudo-huber weights\n 6\t Inner refinement alg: Nister\n 7\t Inner refinement alg: Nister with pseudo-huber weights>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("USACdegenTh", "<Decision threshold on the inlier ratios between Essential matrix and the degenerate configuration (only rotation) to decide if the solution is degenerate or not [Default=0.85]. It is only used for the internal degeneracy check of USAC (4th digit of option cfgUSAC = 2)>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("th", "<Inlier threshold to check if a match corresponds to a model. [Default=0.8]>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("stereoRef", "<If provided, the algorithm assums a stereo configuration and refines the pose using multiple image pairs.>", ArgvParser::NoOptionAttribute);

    /// finally parse and handle return codes (display help etc...)
    if(argc <= 1)
    {
        if(data_path.DirectoryExists())
        {
            char *newargs[10];
            string arg1str = "--img_path=" + data_path.string();

            if(!cmd.isDefinedOption("img_path") || !cmd.isDefinedOption("l_img_pref") || !cmd.isDefinedOption("r_img_pref")
                || !cmd.isDefinedOption("DynKeyP") || !cmd.isDefinedOption("subPixRef") || !cmd.isDefinedOption("c_file")
                || !cmd.isDefinedOption("autoTH") || !cmd.isDefinedOption("BART") || !cmd.isDefinedOption("showRect"))
            {
                std::cout << "Option definitions changed in code!! Exiting." << endl;
                exit(1);
            }

            newargs[0] = argv[0];
            newargs[1] = (char*)arg1str.c_str();
            newargs[2] = "--l_img_pref=left_";
            newargs[3] = "--r_img_pref=right_";
            newargs[4] = "--DynKeyP";
            newargs[5] = "--subPixRef=1";
            newargs[6] = "--c_file=calib_cam_to_cam.txt";
            newargs[7] = "--autoTH";
            newargs[8] = "--BART=1";
            newargs[9] = "--showRect";

            int result = -1;
            result = cmd.parse(10, newargs);
            if (result != ArgvParser::NoParserError)
            {
                std::cout << cmd.parseErrorDescription(result);
                exit(1);
            }

            std::cout << "Executing the following default command: " << endl;
            std::cout << argv[0] << " " << arg1str << " --l_img_pref=left_ --r_img_pref=right_ --DynKeyP --subPixRef=1 --c_file=calib_cam_to_cam.txt --autoTH --BART=1 --showRect" << endl << endl;
            std::cout << "For options see help with option -h" << endl;
        }
        else
        {
            std::cout << "Standard image path not available!" << endl << "Options necessary - see help below." << endl << endl;
            std::cout << cmd.usageDescription();
            exit(1);
        }
    }
    else
    {
        int result = -1;
        result = cmd.parse(argc, argv);
        if (result != ArgvParser::NoParserError)
        {
            std::cout << cmd.parseErrorDescription(result);
        }
    }
}

void startEvaluation(ArgvParser& cmd)
{
    string img_path, l_img_pref, r_img_pref, f_detect, d_extr, matcher, output_path, nmsIdx, nmsQry;
    string show_str;
    string c_file, RobMethod;
    string cfgUSAC;
    string refineRT;
    double th_pix_user;
    int showNr, f_nr;
    bool noRatiot, refineVFC, refineSOF, DynKeyP, drawSingleKps;
    int subPixRef = 0;
    bool noPoseDiff, autoTH, absCoord, histEqual, refineRTold = false;
    bool showRect;
    int Halign;
    int BART = 0;
    bool oneCam = false;
    int err, verbose;
    vector<string> filenamesl, filenamesr;
    cv::Mat src[2];
    std::vector<cv::DMatch> finalMatches;
    std::vector<cv::KeyPoint> kp1;
    std::vector<cv::KeyPoint> kp2;
    int distcoeffNr;
    double USACdegenTh = 0.85;
    int cfgUSACnr[6] = {3,1,1,2,2,5};
    int refineRTnr[2] = { 0,0};
    bool kneipInsteadBA = false;
	bool stereoRef = false;

    noRatiot = cmd.foundOption("noRatiot");
    refineVFC = cmd.foundOption("refineVFC");
    refineSOF = cmd.foundOption("refineSOF");
    DynKeyP = cmd.foundOption("DynKeyP");
    histEqual = cmd.foundOption("histEqual");

    noPoseDiff = cmd.foundOption("noPoseDiff");
    autoTH = cmd.foundOption("autoTH");
    refineRT = cmd.foundOption("refineRT");
    absCoord = cmd.foundOption("absCoord");

    showRect = cmd.foundOption("showRect");

	stereoRef = cmd.foundOption("stereoRef");

    if (cmd.foundOption("subPixRef"))
    {
        subPixRef = atoi(cmd.optionValue("subPixRef").c_str());
    }

    if(cmd.foundOption("Halign"))
    {
        Halign = atoi(cmd.optionValue("Halign").c_str());
        if((Halign < 0) || (Halign > 2))
        {
            std::cout << "The specified option for homography alignment (Halign) is not available. Exiting." << endl;
            exit(0);
        }
    }
    else
    {
        Halign = 0;
    }


    if(cmd.foundOption("c_file"))
        c_file = cmd.optionValue("c_file");
    else
    {
        std::cout << "Calibration file missing!" << endl;
        exit(1);
    }

    if (cmd.foundOption("distcoeffNr"))
        distcoeffNr = atoi(cmd.optionValue("distcoeffNr").c_str());
    else
        distcoeffNr = 5;

    if(cmd.foundOption("BART"))
    {
        BART = atoi(cmd.optionValue("BART").c_str());
        if((BART < 0) || (BART > 2))
        {
            std::cout << "The specified option for bundle adjustment (BART) is not available. Exiting." << endl;
            exit(0);
        }
    }
    else
        BART = 0;

    if(cmd.foundOption("RobMethod"))
        RobMethod = cmd.optionValue("RobMethod");
    else
        RobMethod = "USAC";

    if(RobMethod.compare("ARRSAC") && autoTH)
    {
        std::cout << "With option 'autoTH' only ARRSAC is supported. Using ARRSAC!" << endl;
    }

    if(RobMethod.compare("ARRSAC") && Halign)
    {
        std::cout << "With option 'Halign' only ARRSAC is supported. Using ARRSAC!" << endl;
    }

    if(autoTH && Halign)
    {
        std::cout << "The options 'autoTH' and 'Halign' are mutually exclusive. Chosse only one of them. Exiting." << endl;
        exit(0);
    }

    if (cmd.foundOption("th"))
    {
        th_pix_user = std::stod(cmd.optionValue("th"));
        if (th_pix_user < 0.1)
        {
            std::cout << "User specific threshold of " << th_pix_user << " is too small. Setting it to 0.1" << endl;
            th_pix_user = 0.1;
        }
        else if (th_pix_user > 5.0)
        {
            std::cout << "User specific threshold of " << th_pix_user << " is too large. Setting it to 5.0" << endl;
            th_pix_user = 5.0;
        }
    }
    else
        th_pix_user = PIX_MIN_GOOD_TH;

    if ((subPixRef != 1) && (th_pix_user < 1.2))
        th_pix_user = 1.2;

    if (cmd.foundOption("cfgUSAC"))
        cfgUSAC = cmd.optionValue("cfgUSAC");
    else
        cfgUSAC = "311225";

    if (cmd.foundOption("refineRT"))
        refineRT = cmd.optionValue("refineRT");
    else
        refineRT = "00";

    if (refineRT.size() == 2)
    {
        refineRTnr[0] = atoi(refineRT.substr(0, 1).c_str());
        refineRTnr[1] = atoi(refineRT.substr(1, 1).c_str());
        if (refineRTnr[0] < 0 || refineRTnr[0] > 6)
        {
            std::cout << "Option for 1st digit of refineRT out of range! Taking default value (disable)." << endl;
            refineRTnr[0] = 0;
        }
        if (refineRTnr[1] > 2)
        {
            std::cout << "Option for 2nd digit of refineRT out of range! Taking default value." << endl;
            refineRTnr[0] = 0;
        }
    }
    else
    {
        std::cout << "Option refineRT is corrupt! Taking default values (disable)." << endl;
    }
    //Set up refinement parameters
    int refineMethod = poselib::RefinePostAlg::PR_NO_REFINEMENT;
    if (refineRTnr[0])
    {
        switch (refineRTnr[0])
        {
        case(1):
            refineRTold = true;
            break;
        case(2):
            refineMethod = poselib::RefinePostAlg::PR_8PT;
            break;
        case(3):
            refineMethod = poselib::RefinePostAlg::PR_NISTER;
            break;
        case(4):
            refineMethod = poselib::RefinePostAlg::PR_STEWENIUS;
            break;
        case(5):
            refineMethod = poselib::RefinePostAlg::PR_KNEIP;
            break;
        case(6):
            refineMethod = poselib::RefinePostAlg::PR_KNEIP;
            kneipInsteadBA = true;
            break;
        default:
            break;
        }

        switch (refineRTnr[1])
        {
        case(0):
            refineMethod = (refineMethod | poselib::RefinePostAlg::PR_NO_WEIGHTS);
            break;
        case(1):
            refineMethod = (refineMethod | poselib::RefinePostAlg::PR_TORR_WEIGHTS);
            break;
        case(2):
            refineMethod = (refineMethod | poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS);
            break;
        default:
            break;
        }
    }

    if (cfgUSAC.size() == 6)
    {
        cfgUSACnr[0] = atoi(cfgUSAC.substr(0, 1).c_str());
        cfgUSACnr[1] = atoi(cfgUSAC.substr(1, 1).c_str());
        cfgUSACnr[2] = atoi(cfgUSAC.substr(2, 1).c_str());
        cfgUSACnr[3] = atoi(cfgUSAC.substr(3, 1).c_str());
        cfgUSACnr[4] = atoi(cfgUSAC.substr(4, 1).c_str());
        cfgUSACnr[5] = atoi(cfgUSAC.substr(5, 1).c_str());
        if (cfgUSACnr[0] < 0 || cfgUSACnr[0] > 3)
        {
            std::cout << "Option for 1st digit of cfgUSAC out of range! Taking default value." << endl;
            cfgUSACnr[0] = 3;
        }
        if (cfgUSACnr[0] > 1 && refineVFC)
        {
            std::cout << "Impossible to estimate epsilon for SPRT if option refineVFC is enabled! Disabling option refineVFC!" << endl;
            refineVFC = false;
        }
        if (cfgUSACnr[1] > 1)
        {
            std::cout << "Option for 2nd digit of cfgUSAC out of range! Taking default value." << endl;
            cfgUSACnr[1] = 1;
        }
        if (cfgUSACnr[2] > 1)
        {
            std::cout << "Option for 3rd digit of cfgUSAC out of range! Taking default value." << endl;
            cfgUSACnr[2] = 1;
        }
        if (cfgUSACnr[3] > 2)
        {
            std::cout << "Option for 4th digit of cfgUSAC out of range! Taking default value." << endl;
            cfgUSACnr[3] = 2;
        }
        if (cfgUSACnr[4] > 2)
        {
            std::cout << "Option for 5th digit of cfgUSAC out of range! Taking default value." << endl;
            cfgUSACnr[4] = 2;
        }
        if (cfgUSACnr[5] > 7)
        {
            std::cout << "Option for 6th digit of cfgUSAC out of range! Taking default value." << endl;
            cfgUSACnr[5] = 5;
        }
    }
    else
    {
        std::cout << "Option cfgUSAC is corrupt! Taking default values." << endl;
    }


    if (cmd.foundOption("USACdegenTh"))
        USACdegenTh = std::stod(cmd.optionValue("USACdegenTh"));
    else
        USACdegenTh = 0.85;

    if(cmd.foundOption("f_detect"))
        f_detect = cmd.optionValue("f_detect");
    else
        f_detect = "FAST";

    if(cmd.foundOption("d_extr"))
        d_extr = cmd.optionValue("d_extr");
    else
        d_extr = "FREAK";

    if(cmd.foundOption("matcher"))
        matcher = cmd.optionValue("matcher");
    else
        matcher = "GMBSOF";

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
        f_nr = atoi(cmd.optionValue("f_nr").c_str());
        if(f_nr <= 10)
        {
            std::cout << "The specified maximum number of keypoints is too low!" << endl;
            exit(1);
        }
    }
    else
        f_nr = 8000;

    if(cmd.foundOption("v"))
    {
        verbose = atoi(cmd.optionValue("v").c_str());
    }
    else
        verbose = 7;

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
        std::cout << "Image path or file prefixes missing!" << endl;
        exit(1);
    }

    img_path = cmd.optionValue("img_path");
    l_img_pref = cmd.optionValue("l_img_pref");

    if (cmd.foundOption("output_path"))
        output_path = cmd.optionValue("output_path");
    else
        output_path = "";

    if(oneCam)
    {
        err = loadImageSequence(img_path, l_img_pref, filenamesl);
        if(err || filenamesl.size() < 2)
        {
            std::cout << "Could not find sequence of images! Exiting." << endl;
            exit(0);
        }
    }
    else
    {
        err = loadStereoSequence(img_path, l_img_pref, r_img_pref, filenamesl, filenamesr);
        if(err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
        {
            std::cout << "Could not find stereo images! Exiting." << endl;
            exit(1);
        }
    }

    if(cmd.foundOption("showNr"))
        show_str = cmd.optionValue("showNr");

    if(!show_str.empty())
    {
        showNr = atoi(show_str.c_str());
        if(showNr == -2)
            drawSingleKps = true;
        else
            drawSingleKps = false;
    }
    else
    {
        showNr = 50;
    }

	cv::Mat R0, R1, t0, t1, K0, K1, dist0_5, dist1_5, dist0_8, dist1_8;
	if (distcoeffNr == 5)
	{
		if (loadCalibFile(img_path, c_file, R0, R1, t0, t1, K0, K1, dist0_5, dist1_5) != 0)
		{
			std::cout << "Format of calibration file not supported. Exiting." << endl;
			exit(0);
		}
		dist0_8 = Mat::zeros(1, 8, dist0_5.type());
		dist1_8 = Mat::zeros(1, 8, dist1_5.type());
		if (dist0_5.rows > dist0_5.cols)
		{
			dist0_5 = dist0_5.t();
			dist1_5 = dist1_5.t();
		}
		dist0_5.copyTo(dist0_8.colRange(0, 5));
		dist1_5.copyTo(dist1_8.colRange(0, 5));
	}
	else
	{
		if (loadCalibFile(img_path, c_file, R0, R1, t0, t1, K0, K1, dist0_8, dist1_8, distcoeffNr) != 0)
		{
			std::cout << "Format of calibration file not supported. Exiting." << endl;
			exit(0);
		}
		if (dist0_8.rows > dist0_8.cols)
		{
			dist0_8 = dist0_8.t();
			dist1_8 = dist1_8.t();
		}
	}
	if (oneCam)
	{
		dist1_8 = dist0_8;
		K1 = K0;
	}

	std::unique_ptr<poselib::StereoRefine> stereoObj;
	if (stereoRef)
	{
		poselib::ConfigPoseEstimation cfg_stereo;

		cfg_stereo.dist0_8 = &dist0_8;
		cfg_stereo.dist1_8 = &dist1_8;
		cfg_stereo.K0 = &K0;
		cfg_stereo.K1 = &K1;
		cfg_stereo.th_pix_user = th_pix_user;
		cfg_stereo.verbose = verbose;
		/*cfg_stereo.Halign = Halign;
		cfg_stereo.autoTH = autoTH;
		cfg_stereo.BART = BART;
		cfg_stereo.kneipInsteadBA = kneipInsteadBA;
		cfg_stereo.refineMethod = refineMethod;
		cfg_stereo.refineRTold = refineRTold;
		cfg_stereo.RobMethod = RobMethod;*/
		cfg_stereo.maxPoolCorrespondences = 20000;

		stereoObj.reset(new poselib::StereoRefine(cfg_stereo));
	}

    int failNr = 0;
  int step = 1;
    for(int i = 0; i < (oneCam ? ((int)filenamesl.size() - step):(int)filenamesl.size()); i++)
    {
        if(oneCam)
        {
            src[0] = cv::imread(filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
            src[1] = cv::imread(filenamesl[i + step],CV_LOAD_IMAGE_GRAYSCALE);
        }
        else
        {
      src[0] = cv::imread(filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
            src[1] = cv::imread(filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
        }

        if (histEqual)
        {
            equalizeHist(src[0], src[0]);
            equalizeHist(src[1], src[1]);
        }

        //Matching
        err = matchinglib::getCorrespondences(src[0], src[1], finalMatches, kp1, kp2, f_detect, d_extr, matcher, DynKeyP, f_nr, refineVFC, !noRatiot, refineSOF, subPixRef, ((verbose < 4) || (verbose > 5)) ? verbose:0, nmsIdx, nmsQry);
        if(err)
        {
            if((err == -5) || (err == -6))
            {
                std::cout << "Exiting!" << endl;
                exit(1);
            }
            failNr++;
            if((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f)) || (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
            {
                std::cout << "Matching failed! Trying next pair." << endl;
                continue;
            }
            else
            {
                std::cout << "Matching failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
                exit(1);
            }
        }

        //Sort matches according to their query index
        /*std::sort(finalMatches.begin(), finalMatches.end(), [](cv::DMatch const & first, cv::DMatch const & second) {
            return first.queryIdx < second.queryIdx; });*/

        if(verbose >= 7)
        {
            showMatches(src[0], src[1], kp1, kp2, finalMatches, showNr, drawSingleKps);
        }

        //Pose estimation
        //-------------------------------

        //Get calibration data from file
        double t_mea = 0, t_oa = 0;
        
        

		cv::Mat R, t;
		if (!stereoRef)
		{

			//Extract coordinates from keypoints
			vector<cv::Point2f> points1, points2;
			for (size_t i = 0; i < finalMatches.size(); i++)
			{
				points1.push_back(kp1[finalMatches[i].queryIdx].pt);
				points2.push_back(kp2[finalMatches[i].trainIdx].pt);
			}

			if (verbose > 5)
			{
				t_mea = (double)getTickCount(); //Start time measurement
			}

			//Transfer into camera coordinates
			poselib::ImgToCamCoordTrans(points1, K0);
			poselib::ImgToCamCoordTrans(points2, K1);

			//Undistort
			if (!poselib::Remove_LensDist(points1, points2, dist0_8, dist1_8))
			{
				failNr++;
				if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f)) || (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
				{
					std::cout << "Undistortion failed! Trying next pair." << endl;
					continue;
				}
				else
				{
					std::cout << "Estimation of essential matrix or undistortion or matching failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
					exit(1);
				}
			}

			if (verbose > 5)
			{
				t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
				std::cout << "Time for coordinate conversion & undistortion (2 imgs): " << t_mea << "ms" << endl;
				t_oa = t_mea;
			}

			if (verbose > 3)
			{
				t_mea = (double)getTickCount(); //Start time measurement
			}

			//Set up USAC paramters
			poselib::ConfigUSAC cfg;
			if (!RobMethod.compare("USAC"))
			{
				cinfigureUSAC(cfg,
					cfgUSACnr,
					USACdegenTh,
					K0,
					K1,
					src[0].size(),
					&kp1,
					&kp2,
					&finalMatches,
					th_pix_user);
			}

			//Get essential matrix
			cv::Mat E, mask, p1, p2;
			cv::Mat R_kneip = cv::Mat::eye(3, 3, CV_64FC1), t_kneip = cv::Mat::zeros(3, 1, CV_64FC1);
			p1 = cv::Mat((int)points1.size(), 2, CV_64FC1);
			p2 = cv::Mat((int)points2.size(), 2, CV_64FC1);
			for (int i = 0; i < (int)points1.size(); i++)
			{
				p1.at<double>(i, 0) = (double)points1[i].x;
				p1.at<double>(i, 1) = (double)points1[i].y;
				p2.at<double>(i, 0) = (double)points2[i].x;
				p2.at<double>(i, 1) = (double)points2[i].y;
			}
			double pixToCamFact = 4.0 / (std::sqrt(2.0) * (K0.at<double>(0, 0) + K0.at<double>(1, 1) + K1.at<double>(0, 0) + K1.at<double>(1, 1)));
			double th = th_pix_user * pixToCamFact; //Inlier threshold
			if (autoTH)
			{
				int inlierPoints;
				poselib::AutoThEpi Eautoth(pixToCamFact);
				if (Eautoth.estimateEVarTH(p1, p2, E, mask, &th, &inlierPoints) != 0)
				{
					failNr++;
					if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f)) || (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
					{
						std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
						continue;
					}
					else
					{
						std::cout << "Estimation of essential matrix or undistortion or matching failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
						exit(1);
					}
				}

				std::cout << "Estimated threshold: " << th / pixToCamFact << " pixels" << endl;
			}
			else if (Halign)
			{
				int inliers;
				if (poselib::estimatePoseHomographies(p1, p2, R, t, E, th, inliers, mask, false, Halign > 1 ? true : false) != 0)
				{
					failNr++;
					if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f)) || (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
					{
						std::cout << "Homography alignment failed! Trying next pair." << endl;
						continue;
					}
					else
					{
						std::cout << "Pose estimation failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
						exit(1);
					}
				}
			}
			else
			{
				if (!RobMethod.compare("USAC"))
				{
					bool isDegenerate = false;
					Mat R_degenerate, inliers_degenerate_R;
					bool usacerror = false;
					if (cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP || cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS)
					{
						if (estimateEssentialOrPoseUSAC(p1,
							p2,
							E,
							th,
							cfg,
							isDegenerate,
							mask,
							R_degenerate,
							inliers_degenerate_R,
							R_kneip,
							t_kneip) != 0)
						{
							usacerror = true;
						}
					}
					else
					{
						if (estimateEssentialOrPoseUSAC(p1,
							p2,
							E,
							th,
							cfg,
							isDegenerate,
							mask,
							R_degenerate,
							inliers_degenerate_R) != 0)
						{
							usacerror = true;
						}
					}
					if (usacerror)
					{
						failNr++;
						if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f)) || (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
						{
							std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
							continue;
						}
						else
						{
							std::cout << "Estimation of essential matrix or undistortion or matching failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
							exit(1);
						}
					}
					if (isDegenerate)
					{
						std::cout << "Camera configuration is degenerate and, thus, rotation only. Skipping further calculations! Rotation angles: " << endl;
						double roll, pitch, yaw;
						poselib::getAnglesRotMat(R_degenerate, roll, pitch, yaw);
						std::cout << "roll: " << roll << char(248) << ", pitch: " << pitch << char(248) << ", yaw: " << yaw << char(248) << endl;
						std::cout << "Trying next pair!" << endl;
						continue;
					}
				}
				else
				{
					if (!poselib::estimateEssentialMat(E, p1, p2, RobMethod, th, refineRTold, mask))
					{
						failNr++;
						if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f)) || (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
						{
							std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
							continue;
						}
						else
						{
							std::cout << "Estimation of essential matrix or undistortion or matching failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
							exit(1);
						}
					}
				}
			}
			size_t nr_inliers = (size_t)cv::countNonZero(mask);
			std::cout << "Number of inliers after robust estimation of E: " << nr_inliers << endl;

			//Get R & t
			bool availableRT = false;
			if (Halign)
			{
				R_kneip = R;
				t_kneip = t;
			}
			if (Halign ||
				(!RobMethod.compare("USAC") && (cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP ||
					cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS)))
			{
				double sumt = 0;
				for (int i = 0; i < 3; i++)
				{
					sumt += t_kneip.at<double>(i);
				}
				if (!poselib::nearZero(sumt) && poselib::isMatRoationMat(R_kneip))
				{
					availableRT = true;
				}
			}
			cv::Mat Q;
			if (Halign && ((refineMethod & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT))
			{
				poselib::triangPts3D(R, t, p1, p2, Q, mask);
			}
			else
			{
				if (refineRTold)
				{
					poselib::robustEssentialRefine(p1, p2, E, E, th / 10.0, 0, true, NULL, NULL, cv::noArray(), mask, 0);
					availableRT = false;
				}
				else if (((refineMethod & 0xF) != poselib::RefinePostAlg::PR_NO_REFINEMENT) && !kneipInsteadBA)
				{
					cv::Mat R_tmp, t_tmp;
					if (availableRT)
					{
						R_kneip.copyTo(R_tmp);
						t_kneip.copyTo(t_tmp);

						if (poselib::refineEssentialLinear(p1, p2, E, mask, refineMethod, nr_inliers, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
						{
							if (!R_tmp.empty() && !t_tmp.empty())
							{
								R_tmp.copyTo(R_kneip);
								t_tmp.copyTo(t_kneip);
							}
						}
						else
							std::cout << "Refinement failed!" << std::endl;
					}
					else if ((refineMethod & 0xF) == poselib::RefinePostAlg::PR_KNEIP)
					{

						if (poselib::refineEssentialLinear(p1, p2, E, mask, refineMethod, nr_inliers, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
						{
							if (!R_tmp.empty() && !t_tmp.empty())
							{
								R_tmp.copyTo(R_kneip);
								t_tmp.copyTo(t_kneip);
								availableRT = true;
							}
							else
								std::cout << "Refinement failed!" << std::endl;
						}
					}
					else
					{
						if (!poselib::refineEssentialLinear(p1, p2, E, mask, refineMethod, nr_inliers, cv::noArray(), cv::noArray(), th, 4, 2.0, 0.1, 0.15))
							std::cout << "Refinement failed!" << std::endl;
					}
				}

				if (!availableRT)
					poselib::getPoseTriangPts(E, p1, p2, R, t, Q, mask);
				else
				{
					R = R_kneip;
					t = t_kneip;
					if ((BART > 0) && !kneipInsteadBA)
						poselib::triangPts3D(R, t, p1, p2, Q, mask);
				}
			}

			if (verbose > 3)
			{
				t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
				std::cout << "Time for pose estimation (includes possible linear refinement): " << t_mea << "ms" << endl;
				t_oa += t_mea;
			}

			if (verbose > 4)
			{
				t_mea = (double)getTickCount(); //Start time measurement
			}

			//Bundle adjustment
			bool useBA = true;
			if (kneipInsteadBA)
			{
				cv::Mat R_tmp, t_tmp;
				R.copyTo(R_tmp);
				t.copyTo(t_tmp);
				if (poselib::refineEssentialLinear(p1, p2, E, mask, refineMethod, nr_inliers, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
				{
					if (!R_tmp.empty() && !t_tmp.empty())
					{
						R_tmp.copyTo(R);
						t_tmp.copyTo(t);
						useBA = false;
					}
				}
				else
				{
					std::cout << "Refinement using Kneips Eigen solver instead of bundle adjustment (BA) failed!" << std::endl;
					if (BART > 0)
					{
						std::cout << "Trying bundle adjustment instead!" << std::endl;
						poselib::triangPts3D(R, t, p1, p2, Q, mask);
					}
				}
			}

			if (useBA)
			{
				if (BART == 1)
				{
					poselib::refineStereoBA(p1, p2, R, t, Q, K0, K1, false, mask);
				}
				else if (BART == 2)
				{
					poselib::CamToImgCoordTrans(p1, K0);
					poselib::CamToImgCoordTrans(p2, K1);
					poselib::refineStereoBA(p1, p2, R, t, Q, K0, K1, true, mask);
				}
			}

			if (verbose > 4)
			{
				t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
				std::cout << "Time for bundle adjustment: " << t_mea << "ms" << endl;
				t_oa += t_mea;
			}
			if (verbose > 5)
			{
				std::cout << "Overall pose estimation time: " << t_oa << "ms" << endl;

				std::cout << "Number of inliers after pose estimation and triangulation: " << cv::countNonZero(mask) << endl;
			}
		}
		else
		{
			//Set up USAC paramters
			poselib::ConfigUSAC cfg;
			if (!RobMethod.compare("USAC"))
			{
				cinfigureUSAC(cfg,
					cfgUSACnr,
					USACdegenTh,
					K0,
					K1,
					src[0].size(),
					&kp1,
					&kp2,
					&finalMatches,
					th_pix_user);
			}

			if (!stereoObj->addNewCorrespondences(finalMatches, kp1, kp2, cfg))
			{
				R = stereoObj->R_new;
				t = stereoObj->t_new;
			}
			


		}

        //Calculate the relative pose between the cameras if an absolute pose was given
        if(absCoord)
        {
            R1 = R1 * R0.inv();
            t1 = t1 - R1 * t0;
        }

        //Normalize the original translation vector
        t1 = t1 / cv::norm(t1);

        //Get the difference in the poses
        if(!noPoseDiff)
        {
            double rdiff, tdiff;
            poselib::compareRTs(R, R1, t, t1, &rdiff, &tdiff, true);
        }

        //Get the rotation angles in degrees and display the translation
        double roll, pitch, yaw;
        poselib::getAnglesRotMat(R1, roll, pitch, yaw);
        std::cout << "Angles of  original rotation: roll = " << setprecision(4) << roll << char(248) << " pitch = " << pitch << char(248) << " yaw = " << yaw << char(248) << endl;
        poselib::getAnglesRotMat(R, roll, pitch, yaw);
        std::cout << "Angles of estimated rotation: roll = " << setprecision(4) << roll << char(248) << " pitch = " << pitch << char(248) << " yaw = " << yaw << char(248) << endl;
        std::cout << "Original  translation vector: [ " << setprecision(4) << t1.at<double>(0) << " " << t1.at<double>(1) << " " << t1.at<double>(2) << " ]" << endl;
        std::cout << "Estimated translation vector: [ " << setprecision(4) << t.at<double>(0) << " " << t.at<double>(1) << " " << t.at<double>(2) << " ]" << endl;
        std::cout << endl << endl;

        if(showRect)
        {
            Mat Rect1, Rect2, K0new, K1new, t0_1, mapX1, mapY1, mapX2, mapY2;

            //get rectification matrices
            poselib::getRectificationParameters(R, t, K0, K1, dist0_8, dist1_8, cv::Size(src[0].cols, src[0].rows), Rect1, Rect2, K0new, K1new, 0.2);

            //get translation vector for mapping left (camera) coordinates to right (camera) coordinates
            t0_1 = -1.0 * R.t() * t;

            //get rectification maps
      initUndistortRectifyMap(K0, dist0_8, Rect1, K0new, cv::Size(src[0].cols, src[0].rows), CV_32FC1, mapX1, mapY1);
            initUndistortRectifyMap(K1, dist1_8, Rect2, K1new, cv::Size(src[0].cols, src[0].rows), CV_32FC1, mapX2, mapY2);

            //Show rectified images
      poselib::ShowRectifiedImages(src[0], src[1], mapX1, mapY1, mapX2, mapY2, t0_1, output_path);
        }
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
            cv::drawMatches( img1, kp1, img2, kp2, matches, drawImg, Scalar::all(-1), Scalar(43, 112, 175), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }
        cv::imshow( "All Matches", drawImg );
        cv::waitKey(0);
        cv::destroyWindow("All Matches");
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
                for (size_t i = 0; i < matches.size(); i++)
                {
                    int idx = matches[i].queryIdx;
                    keypL_reduced.push_back(kp1[idx]);
                    matches_reduced.push_back(matches[i]);
                    matches_reduced.back().queryIdx = i;
                    keypR_reduced.push_back(kp2[matches_reduced.back().trainIdx]);
                    matches_reduced.back().trainIdx = i;
                }
                j = 0;
                for (size_t i = 0; i < matches_reduced.size(); i++)
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
                cv::imshow("Reduced set of matches", img_match);
                cv::waitKey(0);
                cv::destroyWindow("Reduced set of matches");
            }
        }
    }
}

void cinfigureUSAC(poselib::ConfigUSAC &cfg, 
	int cfgUSACnr[6], 
	double USACdegenTh, 
	cv::Mat K0, 
	cv::Mat K1, 
	cv::Size imgSize, 
	std::vector<cv::KeyPoint> *kp1, 
	std::vector<cv::KeyPoint> *kp2, 
	std::vector<cv::DMatch> *finalMatches,
	double th_pix_user)
{
	switch (cfgUSACnr[0])
	{
	case(0):
		cfg.automaticSprtInit = poselib::SprtInit::SPRT_DEFAULT_INIT;
		break;
	case(1):
		cfg.automaticSprtInit = poselib::SprtInit::SPRT_DELTA_AUTOM_INIT;
		break;
	case(2):
		cfg.automaticSprtInit = poselib::SprtInit::SPRT_EPSILON_AUTOM_INIT;
		break;
	case(3):
		cfg.automaticSprtInit = poselib::SprtInit::SPRT_DELTA_AUTOM_INIT | poselib::SprtInit::SPRT_EPSILON_AUTOM_INIT;
		break;
	default:
		cfg.automaticSprtInit = poselib::SprtInit::SPRT_DELTA_AUTOM_INIT | poselib::SprtInit::SPRT_EPSILON_AUTOM_INIT;
		break;
	}
	switch (cfgUSACnr[1])
	{
	case(0):
		cfg.noAutomaticProsacParamters = true;
		break;
	case(1):
		cfg.noAutomaticProsacParamters = false;
		break;
	default:
		cfg.noAutomaticProsacParamters = false;
		break;
	}
	switch (cfgUSACnr[2])
	{
	case(0):
		cfg.prevalidateSample = false;
		break;
	case(1):
		cfg.prevalidateSample = true;
		break;
	default:
		cfg.prevalidateSample = true;
		break;
	}
	switch (cfgUSACnr[3])
	{
	case(0):
		cfg.degeneracyCheck = poselib::UsacChkDegenType::DEGEN_NO_CHECK;
		break;
	case(1):
		cfg.degeneracyCheck = poselib::UsacChkDegenType::DEGEN_QDEGSAC;
		break;
	case(2):
		cfg.degeneracyCheck = poselib::UsacChkDegenType::DEGEN_USAC_INTERNAL;
		break;
	default:
		cfg.degeneracyCheck = poselib::UsacChkDegenType::DEGEN_USAC_INTERNAL;
		break;
	}
	switch (cfgUSACnr[4])
	{
	case(0):
		cfg.estimator = poselib::PoseEstimator::POSE_NISTER;
		break;
	case(1):
		cfg.estimator = poselib::PoseEstimator::POSE_EIG_KNEIP;
		break;
	case(2):
		cfg.estimator = poselib::PoseEstimator::POSE_STEWENIUS;
		break;
	default:
		cfg.estimator = poselib::PoseEstimator::POSE_STEWENIUS;
		break;
	}
	switch (cfgUSACnr[5])
	{
	case(0):
		cfg.refinealg = poselib::RefineAlg::REF_WEIGHTS;
		break;
	case(1):
		cfg.refinealg = poselib::RefineAlg::REF_8PT_PSEUDOHUBER;
		break;
	case(2):
		cfg.refinealg = poselib::RefineAlg::REF_EIG_KNEIP;
		break;
	case(3):
		cfg.refinealg = poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS;
		break;
	case(4):
		cfg.refinealg = poselib::RefineAlg::REF_STEWENIUS;
		break;
	case(5):
		cfg.refinealg = poselib::RefineAlg::REF_STEWENIUS_WEIGHTS;
		break;
	case(6):
		cfg.refinealg = poselib::RefineAlg::REF_NISTER;
		break;
	case(7):
		cfg.refinealg = poselib::RefineAlg::REF_NISTER_WEIGHTS;
		break;
	default:
		cfg.refinealg = poselib::RefineAlg::REF_STEWENIUS_WEIGHTS;
		break;
	}
	cfg.degenDecisionTh = USACdegenTh;
	cfg.focalLength = (K0.at<double>(0, 0) + K0.at<double>(1, 1) + K1.at<double>(0, 0) + K1.at<double>(1, 1)) / 4;
	cfg.imgSize = imgSize;// src[0].size();
	cfg.keypoints1 = kp1;
	cfg.keypoints2 = kp2;
	cfg.matches = finalMatches;
	cfg.th_pixels = th_pix_user; //Threshold for checking degeneracy model inliers (like roation only)

	if (cfgUSACnr[0] > 1)
	{
		vector<cv::DMatch> vfcfilteredMatches;

		if (!filterWithVFC(*kp1, *kp2, *finalMatches, vfcfilteredMatches))
		{
			if ((vfcfilteredMatches.size() > 8) || (finalMatches->size() < 24))
			{
				cfg.nrMatchesVfcFiltered = vfcfilteredMatches.size();
			}
			else
			{
				if (cfgUSACnr[0] == 2)
					cfg.automaticSprtInit = poselib::SprtInit::SPRT_DEFAULT_INIT;
				else
					cfg.automaticSprtInit = poselib::SprtInit::SPRT_DELTA_AUTOM_INIT;
			}
		}
		else
		{
			if (cfgUSACnr[0] == 2)
				cfg.automaticSprtInit = poselib::SprtInit::SPRT_DEFAULT_INIT;
			else
				cfg.automaticSprtInit = poselib::SprtInit::SPRT_DELTA_AUTOM_INIT;
		}
	}
}

/** @function main */
int main( int argc, char* argv[])
{
    ArgvParser cmd;
    SetupCommandlineParser(cmd, argc, argv);
    startEvaluation(cmd);

    return 0;
}

#endif
