
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
#include "matchinglib/gms.h"
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

void showMatches(const cv::Mat &img1, const cv::Mat &img2,
                 std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2,
                 std::vector<cv::DMatch> matches,
                 int nrFeatures,
                 bool drawAllKps = false);
bool readCalibMat(std::ifstream& calibFile, const std::string &label, const cv::Size &matsize, cv::Mat& calibMat);
int loadCalibFile(std::string filepath,
        const std::string &filename,
        cv::Mat& R0,
        cv::Mat& R1,
        cv::Mat& t0,
        cv::Mat& t1,
        cv::Mat& K0,
        cv::Mat& K1,
        cv::Mat& dist0,
        cv::Mat& dist1,
        int nrDist = 5);
void cinfigureUSAC(poselib::ConfigUSAC &cfg,
	int cfgUSACnr[6],
	double USACdegenTh,
	cv::Mat K0,
	cv::Mat K1,
	const cv::Size &imgSize,
	std::vector<cv::KeyPoint> *kp1,
	std::vector<cv::KeyPoint> *kp2,
	std::vector<cv::DMatch> *finalMatches,
	double th_pix_user,
	int USACInlratFilt);



int loadCalibFile(std::string filepath,
        const std::string &filename,
        cv::Mat& R0,
        cv::Mat& R1,
        cv::Mat& t0,
        cv::Mat& t1,
        cv::Mat& K0,
        cv::Mat& K1,
        cv::Mat& dist0,
        cv::Mat& dist1,
        int nrDist)
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

bool readCalibMat(std::ifstream& calibFile, const std::string &label, const cv::Size &matsize, cv::Mat& calibMat)
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
    cmd.defineOption("img_path", "<Path to the images and the calibration file. All images are loaded one after another for matching and the pose estimation using the specified file prefixes for left and right images. If only the left prefix is specified, images with the same prefix flollowing after another are matched and used for pose estimation.>", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("l_img_pref", "<Prefix and/or postfix for the left or first images.\n "
		"It can include a folder structure that follows after the filepath, a file prefix, a '*' indicating the position of the number and a postfix. "
		"If it is empty, all files from the folder img_path are used (also if l_img_pref only contains a folder ending with '/', every file within this folder is used). "
		"It is possible to specify only a prefix with or without '*' at the end. "
		"If a prefix is used, all characters until the first number (excluding) must be provided. "
		"For a postfix, '*' must be placed before the postfix.\n "
		"Valid examples : folder/pre_*post, *post, pre_*, pre_, folder/*post, folder/pre_*, folder/pre_, folder/, folder/folder/, folder/folder/pre_*post, ...\n "
		"For non stereo images (consecutive images), r_img_pref must be empty.>", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("r_img_pref", "<Prefix and/or postfix for the right or second images.\n "
		"For non stereo images (consecutive images), r_img_pref must be empty.\n "
		"For further details see the description of l_img_pref.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("f_detect", "<The name of the feature detector (FAST, MSER, ORB, BRISK, KAZE, AKAZE, STAR, MSD)(For SIFT & SURF, the comments of the corresponding code functions must be removed). [Default=FAST]>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("d_extr", "<The name of the descriptor extractor (BRISK, ORB, KAZE, AKAZE, FREAK, DAISY, LATCH, BGM, BGM_HARD, BGM_BILINEAR, LBGM, BINBOOST_64, BINBOOST_128, BINBOOST_256, VGG_120, VGG_80, VGG_64, VGG_48, RIFF, BOLD )(For SIFT & SURF, the comments of the corresponding code functions must be removed). [Default=FREAK]>", ArgvParser::OptionRequiresValue);
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
	cmd.defineOption("refineGMS", "<If provided, the result from the matching algorithm is refined with GMS>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("DynKeyP", "<If provided, the keypoints are detected dynamically to limit the number of keypoints approximately to the maximum number.>", ArgvParser::NoOptionAttribute);
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
    cmd.defineOption("output_path", "<Path where rectified images are saved to. Only if a path is given, the rectified images are stored to memory.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("distcoeffNr", "<Number of used distortion coeffitients in the calibration file. Can be 5 or 8. If not specifyed a default value of 5 is used.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("histEqual", "<If provided, histogram equalization is applied to the source images.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("cfgUSAC", "<Specifies parameters for USAC. It consists of a combination of 6 digits [Default=311225]. "
        "In the following the options for every digit are explained:\n "
        "1st digit:\n 0\t Use default paramters for SPRT\n 1\t Automatic estimation of SPRT delta\n 2\t Automatic estimation of SPRT epsilon (only without option refineVFC and refineGMS)\n 3\t Automatic estimation of SPRT delta and epsilon\n "
        "2nd digit:\n 0\t Use default paramter for PROSAC beta\n 1\t Automatic estimation of PROSAC beta (uses SPRT delta)\n "
        "3rd digit:\n 0\t Disable prevalidation of samples\n 1\t Enable prevalidation of samples\n "
        "4th digit:\n 0\t Disable degeneracy check\n 1\t Use QDEGSAC for checking degeneracy\n 2\t Use USACs internal degeneracy check\n "
        "5th digit:\n 0\t Estimator: Nister\n 1\t Estimator: Kneip's Eigen solver\n 2\t Estimator: Stewenius\n "
        "6th digit:\n 0\t Inner refinement alg: 8pt with Torr weights\n 1\t Inner refinement alg: 8pt with pseudo-huber weights\n 2\t Inner refinement alg: Kneip's Eigen solver\n 3\t Inner refinement alg: Kneip's Eigen solver with Torr weights\n 4\t Inner refinement alg: Stewenius\n 5\t Inner refinement alg: Stewenius with pseudo-huber weights\n 6\t Inner refinement alg: Nister\n 7\t Inner refinement alg: Nister with pseudo-huber weights>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("USACdegenTh", "<Decision threshold on the inlier ratios between Essential matrix and the degenerate configuration (only rotation) to decide if the solution is degenerate or not [Default=0.85]. It is only used for the internal degeneracy check of USAC (4th digit of option cfgUSAC = 2)>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("USACInlratFilt", "<Specifies which filter is used on the matches to estimate an initial inlier ratio for USAC. Choose 0 for GMS and 1 for VFC [Default].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("th", "<Inlier threshold to check if a match corresponds to a model. [Default=0.8]>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("compInitPose", "<If provided, the estimated pose is compared to the given pose (Ground Truth).>", ArgvParser::NoOptionAttribute);
	
	cmd.defineOption("stereoRef", "<If provided, the algorithm assums a stereo configuration and refines the pose using multiple image pairs.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("evStepStereoStable", "<For stereo refinement: If the option stereoRef is provided and the estimated pose is stable, this option specifies the number of image pairs that are skipped until a new evaluation is performed. A value of 0 disables this feature [Default].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("useOnlyStablePose", "<For stereo refinement: If provided, and the option stereoRef is enabled, only a stable pose is used for rectification after the first stable pose is available. For estimations which do not produce a stable pose, the last stable pose is used. If the real pose is expected to change often, this option should not be used.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("useMostLikelyPose", "<For stereo refinement: If provided, the most likely correct pose over the last poses is preferred (if it is stable) instead of the actual pose.>", ArgvParser::NoOptionAttribute);
	
	cmd.defineOption("refineRT_stereo", "<For stereo refinement: Linear refinement of the pose using all correspondences from the pool with one of the following options. It consists of a combination of 2 digits [Default=42]:\n "
        "1st digit - choose a refinement algorithm:"
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
	cmd.defineOption("BART_stereo", "<For stereo refinement: If provided, the pose (R, t) is refined using bundle adjustment (BA). The following options are available:\n 1\t BA for extrinsics only (including structure)\n 2\t BA for extrinsics and intrinsics (including structure)>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minStartAggInlRat", "<For stereo refinement: Minimum inlier ratio [Default=0.2] at robust estimation to start correspondence aggregation.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("relInlRatThLast", "<For stereo refinement: Maximum relative change of the inlier ratio between image pairs to check by a robsut method if the pose changed [Default=0.35].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("relInlRatThNew", "<For stereo refinement: Maximum relative change [Default=0.2] between the inlier ratio with the last E and the new robustly estimated E on the new image pair to check if the pose has really changed or if only the image pair qulity is very bad (Only if relInlRatThLast does not hold).>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minInlierRatSkip", "<For stereo refinement: Maximum inlier ratio [Default=0.38] using the new robustly estimated E to decide if the image pair quality is too bad (Only if relInlRatThNew does not hold and minInlierRatioReInit is not reached). Below this threshold, a fall-back threshold estimated by relMinInlierRatSkip and the inlier ratio of the last image pair can be used, if the resulting threshold is smaller minInlierRatSkip.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("relMinInlierRatSkip", "<For stereo refinement: Multiplication factor on the inlier ratio [Default=0.7] from the last image pair compared to the new robust estimated one to decide if the new image pair quality is too bad. minInlierRatSkip also influences the decision.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxSkipPairs", "<For stereo refinement: Number of consecutive image pairs [Default=5] where a change in pose or a bad pair was detected until the system is reinitialized.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minInlierRatioReInit", "<For stereo refinement: Minimum inlier ratio [Default=0.67] of the new robust estimation after a change in pose was detected to immediately reinitialize the system (Only if relInlRatThNew does not hold).>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minPtsDistance", "<For stereo refinement: Minimum distance [Default=3.0] between correspondences in the pool (holding the correspondences of the last image pairs).>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxPoolCorrespondences", "<For stereo refinement: Maximum number of correspondences in the pool [Default=30000].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minContStablePoses", "<For stereo refinement: Minimum number of poses that must be very similar in terms of their geometric distance to detect stability [Default=3].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("absThRankingStable", "<For stereo refinement: Maximum normalized error range difference between image pairs to detect stability [Default=0.075]. This normalized error is defined as pose_distance_rating = 1.0 - Pose_Distance_to_all_Poses_gravity_center / max_dist_from_center. absThRankingStable defines the distance region arround the actual pose based on pose_distance_rating +- absThRankingStable. If the maximum pool size is reached and no stability was reached, a different measure based on reprojection error statistics from frame to frame is used (as fall-back) to determine if the computed pose is stable.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("useRANSAC_fewMatches", "<For stereo refinement: If provided, RANSAC for robust estimation if less than 100 matches are available is used.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("checkPoolPoseRobust", "<For stereo refinement: After this number of iterations [Default=3] or new image pairs, robust estimation is performed on the pool correspondences. The number automatically grows exponentially after each robust estimation. Options:"
		"\n 0\t Disabled"
		"\n 1\t Robust estimation is used instead of refinement."
		"\n 2-20\t see above>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minNormDistStable", "<For stereo refinement: Minimum normalized distance [Default=0.5] to the center of gravity of all valid poses to detect stability.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("raiseSkipCnt", "<For stereo refinement: If provided, the value of maxSkipPairs is increased after a specific number of stable consecutive poses was detected [Default=00]. The following options are available:\n "
		"1st digit - Factor to increase maxSkipPairs:"
		"\n 0\t Disable [Default]"
		"\n 1-9\t Increase maxSkipPairs to std::ceil(maxSkipPairs * (1.0 + (1st digit) * 0.25)) if a specific number of stable consecutive poses was detected (defined by 2nd digit)."
		"\n 2nd digit - Number of stable consecutive poses to increase maxSkipPairs:"
		"\n 0-9\t nr = (2nd digit) + 1"
		">", ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxRat3DPtsFar", "<For stereo refinement: Maximum ratio [Default=0.5] of 3D points for which their z-value is very large (maxDist3DPtsZ x baseline) compared to the number of all 3D points. Above this threshold, a pose cannot be marked as stable using only a threshold on the Sampson error ranges (see absThRankingStable).>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxDist3DPtsZ", "<Maximum value for the z-coordinates of 3D points [Default=50.0] to be included into BA. Moreover, this value influences the decision if a pose is marked as stable during stereo refinement (see maxRat3DPtsFar).>", ArgvParser::OptionRequiresValue);

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
            strcpy(newargs[1], arg1str.c_str());
            strcpy(newargs[2], string("--l_img_pref=left_").c_str());
            strcpy(newargs[3], string("--r_img_pref=right_").c_str());
            strcpy(newargs[4], string("--DynKeyP").c_str());
            strcpy(newargs[5], string("--subPixRef=1").c_str());
            strcpy(newargs[6], string("--c_file=calib_cam_to_cam.txt").c_str());
            strcpy(newargs[7], string("--autoTH").c_str());
            strcpy(newargs[8], string("--BART=1").c_str());
            strcpy(newargs[9], string("--showRect").c_str());

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
    string refineRT, refineRT_stereo;
    double th_pix_user = 0;
    int showNr = 0, f_nr = 8000;
    bool noRatiot = false, refineVFC = false, refineSOF = false, refineGMS = false, DynKeyP = true, drawSingleKps = false;
    int subPixRef = 1;
    bool noPoseDiff = false, autoTH = false, absCoord = false, histEqual = true, refineRTold = false, refineRTold_stereo = false;
	int USACInlratFilt = 1;
    bool showRect = false;
    int Halign = 0;
    int BART = 0, BART_stereo = 0;
    bool oneCam = false;
    int err = 0, verbose = 0;
    vector<string> filenamesl, filenamesr;
    cv::Mat src[2];
    std::vector<cv::DMatch> finalMatches;
    std::vector<cv::KeyPoint> kp1;
    std::vector<cv::KeyPoint> kp2;
    int distcoeffNr = 5;
    double USACdegenTh = 0.85;
    int cfgUSACnr[6] = {3,1,1,2,2,5};
    int refineRTnr[2] = { 0,0 }, refineRTnr_stereo[2] = { 4,2 };
    bool kneipInsteadBA = false, kneipInsteadBA_stereo = false;
	bool stereoRef = false;
	int evStepStereoStable = 0;
	bool useOnlyStablePose = false;
	bool useMostLikelyPose = false;
	double minStartAggInlRat = 0.2, 
	relInlRatThLast = 0.35, 
	relInlRatThNew = 0.2, 
	minInlierRatSkip = 0.38, 
	relMinInlierRatSkip = 0.7, 
	minInlierRatioReInit = 0.6, 
	absThRankingStable = 0.075, 
	minNormDistStable = 0.5;
	size_t maxPoolCorrespondences = 30000, maxSkipPairs = 5, minContStablePoses = 3, checkPoolPoseRobust = 3;
	float minPtsDistance = 3.f;
	bool useRANSAC_fewMatches = false;
	bool compInitPose = false;
	string raiseSkipCnt = "00";
	int raiseSkipCntnr[2] = { 0,0 };
	double maxRat3DPtsFar = 0.5;
	double maxDist3DPtsZ = 50.0;
	
    noRatiot = cmd.foundOption("noRatiot");
    refineVFC = cmd.foundOption("refineVFC");
    refineSOF = cmd.foundOption("refineSOF");
	refineGMS = cmd.foundOption("refineGMS");
    DynKeyP = cmd.foundOption("DynKeyP");
    histEqual = cmd.foundOption("histEqual");

    noPoseDiff = cmd.foundOption("noPoseDiff");
    autoTH = cmd.foundOption("autoTH");
    absCoord = cmd.foundOption("absCoord");

    showRect = cmd.foundOption("showRect");
	compInitPose = cmd.foundOption("compInitPose");

	stereoRef = cmd.foundOption("stereoRef");
	useOnlyStablePose = cmd.foundOption("useOnlyStablePose");
	useMostLikelyPose = cmd.foundOption("useMostLikelyPose");

	if (cmd.foundOption("evStepStereoStable"))
	{
		evStepStereoStable = stoi(cmd.optionValue("evStepStereoStable"));
		if (evStepStereoStable < 0 || evStepStereoStable > 1000)
		{
			std::cout << "The number of image pairs skipped " << evStepStereoStable << " before estimating a new pose is out of range. Using default value of 0." << std::endl;
			evStepStereoStable = 0;
		}
	}
	else {
        evStepStereoStable = 0;
    }

    if (cmd.foundOption("subPixRef"))
    {
        subPixRef = stoi(cmd.optionValue("subPixRef"));
    }

    if(cmd.foundOption("Halign"))
    {
        Halign = stoi(cmd.optionValue("Halign"));
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


    if(cmd.foundOption("c_file")) {
        c_file = cmd.optionValue("c_file");
    }
    else
    {
        std::cout << "Calibration file missing!" << endl;
        exit(1);
    }

    if (cmd.foundOption("distcoeffNr")) {
        distcoeffNr = stoi(cmd.optionValue("distcoeffNr"));
    }
    else {
        distcoeffNr = 5;
    }

    if(cmd.foundOption("BART"))
    {
        BART = stoi(cmd.optionValue("BART"));
        if((BART < 0) || (BART > 2))
        {
            std::cout << "The specified option for bundle adjustment (BART) is not available. Exiting." << endl;
            exit(0);
        }
    }
    else {
        BART = 0;
    }

	if (cmd.foundOption("maxDist3DPtsZ"))
	{
		maxDist3DPtsZ = std::stod(cmd.optionValue("maxDist3DPtsZ"));
		if (maxDist3DPtsZ < 5.0 || maxDist3DPtsZ > 1000.0)
		{
			std::cout << "Value for maxDist3DPtsZ out of range. Using default value." << endl;
			maxDist3DPtsZ = 50.0;
		}
	}
	else {
        maxDist3DPtsZ = 50.0;
    }

    if(cmd.foundOption("RobMethod")) {
        RobMethod = cmd.optionValue("RobMethod");
    }
    else {
        RobMethod = "USAC";
    }

	if (cmd.foundOption("USACInlratFilt"))
	{
		USACInlratFilt = stoi(cmd.optionValue("USACInlratFilt"));
		if (USACInlratFilt < 0 || USACInlratFilt > 1)
		{
			std::cout << "The specified option forUSACInlratFilt is not available. Changing to default: VFC filtering" << std::endl;
			USACInlratFilt = 1;
		}
	}
	else {
        USACInlratFilt = 1;
    }

    if((RobMethod != "ARRSAC") && autoTH)
    {
        std::cout << "With option 'autoTH' only ARRSAC is supported. Using ARRSAC!" << endl;
    }

    if((RobMethod != "ARRSAC") && Halign)
    {
        std::cout << "With option 'Halign' only ARRSAC is supported. Using ARRSAC!" << endl;
    }

    if(autoTH && Halign)
    {
        std::cout << "The options 'autoTH' and 'Halign' are mutually exclusive. Chosse only one of them. Exiting." << endl;
        exit(1);
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
    else {
        th_pix_user = PIX_MIN_GOOD_TH;
    }

    if ((subPixRef != 1) && (th_pix_user < 1.2)) {
        th_pix_user = 1.2;
    }

    if (cmd.foundOption("cfgUSAC")) {
        cfgUSAC = cmd.optionValue("cfgUSAC");
    }
    else {
        cfgUSAC = "311225";
    }

    if (cmd.foundOption("refineRT")) {
        refineRT = cmd.optionValue("refineRT");
    }
    else {
        refineRT = "00";
    }

    if (refineRT.size() == 2)
    {
        refineRTnr[0] = stoi(refineRT.substr(0, 1));
        refineRTnr[1] = stoi(refineRT.substr(1, 1));
        if (refineRTnr[0] < 0 || refineRTnr[0] > 6)
        {
            std::cout << "Option for 1st digit of refineRT out of range! Taking default value (disable)." << endl;
            refineRTnr[0] = 0;
        }
        if (refineRTnr[1] > 2)
        {
            std::cout << "Option for 2nd digit of refineRT out of range! Taking default value." << endl;
            refineRTnr[1] = 0;
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

	//For stereo refinement
	if (cmd.foundOption("refineRT_stereo")) {
        refineRT_stereo = cmd.optionValue("refineRT_stereo");
    }
	else {
        refineRT_stereo = "42";
    }
	if (refineRT_stereo.size() == 2)
	{
		refineRTnr_stereo[0] = stoi(refineRT_stereo.substr(0, 1));
		refineRTnr_stereo[1] = stoi(refineRT_stereo.substr(1, 1));
		if (refineRTnr_stereo[0] < 1 || refineRTnr_stereo[0] > 6)
		{
			std::cout << "Option for 1st digit of refineRT_stereo out of range! Taking default value (Stewenius)." << endl;
			refineRTnr_stereo[0] = 4;
		}
		if (refineRTnr_stereo[1] > 2)
		{
			std::cout << "Option for 2nd digit of refineRT_stereo out of range! Taking default value." << endl;
			refineRTnr_stereo[1] = 2;
		}
	}
	else
	{
		std::cout << "Option refineRT is corrupt! Taking default values." << endl;
	}
	//Set up refinement parameters
	int refineMethod_stereo = poselib::RefinePostAlg::PR_NO_REFINEMENT;
	if (refineRTnr_stereo[0])
	{
		switch (refineRTnr_stereo[0])
		{
		case(1):
			refineRTold_stereo = true;
			break;
		case(2):
			refineMethod_stereo = poselib::RefinePostAlg::PR_8PT;
			break;
		case(3):
			refineMethod_stereo = poselib::RefinePostAlg::PR_NISTER;
			break;
		case(4):
			refineMethod_stereo = poselib::RefinePostAlg::PR_STEWENIUS;
			break;
		case(5):
			refineMethod_stereo = poselib::RefinePostAlg::PR_KNEIP;
			break;
		case(6):
			refineMethod_stereo = poselib::RefinePostAlg::PR_KNEIP;
			kneipInsteadBA_stereo = true;
			break;
		default:
			break;
		}

		switch (refineRTnr_stereo[1])
		{
		case(0):
			refineMethod_stereo = (refineMethod_stereo | poselib::RefinePostAlg::PR_NO_WEIGHTS);
			break;
		case(1):
			refineMethod_stereo = (refineMethod_stereo | poselib::RefinePostAlg::PR_TORR_WEIGHTS);
			break;
		case(2):
			refineMethod_stereo = (refineMethod_stereo | poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS);
			break;
		default:
			break;
		}
	}

	if (cmd.foundOption("BART_stereo"))
	{
		BART_stereo = stoi(cmd.optionValue("BART_stereo"));
		if ((BART_stereo < 0) || (BART_stereo > 2))
		{
			std::cout << "The specified option for bundle adjustment (BART_stereo) is not available. Exiting." << endl;
			exit(0);
		}
	}
	else {
        BART_stereo = 0;
    }

	if (cmd.foundOption("minStartAggInlRat"))
	{
		minStartAggInlRat = std::stod(cmd.optionValue("minStartAggInlRat"));
		if (minStartAggInlRat < 0.01 || minStartAggInlRat >= 1.0)
		{
			std::cout << "Value for minStartAggInlRat out of range. Using default value." << endl;
			minStartAggInlRat = 0.2;
		}
	}
	else {
        minStartAggInlRat = 0.2;
    }

	if (cmd.foundOption("relInlRatThLast"))
	{
		relInlRatThLast = std::stod(cmd.optionValue("relInlRatThLast"));
		if (relInlRatThLast < 0.01 || relInlRatThLast >= 1.0)
		{
			std::cout << "Value for relInlRatThLast out of range. Using default value." << endl;
			relInlRatThLast = 0.35;
		}
	}
	else {
        relInlRatThLast = 0.35;
    }

	if (cmd.foundOption("relInlRatThNew"))
	{
		relInlRatThNew = std::stod(cmd.optionValue("relInlRatThNew"));
		if (relInlRatThNew < 0.01 || relInlRatThNew >= 1.0)
		{
			std::cout << "Value for relInlRatThNew out of range. Using default value." << endl;
			relInlRatThNew = 0.2;
		}
	}
	else {
        relInlRatThNew = 0.2;
    }

	if (cmd.foundOption("minInlierRatSkip"))
	{
		minInlierRatSkip = std::stod(cmd.optionValue("minInlierRatSkip"));
		if (minInlierRatSkip < 0.01 || minInlierRatSkip >= 1.0)
		{
			std::cout << "Value for minInlierRatSkip out of range. Using default value." << endl;
			minInlierRatSkip = 0.38;
		}
	}
	else {
        minInlierRatSkip = 0.38;
    }

	if (cmd.foundOption("relMinInlierRatSkip"))
	{
		relMinInlierRatSkip = std::stod(cmd.optionValue("relMinInlierRatSkip"));
		if (relMinInlierRatSkip < 0.01 || relMinInlierRatSkip >= 1.0)
		{
			std::cout << "Value for relMinInlierRatSkip out of range. Using default value." << endl;
			relMinInlierRatSkip = 0.7;
		}
	}
	else {
        relMinInlierRatSkip = 0.7;
    }

	if (cmd.foundOption("maxSkipPairs"))
	{
		std::stringstream ss;
		ss.str(cmd.optionValue("maxSkipPairs"));
		ss >> maxSkipPairs;
		if (maxSkipPairs > 1000)
		{
			std::cout << "Value for maxSkipPairs out of range. Using default value." << endl;
			maxSkipPairs = 5;
		}
	}
	else {
        maxSkipPairs = 5;
    }

	if (cmd.foundOption("minInlierRatioReInit"))
	{
		minInlierRatioReInit = std::stod(cmd.optionValue("minInlierRatioReInit"));
		if (minInlierRatioReInit < 0.01 || minInlierRatioReInit >= 1.0)
		{
			std::cout << "Value for minInlierRatioReInit out of range. Using default value." << endl;
			minInlierRatioReInit = 0.6;
		}
	}
	else {
        minInlierRatioReInit = 0.6;
    }

	if (cmd.foundOption("minPtsDistance"))
	{
		minPtsDistance = std::stof(cmd.optionValue("minPtsDistance"));
		if (minPtsDistance < 1.42f || minPtsDistance > 10.f)
		{
			std::cout << "Value for minPtsDistance out of range. Using default value." << endl;
			minPtsDistance = 3.f;
		}
	}
	else {
        minPtsDistance = 3.f;
    }

	if (cmd.foundOption("maxPoolCorrespondences"))
	{
		std::stringstream ss;
		ss.str(cmd.optionValue("maxPoolCorrespondences"));
		ss >> maxPoolCorrespondences;
		if (maxPoolCorrespondences > 60000)
		{
			std::cout << "Value for maxPoolCorrespondences out of range. Using default value." << endl;
			maxPoolCorrespondences = 30000;
		}
	}
	else {
        maxPoolCorrespondences = 30000;
    }

	if (cmd.foundOption("minContStablePoses"))
	{
		std::stringstream ss;
		ss.str(cmd.optionValue("minContStablePoses"));
		ss >> minContStablePoses;
		if (minContStablePoses > 100)
		{
			std::cout << "Value for minContStablePoses out of range. Using default value." << endl;
			minContStablePoses = 3;
		}
	}
	else {
        minContStablePoses = 3;
    }

	if (cmd.foundOption("absThRankingStable"))
	{
		absThRankingStable = std::stod(cmd.optionValue("absThRankingStable"));
		if (absThRankingStable < 0.01 || absThRankingStable >= 1.0)
		{
			std::cout << "Value for absThRankingStable out of range. Using default value." << endl;
			absThRankingStable = 0.075;
		}
	}
	else {
        absThRankingStable = 0.075;
    }

	useRANSAC_fewMatches = cmd.foundOption("useRANSAC_fewMatches");

	if (cmd.foundOption("checkPoolPoseRobust"))
	{
		std::stringstream ss;
		ss.str(cmd.optionValue("checkPoolPoseRobust"));
		ss >> checkPoolPoseRobust;
		if (checkPoolPoseRobust > 90)
		{
			std::cout << "Value for checkPoolPoseRobust out of range. Using default value." << endl;
			checkPoolPoseRobust = 3;
		}
	}
	else {
        checkPoolPoseRobust = 3;
    }

	if (cmd.foundOption("minNormDistStable"))
	{
		minNormDistStable = std::stod(cmd.optionValue("minNormDistStable"));
		if (minNormDistStable < 0.01 || minNormDistStable >= 1.0)
		{
			std::cout << "Value for minNormDistStable out of range. Using default value." << endl;
			minNormDistStable = 0.5;
		}
	}
	else {
        minNormDistStable = 0.5;
    }

	if (cmd.foundOption("raiseSkipCnt")) {
        raiseSkipCnt = cmd.optionValue("raiseSkipCnt");
    }
	else {
        raiseSkipCnt = "00";
    }
	if (raiseSkipCnt.size() == 2)
	{
		raiseSkipCntnr[0] = stoi(raiseSkipCnt.substr(0, 1));
		raiseSkipCntnr[1] = stoi(raiseSkipCnt.substr(1, 1));
		if (raiseSkipCntnr[0] < 0 || refineRTnr[0] > 9)
		{
			std::cout << "Option for 1st digit of raiseSkipCnt out of range! Taking default value." << endl;
			raiseSkipCntnr[0] = 0;
		}
		if (raiseSkipCntnr[1] < 0 || raiseSkipCntnr[1] > 9)
		{
			std::cout << "Option for 2nd digit of raiseSkipCnt out of range! Taking default value." << endl;
			raiseSkipCntnr[1] = 0;
		}
	}
	else
	{
		std::cout << "Option raiseSkipCnt is corrupt! Taking default values." << endl;
	}

	if (cmd.foundOption("maxRat3DPtsFar"))
	{
		maxRat3DPtsFar = std::stod(cmd.optionValue("maxRat3DPtsFar"));
		if (maxRat3DPtsFar < 0.1 || maxRat3DPtsFar >= 1.0)
		{
			std::cout << "Value for maxRat3DPtsFar out of range. Using default value." << endl;
			maxRat3DPtsFar = 0.5;
		}
	}
	else {
        maxRat3DPtsFar = 0.5;
    }

	//USAC config
    if (cfgUSAC.size() == 6)
    {
        cfgUSACnr[0] = stoi(cfgUSAC.substr(0, 1));
        cfgUSACnr[1] = stoi(cfgUSAC.substr(1, 1));
        cfgUSACnr[2] = stoi(cfgUSAC.substr(2, 1));
        cfgUSACnr[3] = stoi(cfgUSAC.substr(3, 1));
        cfgUSACnr[4] = stoi(cfgUSAC.substr(4, 1));
        cfgUSACnr[5] = stoi(cfgUSAC.substr(5, 1));
        if (cfgUSACnr[0] < 0 || cfgUSACnr[0] > 3)
        {
            std::cout << "Option for 1st digit of cfgUSAC out of range! Taking default value." << endl;
            cfgUSACnr[0] = 3;
        }
        if ((cfgUSACnr[0] > 1) && (refineVFC || refineGMS))
        {
            std::cout << "Impossible to estimate epsilon for SPRT if option refineVFC or refineGMS is enabled! "
                         "Disabling option refineVFC and refineGMS!" << endl;
            refineVFC = false;
            refineGMS = false;
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


    if (cmd.foundOption("USACdegenTh")) {
        USACdegenTh = std::stod(cmd.optionValue("USACdegenTh"));
    }
    else {
        USACdegenTh = 0.85;
    }

    if(cmd.foundOption("f_detect")) {
        f_detect = cmd.optionValue("f_detect");
    }
    else {
        f_detect = "FAST";
    }

    if(cmd.foundOption("d_extr")) {
        d_extr = cmd.optionValue("d_extr");
    }
    else {
        d_extr = "FREAK";
    }

    if(cmd.foundOption("matcher")) {
        matcher = cmd.optionValue("matcher");
    }
    else {
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
            std::cout << "The specified maximum number of keypoints is too low!" << endl;
            exit(1);
        }
    }
    else {
        f_nr = 8000;
    }

    if(cmd.foundOption("v"))
    {
        verbose = stoi(cmd.optionValue("v"));
    }
    else {
        verbose = 7;
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
        std::cout << "Image path or file prefixes missing!" << endl;
        exit(1);
    }

    img_path = cmd.optionValue("img_path");
    l_img_pref = cmd.optionValue("l_img_pref");

    if (cmd.foundOption("output_path")) {
        output_path = cmd.optionValue("output_path");
    }
    else {
        output_path = "";
    }

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

    if(cmd.foundOption("showNr")) {
        show_str = cmd.optionValue("showNr");
    }

    if(!show_str.empty())
    {
        showNr = stoi(show_str);
        drawSingleKps = false;
        if(showNr == -2) {
            drawSingleKps = true;
        }

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
		cfg_stereo.keypointType = f_detect;
		cfg_stereo.descriptorType = d_extr;
		cfg_stereo.th_pix_user = th_pix_user;
		cfg_stereo.verbose = verbose;
		cfg_stereo.Halign = Halign;
		cfg_stereo.autoTH = autoTH;
		cfg_stereo.BART = BART;
		cfg_stereo.kneipInsteadBA = kneipInsteadBA;
		cfg_stereo.refineMethod = refineMethod;
		cfg_stereo.refineRTold = refineRTold;
		cfg_stereo.RobMethod = RobMethod;

		cfg_stereo.refineMethod_CorrPool = refineMethod_stereo;
		cfg_stereo.refineRTold_CorrPool = refineRTold_stereo;
		cfg_stereo.kneipInsteadBA_CorrPool = kneipInsteadBA_stereo;
		cfg_stereo.BART_CorrPool = BART_stereo;
		cfg_stereo.minStartAggInlRat = minStartAggInlRat;
		cfg_stereo.relInlRatThLast = relInlRatThLast;
		cfg_stereo.relInlRatThNew = relInlRatThNew;
		cfg_stereo.minInlierRatSkip = minInlierRatSkip;
		cfg_stereo.relMinInlierRatSkip = relMinInlierRatSkip;
		cfg_stereo.maxSkipPairs = maxSkipPairs;
		cfg_stereo.minInlierRatioReInit = minInlierRatioReInit;
		cfg_stereo.minPtsDistance = minPtsDistance;
		cfg_stereo.maxPoolCorrespondences = maxPoolCorrespondences;
		cfg_stereo.minContStablePoses = minContStablePoses;
		cfg_stereo.absThRankingStable = absThRankingStable;
		cfg_stereo.useRANSAC_fewMatches = useRANSAC_fewMatches;
		cfg_stereo.checkPoolPoseRobust = checkPoolPoseRobust;
		cfg_stereo.minNormDistStable = minNormDistStable;
		cfg_stereo.raiseSkipCnt = (raiseSkipCntnr[0] | (raiseSkipCntnr[1] << 4));
		cfg_stereo.maxRat3DPtsFar = maxRat3DPtsFar;
		cfg_stereo.maxDist3DPtsZ = maxDist3DPtsZ;

		stereoObj.reset(new poselib::StereoRefine(cfg_stereo, verbose > 0));
	}

    int failNr = 0;
	int step = 1;
	const int evStepStereoStable_tmp = evStepStereoStable + 1;
	int evStepStereoStable_cnt = evStepStereoStable_tmp;
	cv::Mat R_stable, t_stable;
    for(int i = 0; i < (oneCam ? ((int)filenamesl.size() - step):(int)filenamesl.size()); i++)
    {
        if(oneCam)
        {
            src[0] = cv::imread(filenamesl[i],cv::IMREAD_GRAYSCALE);
            src[1] = cv::imread(filenamesl[i + step],cv::IMREAD_GRAYSCALE);
        }
        else
        {
            src[0] = cv::imread(filenamesl[i],cv::IMREAD_GRAYSCALE);
            src[1] = cv::imread(filenamesr[i],cv::IMREAD_GRAYSCALE);
        }

		if (!stereoRef || (evStepStereoStable_cnt == evStepStereoStable_tmp) || (evStepStereoStable_cnt == 0))
		{
			if (histEqual)
			{
				equalizeHist(src[0], src[0]);
				equalizeHist(src[1], src[1]);
			}

			//Matching
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
			        ((verbose < 4) || (verbose > 5)) ? verbose : 0,
			        nmsIdx,
			        nmsQry);
			if (err)
			{
				if ((err == -5) || (err == -6))
				{
					std::cout << "Exiting!" << endl;
					exit(1);
				}
				failNr++;
				if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f))
				|| (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
				{
					std::cout << "Matching failed! Trying next pair." << endl;
					continue;
				}
				else
				{
					std::cout << "Matching failed for " << failNr << " image pairs. "
                                                      "Something is wrong with your data! Exiting." << endl;
					exit(1);
				}
			}

			//Sort matches according to their query index
			/*std::sort(finalMatches.begin(), finalMatches.end(), [](cv::DMatch const & first, cv::DMatch const & second) {
				return first.queryIdx < second.queryIdx; });*/

			if (verbose >= 7)
			{
				showMatches(src[0], src[1], kp1, kp2, finalMatches, showNr, drawSingleKps);
			}
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
			for (auto &j : finalMatches)
			{
				points1.push_back(kp1[j.queryIdx].pt);
				points2.push_back(kp2[j.trainIdx].pt);
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
				if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f))
				|| (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
				{
					std::cout << "Undistortion failed! Trying next pair." << endl;
					continue;
				}
				else
				{
					std::cout << "Estimation of essential matrix or undistortion or matching failed for "
					<< failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
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
			if (RobMethod == "USAC")
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
					th_pix_user,
					USACInlratFilt);
			}

			//Get essential matrix
			cv::Mat E, mask, p1, p2;
			cv::Mat R_kneip = cv::Mat::eye(3, 3, CV_64FC1), t_kneip = cv::Mat::zeros(3, 1, CV_64FC1);
			p1 = cv::Mat((int)points1.size(), 2, CV_64FC1);
			p2 = cv::Mat((int)points2.size(), 2, CV_64FC1);
			for (int k = 0; k < (int)points1.size(); k++)
			{
				p1.at<double>(k, 0) = (double)points1[k].x;
				p1.at<double>(k, 1) = (double)points1[k].y;
				p2.at<double>(k, 0) = (double)points2[k].x;
				p2.at<double>(k, 1) = (double)points2[k].y;
			}
			double pixToCamFact = 4.0 / (std::sqrt(2.0) * (K0.at<double>(0, 0)
			        + K0.at<double>(1, 1) + K1.at<double>(0, 0) + K1.at<double>(1, 1)));
			double th = th_pix_user * pixToCamFact; //Inlier threshold
			if (autoTH)
			{
				int inlierPoints;
				poselib::AutoThEpi Eautoth(pixToCamFact);
				if (Eautoth.estimateEVarTH(p1, p2, E, mask, &th, &inlierPoints) != 0)
				{
					failNr++;
					if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f))
					|| (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
					{
						std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
						continue;
					}
					else
					{
						std::cout << "Estimation of essential matrix or undistortion or matching failed for "
						<< failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
						exit(1);
					}
				}

				std::cout << "Estimated threshold: " << th / pixToCamFact << " pixels" << endl;
			}
			else if (Halign)
			{
				int inliers;
				if (poselib::estimatePoseHomographies(p1, p2, R, t, E, th, inliers, mask, false, Halign > 1) != 0)
				{
					failNr++;
					if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f))
					|| (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
					{
						std::cout << "Homography alignment failed! Trying next pair." << endl;
						continue;
					}
					else
					{
						std::cout << "Pose estimation failed for "
						<< failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
						exit(1);
					}
				}
			}
			else
			{
				if (RobMethod == "USAC")
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
							t_kneip,
							verbose > 0) != 0)
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
							inliers_degenerate_R,
							cv::noArray(),
							cv::noArray(),
							verbose > 0) != 0)
						{
							usacerror = true;
						}
					}
					if (usacerror)
					{
						failNr++;
						if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f))
						|| (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
						{
							std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
							continue;
						}
						else
						{
							std::cout << "Estimation of essential matrix or undistortion or matching failed for "
							<< failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
							exit(1);
						}
					}
					if (isDegenerate)
					{
						std::cout << "Camera configuration is degenerate and, thus, rotation only. "
                   "Skipping further calculations! Rotation angles: " << endl;
						double roll, pitch, yaw;
						poselib::getAnglesRotMat(R_degenerate, roll, pitch, yaw);
						std::cout << "roll: " << roll << char(248)
						<< ", pitch: " << pitch << char(248)
						<< ", yaw: " << yaw << char(248) << endl;
						std::cout << "Trying next pair!" << endl;
						continue;
					}
				}
				else
				{
					if (!poselib::estimateEssentialMat(E, p1, p2, RobMethod, th, refineRTold, mask))
					{
						failNr++;
						if ((!oneCam && ((float)failNr / (float)filenamesl.size() < 0.5f))
						|| (oneCam && ((float)(2 * failNr) / (float)filenamesl.size() < 0.5f)))
						{
							std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
							continue;
						}
						else
						{
							std::cout << "Estimation of essential matrix or undistortion or matching failed for "
							<< failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
							exit(1);
						}
					}
				}
			}
			size_t nr_inliers = (size_t)cv::countNonZero(mask);
			if(verbose) {
                std::cout << "Number of inliers after robust estimation of E: " << nr_inliers << endl;
            }

			//Get R & t
			bool availableRT = false;
			if (Halign)
			{
				R_kneip = R;
				t_kneip = t;
			}
			if (Halign ||
				((RobMethod == "USAC") && (cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP ||
					cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS)))
			{
				double sumt = 0;
				for (int j = 0; j < 3; j++)
				{
					sumt += t_kneip.at<double>(j);
				}
				if (!poselib::nearZero(sumt) && poselib::isMatRoationMat(R_kneip))
				{
					availableRT = true;
				}
			}
			cv::Mat Q;
			if (Halign && ((refineMethod & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT))
			{
				if(poselib::triangPts3D(R, t, p1, p2, Q, mask, maxDist3DPtsZ) <= 0){
                    cout << "Triangulating points not successful." << endl;
				}
			}
			else
			{
				if (refineRTold)
				{
					poselib::robustEssentialRefine(p1, p2, E, E, th / 10.0, 0, true, nullptr, nullptr, cv::noArray(), mask, 0);
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
						else {
                            std::cout << "Refinement failed!" << std::endl;
                        }
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
							else {
                                std::cout << "Refinement failed!" << std::endl;
                            }
						}
					}
					else
					{
						if (!poselib::refineEssentialLinear(p1,
						        p2,
						        E,
						        mask,
						        refineMethod,
						        nr_inliers,
						        cv::noArray(),
						        cv::noArray(),
						        th,
						        4, 2.0, 0.1, 0.15))
							std::cout << "Refinement failed!" << std::endl;
					}
				}

				if (!availableRT) {
                    if (poselib::getPoseTriangPts(E, p1, p2, R, t, Q, mask, maxDist3DPtsZ) <= 0) {
                        cout << "Triangulating points not successful." << endl;
                    }
                }
				else
				{
					R = R_kneip;
					t = t_kneip;
//					if ((BART > 0) && !kneipInsteadBA)
                    Mat mask_tmp = mask.clone();
                    int nr_triang = poselib::triangPts3D(R, t, p1, p2, Q, mask, maxDist3DPtsZ);
                    if(nr_triang < 0){
                        if (verbose) {
                            cout << "Triangulating points not successful." << endl;
                        }
                    }else if((nr_triang < ((int)nr_inliers / 3)) && (nr_triang < 100)){
                        Q.release();
                        mask_tmp.copyTo(mask);
                        if (poselib::getPoseTriangPts(E, p1, p2, R, t, Q, mask, maxDist3DPtsZ) <= 0) {
                            if (verbose) {
                                cout << "Triangulating points not successful." << endl;
                            }
                        }
                    }
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
                        Mat mask_tmp = mask.clone();
                        int nr_triang = poselib::triangPts3D(R, t, p1, p2, Q, mask, maxDist3DPtsZ);
                        if(nr_triang < 0){
                            if (verbose) {
                                cout << "Triangulating points not successful." << endl;
                            }
                        }else if((nr_triang < ((int)nr_inliers / 3)) && (nr_triang < 100)){
                            Q.release();
                            mask_tmp.copyTo(mask);
                            if (poselib::getPoseTriangPts(E, p1, p2, R, t, Q, mask, maxDist3DPtsZ) <= 0) {
                                if (verbose) {
                                    cout << "Triangulating points not successful." << endl;
                                }
                            }
                        }
					}
				}
				else
				{
					std::cout << "Refinement using Kneips Eigen solver instead of bundle adjustment (BA) failed!" << std::endl;
					if (BART > 0)
					{
						std::cout << "Trying bundle adjustment instead!" << std::endl;
						poselib::triangPts3D(R, t, p1, p2, Q, mask, maxDist3DPtsZ);
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
			if ((evStepStereoStable_cnt == evStepStereoStable_tmp) || (evStepStereoStable_cnt == 0))
			{
				//Set up USAC paramters
				poselib::ConfigUSAC cfg;
				if (RobMethod == "USAC")
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
						th_pix_user,
						USACInlratFilt);
				}

				static bool poseWasStable = false;
				if (stereoObj->addNewCorrespondences(finalMatches, kp1, kp2, cfg) != -1)
				{
					R = stereoObj->R_new;
					t = stereoObj->t_new;
				}
				else
				{
					cout << "Pose estimation failed!" << endl;
					continue;
				}

				if(evStepStereoStable_cnt == 0)
					evStepStereoStable_cnt = evStepStereoStable_tmp;

				if (useMostLikelyPose && stereoObj->mostLikelyPose_stable)
				{
					R = stereoObj->R_mostLikely;
					t = stereoObj->t_mostLikely;
				}

				if (stereoObj->poseIsStable)
				{
					evStepStereoStable_cnt--;
					poseWasStable = true;
					R.copyTo(R_stable);
					t.copyTo(t_stable);
				}
				else if (poseWasStable && useOnlyStablePose && !(useMostLikelyPose && stereoObj->mostLikelyPose_stable))
				{
					R_stable.copyTo(R);
					t_stable.copyTo(t);
				}
			}
			else
			{
				evStepStereoStable_cnt--;
				R = R_stable;
				t = t_stable;
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
        if(!noPoseDiff && compInitPose)
        {
            double rdiff, tdiff, tdiff_angle;
            poselib::compareRTs(R, R1, t, t1, &rdiff, &tdiff, true);
			cv::Mat R_diff = R * R1.t();
			double roll_d, pitch_d, yaw_d;
			poselib::getAnglesRotMat(R_diff, roll_d, pitch_d, yaw_d);
			std::cout << "Angle between rotation matrices: roll = " << setprecision(4) << roll_d << char(248) << " pitch = " << pitch_d << char(248) << " yaw = " << yaw_d << char(248) << endl;
			tdiff_angle = poselib::getAnglesBetwVectors(t, t1);
			std::cout << "Angle between translation vectors: " << setprecision(3) << tdiff_angle << char(248) << endl;
        }

        //Get the rotation angles in degrees and display the translation
        double roll, pitch, yaw;
		if (compInitPose)
		{
			poselib::getAnglesRotMat(R1, roll, pitch, yaw);
			std::cout << "Angles of  original rotation: roll = " << setprecision(4) << roll << char(248) << " pitch = " << pitch << char(248) << " yaw = " << yaw << char(248) << endl;
		}
        poselib::getAnglesRotMat(R, roll, pitch, yaw);
        std::cout << "Angles of estimated rotation: roll = " << setprecision(4) << roll << char(248) << " pitch = " << pitch << char(248) << " yaw = " << yaw << char(248) << endl;
		std::cout << "Rotation matrix:" << std::endl;
		for (size_t m = 0; m < 3; m++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				std::cout << setprecision(6) << R.at<double>(m, j) << "  ";
			}
			std::cout << std::endl;
		}
		if (compInitPose)
		{
			std::cout << "Original  translation vector: [ " << setprecision(4) << t1.at<double>(0) << " " << t1.at<double>(1) << " " << t1.at<double>(2) << " ]" << endl;
		}
        std::cout << "Estimated translation vector: [ " << setprecision(4) << t.at<double>(0) << " " << t.at<double>(1) << " " << t.at<double>(2) << " ]" << endl;
        std::cout << std::endl << std::endl;

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

void showMatches(const cv::Mat &img1, const cv::Mat &img2,
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
            cv::drawMatches( img1,
                    kp1,
                    img2,
                    kp2,
                    matches,
                    drawImg,
                    Scalar::all(-1),
                    Scalar(43, 112, 175),
                    vector<char>(),
                            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
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
	const cv::Size &imgSize,
	std::vector<cv::KeyPoint> *kp1, 
	std::vector<cv::KeyPoint> *kp2, 
	std::vector<cv::DMatch> *finalMatches,
	double th_pix_user,
	int USACInlratFilt)
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
		int err = 0;
		unsigned int n_f = 0;
		if (USACInlratFilt == 0)
		{
			n_f = (unsigned int)filterMatchesGMS(*kp1, imgSize, *kp2, imgSize, *finalMatches, vfcfilteredMatches);
			if (n_f == 0)
				err = -1;
		}
		else
		{
			err = filterWithVFC(*kp1, *kp2, *finalMatches, vfcfilteredMatches);
			n_f = (unsigned int)vfcfilteredMatches.size();
		}

		if (!err)
		{
			if ((n_f > 8) || (finalMatches->size() < 24))
			{
				cfg.nrMatchesVfcFiltered = n_f;
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
