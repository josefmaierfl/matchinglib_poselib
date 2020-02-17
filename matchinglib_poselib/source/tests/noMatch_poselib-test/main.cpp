
/*#if 0

#include <gmock/gmock.h>

int main(int argc, char* argv[])
{
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}

#else*/

#define TESTOUT 0
#define FAIL_SKIP_REST 0

// ideal case
#include "matchinglib/matchinglib.h"
#include "matchinglib/vfcMatches.h"
#include "matchinglib/gms.h"
#include "poselib/pose_estim.h"
#include "poselib/pose_helper.h"
#include "poselib/pose_homography.h"
#include "poselib/pose_linear_refinement.h"
#include "poselib/stereo_pose_refinement.h"

#include "load_py_ngransac.hpp"
// ---------------------

#include "opencv2/imgproc/imgproc.hpp"

#include "argvparser.h"
#include "io_data.h"
#include "gtest/gtest.h"
#include <opencv2/imgproc.hpp>
#include "loadMatches.h"

#include <fstream>
#include <memory>
#include "chrono"

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
/*#if defined(__linux__)

#include "unistd.h"
#include "thread"
//#include "sys/file.h"
#endif*/

using namespace std;
using namespace cv;
using namespace CommandLineProcessing;

void printCVMat(const cv::Mat &m, std::ofstream &os, const std::string &name = "");

struct timeMeasurements{
    double filtering;
    double robEstimationAndRef;
    double linRefinement;
    double bundleAdjust;
    double stereoRefine;

    timeMeasurements():
            filtering(-1.0),
            robEstimationAndRef(-1.0),
            linRefinement(-1.0),
            bundleAdjust(-1.0),
            stereoRefine(-1.0){};

    void print(std::ofstream &os, const bool pName) const{
        if(pName){
            os << "filtering_us;robEstimationAndRef_us;linRefinement_us;bundleAdjust_us;stereoRefine_us";
        }else{
            os << filtering << ";";
            os << robEstimationAndRef << ";";
            os << linRefinement << ";";
            os << bundleAdjust << ";";
            os << stereoRefine;
        }
    }
};

struct rotAngles{
    double roll;
    double pitch;
    double yaw;

    rotAngles():
    roll(0),
    pitch(0),
    yaw(0){}

    rotAngles(double &roll_, double &pitch_, double &yaw_):
    roll(roll_),
    pitch(pitch_),
    yaw(yaw_){};

    void print(std::ofstream &os, const std::string &baseName = "") const{
        if(!baseName.empty()){
            os << baseName << "_roll_deg;" << baseName << "_pitch_deg;" << baseName << "_yaw_deg";
        }else{
            os << roll << ";";
            os << pitch << ";";
            os << yaw;
        }
    }
};
struct CamMatDiff{
    double fxDiff, fyDiff, fxyDiffNorm, cxDiff, cyDiff, cxyDiffNorm, cxyfxfyNorm;

    CamMatDiff():
            fxDiff(0),
            fyDiff(0),
            fxyDiffNorm(0),
            cxDiff(0),
            cyDiff(0),
            cxyDiffNorm(0),
            cxyfxfyNorm(0){};

    void calcDiff(const cv::Mat &K_estimated, const cv::Mat &K_GT){
        fxDiff = K_estimated.at<double>(0,0) - K_GT.at<double>(0,0);
        fyDiff = K_estimated.at<double>(1,1) - K_GT.at<double>(1,1);
        fxyDiffNorm = sqrt(fxDiff * fxDiff + fyDiff * fyDiff);
        cxDiff = K_estimated.at<double>(0,2) - K_GT.at<double>(0,2);
        cyDiff = K_estimated.at<double>(1,2) - K_GT.at<double>(1,2);
        cxyDiffNorm = sqrt(cxDiff * cxDiff + cyDiff * cyDiff);
        cxyfxfyNorm = sqrt(cxDiff * cxDiff + cyDiff * cyDiff + fxDiff * fxDiff + fyDiff * fyDiff);
    };

    void print(std::ofstream &os, const std::string &baseName = "") const{
        if(!baseName.empty()){
            os << baseName << "_fxDiff;"
                    << baseName << "_fyDiff;"
                    << baseName << "_fxyDiffNorm;"
                    << baseName << "_cxDiff;"
                    << baseName << "_cyDiff;"
                    << baseName << "_cxyDiffNorm;"
                    << baseName << "_cxyfxfyNorm";
        }else{
            os << fxDiff << ";";
            os << fyDiff << ";";
            os << fxyDiffNorm << ";";
            os << cxDiff << ";";
            os << cyDiff << ";";
            os << cxyDiffNorm << ";";
            os << cxyfxfyNorm;
        }
    }
};
struct tElemsDiff{
    double tx, ty, tz;

    tElemsDiff():
    tx(0),
    ty(0),
    tz(0){}

    void calcDiff(const cv::Mat &t_estimated,
            const cv::Mat &t_GT){
        if(t_estimated.empty()){
            return;
        }
        Mat diff = t_estimated - t_GT;
        tx = diff.at<double>(0);
        ty = diff.at<double>(1);
        tz = diff.at<double>(2);
    };

    void print(std::ofstream &os, const std::string &baseName = "") const{
        if(!baseName.empty()){
            os << baseName << "_tx;" << baseName << "_ty;" << baseName << "_tz";
        }else{
            os << tx << ";";
            os << ty << ";";
            os << tz;
        }
    }
};
struct algorithmResult{
    cv::Mat R;
    cv::Mat t;
    cv::Mat R_mostLikely;
    cv::Mat t_mostLikely;
    bool poseIsStable;
    bool mostLikelyPose_stable;
    cv::Mat R_GT;
    cv::Mat t_GT;
    cv::Mat K1, K2;
    cv::Mat K1_GT, K2_GT;
    cv::Mat K1_degenerate, K2_degenerate;
    double inlRat_GT;
    int nrCorrs_GT;
    int nrCorrs_filtered;
    int nrCorrs_estimated;
    double inlRat_estimated;
    timeMeasurements tm;
    rotAngles R_diff;
    double R_diffAll;
    double t_angDiff;
    double t_distDiff;
    tElemsDiff t_elemDiff;
    rotAngles R_mostLikely_diff;
    double R_mostLikely_diffAll;
    double t_mostLikely_angDiff;
    double t_mostLikely_distDiff;
    tElemsDiff t_mostLikely_elemDiff;
    CamMatDiff K1_diff, K2_diff;
    std::string addSequInfo;
    int poolSize;
    int ransac_agg;
    rotAngles R_GT_n_diff;
    tElemsDiff t_GT_n_elemDiff;
    double R_GT_n_diffAll;
    double t_GT_n_angDiff;

    algorithmResult(const std::string &addSequInfo_ = ""): addSequInfo(addSequInfo_){
        R = cv::Mat::zeros(3,3,CV_64FC1);
        t = cv::Mat::zeros(3,1,CV_64FC1);
        R_mostLikely = cv::Mat::zeros(3,3,CV_64FC1);
        t_mostLikely = cv::Mat::zeros(3,1,CV_64FC1);
        poseIsStable = false;
        mostLikelyPose_stable = false;
        R_GT = cv::Mat::zeros(3,3,CV_64FC1);
        t_GT = cv::Mat::zeros(3,1,CV_64FC1);
        K1 = cv::Mat::zeros(3,3,CV_64FC1);
        K2 = cv::Mat::zeros(3,3,CV_64FC1);
        K1_GT = cv::Mat::zeros(3,3,CV_64FC1);
        K2_GT = cv::Mat::zeros(3,3,CV_64FC1);
        K1_degenerate = cv::Mat::zeros(3,3,CV_64FC1);
        K2_degenerate = cv::Mat::zeros(3,3,CV_64FC1);
        inlRat_GT = -1.0;
        nrCorrs_GT = -1;
        nrCorrs_filtered = -1;
        nrCorrs_estimated = -1;
        inlRat_estimated = -1.0;
        tm = timeMeasurements();
        R_diff = rotAngles();
        R_diffAll = 0;
        t_angDiff = 0;
        t_distDiff = 0;
        t_elemDiff = tElemsDiff();
        R_mostLikely_diff = rotAngles();
        R_mostLikely_diffAll = 0;
        t_mostLikely_angDiff = 0;
        t_mostLikely_distDiff = 0;
        t_mostLikely_elemDiff = tElemsDiff();
        K1_diff = CamMatDiff();
        K2_diff = CamMatDiff();
        poolSize = 0;
        ransac_agg = 1;
        R_GT_n_diff = rotAngles();
        t_GT_n_elemDiff = tElemsDiff();
        R_GT_n_diffAll = 0;
        t_GT_n_angDiff = 0;
    }

    void calcRTDiff(const cv::Mat &R_estimated,
                    const cv::Mat &R_GT_,
                    const cv::Mat &t_estimated,
                    const cv::Mat &t_GT_,
                    bool isMostLikely,
                    bool verbose = false,
                    bool gt_elem_diff = false){
        if(R_estimated.empty()
           || (R_estimated.rows != 3)
           || (R_estimated.cols != 3)
           || (R_estimated.type() != CV_64FC1)
           || t_estimated.empty()
           || (t_estimated.rows != 3)
           || (t_estimated.cols != 1)
           || (t_estimated.type() != CV_64FC1)
           || !poselib::isMatRoationMat(R_estimated)
           || !poselib::isMatRoationMat(R_GT_)){
            return;
        }

        //Normalize the translation vectors
        cv::Mat t_estimated_tmp = t_estimated / cv::norm(t_estimated);
        double scale = cv::norm(t_GT_);
        cv::Mat t_GT_tmp = t_GT_ / scale;

        //Store extrinsics
        if(isMostLikely){
            R_estimated.copyTo(R_mostLikely);
            t_mostLikely = scale * t_estimated_tmp;
        }else if(!gt_elem_diff){
            R_estimated.copyTo(R);
            t = scale * t_estimated_tmp;
        }
        if(!gt_elem_diff) {
            R_GT_.copyTo(R_GT);
            t_GT_.copyTo(t_GT);
        }

        if(isMostLikely){
            t_mostLikely_elemDiff.calcDiff(t_estimated_tmp, t_GT_tmp);
        }else if(gt_elem_diff){
            t_GT_n_elemDiff.calcDiff(t_estimated_tmp, t_GT_tmp);
        }else {
            t_elemDiff.calcDiff(t_estimated_tmp, t_GT_tmp);
        }

        double rdiff, tdiff, tdiff_angle;
        poselib::compareRTs(R_estimated, R_GT_, t_estimated_tmp, t_GT_tmp, &rdiff, &tdiff, verbose);
        if(isMostLikely){
            R_mostLikely_diffAll = 180.0 * rdiff / M_PI;
            t_mostLikely_distDiff = tdiff;
        }else if(gt_elem_diff){
            R_GT_n_diffAll = round(180000000.0 * rdiff / M_PI) / 1000000.0;
        }else{
            R_diffAll = 180.0 * rdiff / M_PI;
            t_distDiff = tdiff;
        }
        cv::Mat R_diff_tmp = R_estimated * R_GT_.t();
        double roll_d, pitch_d, yaw_d;
        poselib::getAnglesRotMat(R_diff_tmp, roll_d, pitch_d, yaw_d);
        tdiff_angle = poselib::getAnglesBetwVectors(t_estimated_tmp, t_GT_tmp);
        if(isMostLikely){
            R_mostLikely_diff = rotAngles(roll_d, pitch_d, yaw_d);
            t_mostLikely_angDiff = tdiff_angle;
        }else if(gt_elem_diff){
            R_GT_n_diff = rotAngles(roll_d, pitch_d, yaw_d);
            t_GT_n_angDiff = round(1000000.0 * tdiff_angle) / 1000000.0;
        }else{
            R_diff = rotAngles(roll_d, pitch_d, yaw_d);
            t_angDiff = tdiff_angle;
        }
        if(verbose) {
            std::cout << "Angle between rotation matrices: roll = "
                      << setprecision(4) << roll_d << char(248)
                      << " pitch = " << pitch_d << char(248)
                      << " yaw = " << yaw_d << char(248) << endl;
            std::cout << "Angle between translation vectors: " << setprecision(3) << tdiff_angle << char(248) << endl;
        }
    }

    void calcInlRatGT(const std::vector<bool> &inliers){
        nrCorrs_GT = (int)inliers.size();
        int tmp = 0;
        for(auto &&i : inliers){
            if(i){
                tmp++;
            }
        }
        inlRat_GT = (double)tmp / (double)nrCorrs_GT;
    }

    void calcInlRatEstimated(const cv::Mat &mask){
        if(mask.rows > mask.cols){
            nrCorrs_estimated = mask.rows;
        }else{
            nrCorrs_estimated = mask.cols;
        }
        int tmp = cv::countNonZero(mask);
        inlRat_estimated = (double)tmp / (double)nrCorrs_estimated;
    }

    void calcCamMatDiff(){
        CV_Assert(!K1.empty() && !K2.empty() && !K1_GT.empty() && !K2_GT.empty());
        K1_diff.calcDiff(K1, K1_GT);
        K2_diff.calcDiff(K2, K2_GT);
    }

    void enterCamMats(const cv::Mat &K1_,
                      const cv::Mat &K2_,
                      const cv::Mat &K1_GT_,
                      const cv::Mat &K2_GT_,
                      const cv::Mat &K1_degenerate_,
                      const cv::Mat &K2_degenerate_){
        K1_.copyTo(K1);
        K2_.copyTo(K2);
        K1_GT_.copyTo(K1_GT);
        K2_GT_.copyTo(K2_GT);
        K1_degenerate_.copyTo(K1_degenerate);
        K2_degenerate_.copyTo(K2_degenerate);
        calcCamMatDiff();
    }

    void print(std::ofstream &os, const bool pName) const{
        if(pName){
            os << "R_diffAll;";
            R_diff.print(os,"R_diff");
            os << ";";
            os << "t_angDiff_deg;";
            os << "t_distDiff;";
            t_elemDiff.print(os, "t_diff");
            os << ";";
            os << "R_mostLikely_diffAll;";
            R_mostLikely_diff.print(os, "R_mostLikely_diff");
            os << ";";
            os << "t_mostLikely_angDiff_deg;";
            os << "t_mostLikely_distDiff;";
            t_mostLikely_elemDiff.print(os, "t_mostLikely_diff");
            os << ";";
            printCVMat(R, os, "R_out");
            os << ";";
            printCVMat(t, os, "t_out");
            os << ";";
            printCVMat(R_mostLikely, os, "R_mostLikely");
            os << ";";
            printCVMat(t_mostLikely, os, "t_mostLikely");
            os << ";";
            printCVMat(R_GT, os, "R_GT");
            os << ";";
            printCVMat(t_GT, os, "t_GT");
            os << ";";
            os << "R_GT_n_diffAll;";
            R_GT_n_diff.print(os, "R_GT_n_diff");
            os << ";";
            os << "t_GT_n_angDiff;";
            t_GT_n_elemDiff.print(os, "t_GT_n_elemDiff");
            os << ";";
            os << "poseIsStable;";
            os << "mostLikelyPose_stable;";
            os << "poolSize;";
            os << "ransac_agg;";
            K1_diff.print(os, "K1");
            os << ";";
            K2_diff.print(os, "K2");
            os << ";";
            printCVMat(K1, os, "K1");
            os << ";";
            printCVMat(K2, os, "K2");
            os << ";";
            printCVMat(K1_GT, os, "K1_GT");
            os << ";";
            printCVMat(K2_GT, os, "K2_GT");
            os << ";";
            printCVMat(K1_degenerate, os, "K1_degenerate");
            os << ";";
            printCVMat(K2_degenerate, os, "K2_degenerate");
            os << ";";
            os << "inlRat_estimated;";
            os << "inlRat_GT;";
            os << "nrCorrs_filtered;";
            os << "nrCorrs_estimated;";
            os << "nrCorrs_GT;";
            tm.print(os, true);
            if(!addSequInfo.empty()) {
                os << ";";
                os << "addSequInfo";
            }
            os << endl;
        }else{
            os << R_diffAll << ";";
            R_diff.print(os);
            os << ";";
            os << t_angDiff << ";";
            os << t_distDiff << ";";
            t_elemDiff.print(os);
            os << ";";
            os << R_mostLikely_diffAll << ";";
            R_mostLikely_diff.print(os);
            os << ";";
            os << t_mostLikely_angDiff << ";";
            os << t_mostLikely_distDiff << ";";
            t_mostLikely_elemDiff.print(os);
            os << ";";
            printCVMat(R, os);
            os << ";";
            printCVMat(t, os);
            os << ";";
            printCVMat(R_mostLikely, os);
            os << ";";
            printCVMat(t_mostLikely, os);
            os << ";";
            printCVMat(R_GT, os);
            os << ";";
            printCVMat(t_GT, os);
            os << ";";
            os << R_GT_n_diffAll << ";";
            R_GT_n_diff.print(os);
            os << ";";
            os << t_GT_n_angDiff << ";";
            t_GT_n_elemDiff.print(os);
            os << ";";
            os << poseIsStable << ";";
            os << mostLikelyPose_stable << ";";
            os << poolSize << ";";
            os << ransac_agg << ";";
            K1_diff.print(os);
            os << ";";
            K2_diff.print(os);
            os << ";";
            printCVMat(K1, os);
            os << ";";
            printCVMat(K2, os);
            os << ";";
            printCVMat(K1_GT, os);
            os << ";";
            printCVMat(K2_GT, os);
            os << ";";
            printCVMat(K1_degenerate, os);
            os << ";";
            printCVMat(K2_degenerate, os);
            os << ";";
            os << inlRat_estimated << ";";
            os << inlRat_GT << ";";
            os << nrCorrs_filtered << ";";
            os << nrCorrs_estimated << ";";
            os << nrCorrs_GT << ";";
            tm.print(os, false);
            if(!addSequInfo.empty()) {
                os << ";";
                os << addSequInfo;
            }
            os << endl;
        }
    }
};

struct matchFilteringOpt{
    bool refineVFC;
    bool refineSOF;
    bool refineGMS;

    matchFilteringOpt():
    refineVFC(false),
    refineSOF(false),
    refineGMS(false){}
};
struct calibPars{
    std::string sequ_path;
    int matchData_idx;
    std::string hashMatchingPars;
    matchFilteringOpt mfo;
    bool autoTH;
    int refineMethod;
    bool refineRTold;
    int BART;
    std::string RobMethod;
    int Halign;
    poselib::ConfigUSAC cfg;
    double USACdegenTh;
    int USACInlratFilt;
    double th;
    poselib::ConfigPoseEstimation cfg_stereo;
    int evStepStereoStable;
    bool useOnlyStablePose;
    bool useMostLikelyPose;
    bool stereoRef;
    bool kneipInsteadBA;
    int accumCorrs;

    calibPars():
    sequ_path(""),
    matchData_idx(0),
    hashMatchingPars(""),
    mfo(matchFilteringOpt()),
    autoTH(false),
    refineMethod(0),
    refineRTold(false),
    BART(0),
    RobMethod(""),
    Halign(0),
    cfg(poselib::ConfigUSAC()),
    USACdegenTh(0.85),
    USACInlratFilt(1),
    th(0),
    cfg_stereo(poselib::ConfigPoseEstimation()),
    evStepStereoStable(0),
    useOnlyStablePose(false),
    useMostLikelyPose(false),
    stereoRef(false),
    kneipInsteadBA(false),
    accumCorrs(1){}
};

void cinfigureUSAC(poselib::ConfigUSAC &cfg,
	int cfgUSACnr[6],
	double USACdegenTh,
	const cv::Mat &K0,
	const cv::Mat &K1,
	cv::Size imgSize,
	std::vector<cv::KeyPoint> *kp1,
	std::vector<cv::KeyPoint> *kp2,
	std::vector<cv::DMatch> *finalMatches,
	double th_pix_user,
	int USACInlratFilt);
bool loadFeatureName(const string &filename,
                     const int parSetNr,
                     string &descriptorName,
                     string &keyPName,
                     string &matchesSubFolder);
bool loadImgSize(const string &filename,
                 cv::Size &imgSize);
size_t getHashCalibPars(const calibPars &cp);
bool genOutFileName(const std::string &path,
                    const calibPars &cp,
                    std::string &filename);
bool getNrEntriesYAML(const std::string &filename, const string &buzzword, int &nrEntries);
bool writeResultsOverview(const string &filename,
                          const calibPars &cp,
                          const string &resultsFileName);
FileStorage& operator << (FileStorage& fs, bool &value);
void writeTestingParameters(cv::FileStorage &fs,
                            const calibPars &cp);
bool writeResultsDisk(const std::vector<algorithmResult> &ar, const string &filename);

int SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
{
//    testing::internal::FilePath program(argv[0]);
//    testing::internal::FilePath program_dir = program.RemoveFileName();
//    testing::internal::FilePath data_path =
//            testing::internal::FilePath::ConcatPaths(program_dir,testing::internal::FilePath("imgs//stereo"));

    cmd.setIntroductoryDescription("Interface for testing the autocalibration with virtual generated test data.");
    //define error codes
    cmd.addErrorCode(0, "Success");
    cmd.addErrorCode(1, "Error");

    cmd.setHelpOption("h", "help","<Shows this help message.>");
    cmd.defineOption("sequ_path", "<Path to the single sequence frame data that contains the camera parameters.>",
            ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("matchData_idx", "<Index of the used matches of a sequence to load the corresponding matches "
                                    " using the information in file matchInfos.>",
                     ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("ovf_ext", "<Extension (yaml/xml/yaml.gz/xml.gz) used to store the testing data.>",
                     ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("output_path", "<Output path for storing the results of testing.>",
                     ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);

    cmd.defineOption("v", "<Verbose value [Default=0].\n "
                          "0\t Display nothing\n "
                          "1\t Display only pose\n "
                          "2\t Display pose & pose estimation time\n "
                          "3\t Display pose and pose estimation & refinement times\n "
                          "4\t Display all available information.>",
                          ArgvParser::OptionRequiresValue);

    cmd.defineOption("refineVFC", "<If provided, the result from the matching algorithm is refined with VFC>",
            ArgvParser::NoOptionAttribute);
    cmd.defineOption("refineSOF", "<If provided, the result from the matching algorithm is refined with SOF>",
            ArgvParser::NoOptionAttribute);
    cmd.defineOption("refineGMS", "<If provided, the result from the matching algorithm is refined with GMS>",
            ArgvParser::NoOptionAttribute);

    cmd.defineOption("noPoseDiff", "<If provided, the calculation of the difference to the given pose is disabled.>",
            ArgvParser::NoOptionAttribute);
    cmd.defineOption("autoTH", "<If provided, the threshold for estimating the pose is automatically adapted to "
                               "the data. This methode always uses ARRSAC with subsequent refinement.>",
                               ArgvParser::NoOptionAttribute);
    cmd.defineOption("refineRT", "<If provided, the pose (R, t) is linearly refined using one of the following "
                                 "options. It consists of a combination of 2 digits [Default=00]:\n "
        "1st digit - choose a refinement algorithm:"
        "\n 0\t no refinement"
        "\n 1\t 8 point algorithm with a pseudo-huber cost-function (old version). Here, the second digit has no effect."
        "\n 2\t 8 point algorithm"
        "\n 3\t Nister"
        "\n 4\t Stewenius"
        "\n 5\t Kneip's Eigen solver is applied on the result (the Essential matrix or for USAC: R,t) "
        "of RANSAC, ARRSAC, or USAC directly."
        "\n 6\t Kneip's Eigen solver is applied after extracting R & t and triangulation. This option can be "
        "seen as an alternative to bundle adjustment (BA)."
        "\n 2nd digit - choose a weighting function:"
        "\n 0\t Don't use weights"
        "\n 1\t Torr weights (ref: torr dissertation, eqn. 2.25)"
        "\n 2\t Pseudo-Huber weights"
        ">", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("BART", "<If provided, the pose (R, t) is refined using bundle adjustment (BA). "
                             "Try using the option --refineRT in addition to BA. This can lead to a better "
                             "solution (if --autoTH is enabled, --refineRT is always used). "
                             "The following options are available:\n 1\t BA for extrinsics only "
                             "(including structure)\n 2\t BA for extrinsics and intrinsics "
                             "(including structure)>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("RobMethod", "<Specifies the method for the robust estimation of the essential "
                                  "matrix [Default=USAC]. The following options are available:\n "
                                  "USAC\n ARRSAC\n RANSAC\n LMEDS>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("Halign", "<If provided, the pose is estimated using homography alignment. "
                               "Thus, multiple homographies are estimated using ARRSAC. The following options are "
                               "available:\n "
                               "1\t Estimate homographies without a variable threshold\n "
                               "2\t Estimate homographies with a variable threshold>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("cfgUSAC", "<Specifies parameters for USAC. It consists of a combination of 6 "
                                "digits [Default=311225]. "
        "In the following the options for every digit are explained:\n "
        "1st digit:\n "
        "0\t Use default paramters for SPRT\n "
        "1\t Automatic estimation of SPRT delta\n "
        "2\t Automatic estimation of SPRT epsilon (only without option refineVFC and refineGMS)\n "
        "3\t Automatic estimation of SPRT delta and epsilon\n "
        "2nd digit:\n "
        "0\t Use default paramter for PROSAC beta\n "
        "1\t Automatic estimation of PROSAC beta (uses SPRT delta)\n "
        "3rd digit:\n "
        "0\t Disable prevalidation of samples\n "
        "1\t Enable prevalidation of samples\n "
        "4th digit:\n "
        "0\t Disable degeneracy check\n "
        "1\t Use QDEGSAC for checking degeneracy\n "
        "2\t Use USACs internal degeneracy check\n "
        "5th digit:\n "
        "0\t Estimator: Nister\n "
        "1\t Estimator: Kneip's Eigen solver\n "
        "2\t Estimator: Stewenius\n "
        "6th digit:\n "
        "0\t Inner refinement alg: 8pt with Torr weights\n "
        "1\t Inner refinement alg: 8pt with pseudo-huber weights\n "
        "2\t Inner refinement alg: Kneip's Eigen solver\n "
        "3\t Inner refinement alg: Kneip's Eigen solver with Torr weights\n "
        "4\t Inner refinement alg: Stewenius\n "
        "5\t Inner refinement alg: Stewenius with pseudo-huber weights\n "
        "6\t Inner refinement alg: Nister\n "
        "7\t Inner refinement alg: Nister with pseudo-huber weights>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("USACdegenTh", "<Decision threshold on the inlier ratios between Essential matrix and the "
                                    "degenerate configuration (only rotation) to decide if the solution is "
                                    "degenerate or not [Default=0.85]. It is only used for the internal "
                                    "degeneracy check of USAC (4th digit of option cfgUSAC = 2)>",
                     ArgvParser::OptionRequiresValue);
	cmd.defineOption("USACInlratFilt", "<Specifies which filter is used on the matches to estimate an initial "
                                    "inlier ratio for USAC. Choose 0 for GMS and 1 for VFC [Default].>",
                                    ArgvParser::OptionRequiresValue);
	cmd.defineOption("th", "<Inlier threshold to check if a match corresponds to a model. [Default=0.8]>",
	        ArgvParser::OptionRequiresValue);
	cmd.defineOption("compInitPose", "<If provided, the estimated pose is compared to the given pose (Ground Truth).>",
	        ArgvParser::NoOptionAttribute);
	
	cmd.defineOption("stereoRef", "<If provided, the algorithm assums a stereo configuration and refines the "
                               "pose using multiple image pairs.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("evStepStereoStable", "<For stereo refinement: If the option stereoRef is provided and "
                                        "the estimated pose is stable, this option specifies the number of "
                                        "image pairs that are skipped until a new evaluation is performed. "
                                        "A value of 0 disables this feature [Default].>",
                                        ArgvParser::OptionRequiresValue);
	cmd.defineOption("useOnlyStablePose", "<For stereo refinement: If provided, and the option stereoRef is enabled, "
                                       "only a stable pose is used for rectification after the first stable pose is "
                                       "available. For estimations which do not produce a stable pose, the last "
                                       "stable pose is used. If the real pose is expected to change often, "
                                       "this option should not be used.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("useMostLikelyPose", "<For stereo refinement: If provided, the most likely correct pose over "
                                       "the last poses is preferred (if it is stable) instead of the actual pose.>",
                                       ArgvParser::NoOptionAttribute);
	
	cmd.defineOption("refineRT_stereo", "<For stereo refinement: Linear refinement of the pose using all "
                                     "correspondences from the pool with one of the following options. "
                                     "It consists of a combination of 2 digits [Default=42]:\n "
        "1st digit - choose a refinement algorithm:"
        "\n 1\t 8 point algorithm with a pseudo-huber cost-function (old version). Here, the second digit has no effect."
        "\n 2\t 8 point algorithm"
        "\n 3\t Nister"
        "\n 4\t Stewenius"
        "\n 5\t Kneip's Eigen solver is applied on the result (the Essential matrix or for USAC: R,t) "
        "of RANSAC, ARRSAC, or USAC directly."
        "\n 6\t Kneip's Eigen solver is applied after extracting R & t and triangulation. "
        "This option can be seen as an alternative to bundle adjustment (BA)."
        "\n 2nd digit - choose a weighting function:"
        "\n 0\t Don't use weights"
        "\n 1\t Torr weights (ref: torr dissertation, eqn. 2.25)"
        "\n 2\t Pseudo-Huber weights"
		">", ArgvParser::OptionRequiresValue);
	cmd.defineOption("BART_stereo", "<For stereo refinement: If provided, the pose (R, t) is refined using bundle "
                                 "adjustment (BA). The following options are available:\n "
                                 "1\t BA for extrinsics only (including structure)\n "
                                 "2\t BA for extrinsics and intrinsics (including structure)>",
                                 ArgvParser::OptionRequiresValue);
	cmd.defineOption("minStartAggInlRat", "<For stereo refinement: Minimum inlier ratio [Default=0.2] at robust "
                                       "estimation to start correspondence aggregation.>",
                                       ArgvParser::OptionRequiresValue);
	cmd.defineOption("relInlRatThLast", "<For stereo refinement: Maximum relative change of the inlier ratio "
                                     "between image pairs to check by a robsut method if the pose "
                                     "changed [Default=0.35].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("relInlRatThNew", "<For stereo refinement: Maximum relative change [Default=0.2] between "
                                    "the inlier ratio with the last E and the new robustly estimated E on "
                                    "the new image pair to check if the pose has really changed or if only "
                                    "the image pair qulity is very bad (Only if relInlRatThLast does not hold).>",
                                    ArgvParser::OptionRequiresValue);
	cmd.defineOption("minInlierRatSkip", "<For stereo refinement: Maximum inlier ratio [Default=0.38] using the "
                                      "new robustly estimated E to decide if the image pair quality is too "
                                      "bad (Only if relInlRatThNew does not hold and minInlierRatioReInit is "
                                      "not reached). Below this threshold, a fall-back threshold estimated by "
                                      "relMinInlierRatSkip and the inlier ratio of the last image pair can be "
                                      "used, if the resulting threshold is smaller minInlierRatSkip.>",
                                      ArgvParser::OptionRequiresValue);
	cmd.defineOption("relMinInlierRatSkip", "<For stereo refinement: Multiplication factor on the inlier "
                                         "ratio [Default=0.7] from the last image pair compared to the new "
                                         "robust estimated one to decide if the new image pair quality is "
                                         "too bad. minInlierRatSkip also influences the decision.>",
                                         ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxSkipPairs", "<For stereo refinement: Number of consecutive image pairs [Default=5] "
                                  "where a change in pose or a bad pair was detected until the system is "
                                  "reinitialized.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minInlierRatioReInit", "<For stereo refinement: Minimum inlier ratio [Default=0.67] of the "
                                          "new robust estimation after a change in pose was detected to immediately "
                                          "reinitialize the system (Only if relInlRatThNew does not hold).>",
                                          ArgvParser::OptionRequiresValue);
	cmd.defineOption("minPtsDistance", "<For stereo refinement: Minimum distance [Default=3.0] between "
                                    "correspondences in the pool (holding the correspondences of the last "
                                    "image pairs).>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxPoolCorrespondences", "<For stereo refinement: Maximum number of correspondences in the "
                                            "pool [Default=30000].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minContStablePoses", "<For stereo refinement: Minimum number of poses that must be very "
                                        "similar in terms of their geometric distance to detect stability "
                                        "[Default=3].>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("absThRankingStable", "<For stereo refinement: Maximum normalized error range difference "
                                        "between image pairs to detect stability [Default=0.075]. "
                                        "This normalized error is defined as pose_distance_rating = "
                                        "1.0 - Pose_Distance_to_all_Poses_gravity_center / max_dist_from_center. "
                                        "absThRankingStable defines the distance region arround the actual pose "
                                        "based on pose_distance_rating +- absThRankingStable. If the maximum pool "
                                        "size is reached and no stability was reached, a different measure based "
                                        "on reprojection error statistics from frame to frame is used (as fall-back) "
                                        "to determine if the computed pose is stable.>",
                                        ArgvParser::OptionRequiresValue);
	cmd.defineOption("useRANSAC_fewMatches", "<For stereo refinement: If provided, RANSAC for robust estimation "
                                          "if less than 100 matches are available is used.>",
                                          ArgvParser::NoOptionAttribute);
	cmd.defineOption("checkPoolPoseRobust", "<For stereo refinement: After this number of iterations [Default=3] "
                                         "or new image pairs, robust estimation is performed on the pool "
                                         "correspondences. The number automatically grows exponentially "
                                         "after each robust estimation. Options:"
		"\n 0\t Disabled"
		"\n 1\t Robust estimation is used instead of refinement."
		"\n 2-20\t see above>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("minNormDistStable", "<For stereo refinement: Minimum normalized distance [Default=0.5] "
                                       "to the center of gravity of all valid poses to detect stability.>",
                                       ArgvParser::OptionRequiresValue);
	cmd.defineOption("raiseSkipCnt", "<For stereo refinement: If provided, the value of maxSkipPairs is increased "
                                  "after a specific number of stable consecutive poses was detected [Default=00]. "
                                  "The following options are available:\n "
		"1st digit - Factor to increase maxSkipPairs:"
		"\n 0\t Disable [Default]"
		"\n 1-9\t Increase maxSkipPairs to std::ceil(maxSkipPairs * (1.0 + (1st digit) * 0.25)) if a specific "
  "number of stable consecutive poses was detected (defined by 2nd digit)."
		"\n 2nd digit - Number of stable consecutive poses to increase maxSkipPairs:"
		"\n 0-9\t nr = (2nd digit) + 1"
		">", ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxRat3DPtsFar", "<For stereo refinement: Maximum ratio [Default=0.5] of 3D points for "
                                    "which their z-value is very large (maxDist3DPtsZ x baseline) compared "
                                    "to the number of all 3D points. Above this threshold, a pose cannot "
                                    "be marked as stable using only a threshold on the Sampson error ranges "
                                    "(see absThRankingStable).>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("maxDist3DPtsZ", "<Maximum value for the z-coordinates of 3D points [Default=50.0] "
                                   "to be included into BA. Moreover, this value influences the decision "
                                   "if a pose is marked as stable during stereo refinement (see maxRat3DPtsFar).>",
                                   ArgvParser::OptionRequiresValue);
    cmd.defineOption("useGTCamMat", "<If provided, the GT camera matrices are always used and the distorted ones "
                                    "are ignored.>",
                     ArgvParser::NoOptionAttribute);
    cmd.defineOption("addSequInfo", "<Optional additional information (string) about the used test sequence or"
                                    " matches that will be stored in a seperate column in the results csv-file.>",
                     ArgvParser::OptionRequiresValue);
    cmd.defineOption("accumCorrs", "<If provided, correspondences are aggregated over a given number of stereo frames "
                                   "in a sliding window scheme "
                                   "(e.g. 4 frames: 1st frame: aggregation of matches from 1 frame, "
                                   "2nd frame: aggregation of matches from 2 frames, "
                                   "3rd frame: aggregation of matches from 3 frames, "
                                   "4th frame: aggregation of matches from 4 frames, "
                                   "5th frame: aggregation of matches from last 4 frames, ...,"
                                   "nth frame: aggregation of matches from last 4 frames). "
                                   "This option is ignored if the option stereoRef is provided.>",
                     ArgvParser::OptionRequiresValue);

    /// finally parse and handle return codes (display help etc...)
    int result = -1;
    result = cmd.parse(argc, argv);
    if (result != ArgvParser::NoParserError)
    {
        std::cerr << cmd.parseErrorDescription(result) << endl;
    }

    return result;
}

bool startEvaluation(ArgvParser& cmd)
{
    string matchData_idx_str, output_path, ovf_ext, ovf_ext1part, addSequInfo;
    string show_str;
    string cfgUSAC;
    string refineRT, refineRT_stereo;
    string descrName, kpNameM;
    bool noPoseDiff = true, refineRTold_stereo = false;
    bool useGTCamMat = false;
    int BART_stereo = 0;
    int err, verbose;
    vector<string> filenamesRt, filenamesMatches;
    int cfgUSACnr[6] = {3,1,1,2,2,5};
    int refineRTnr[2] = { 0,0 }, refineRTnr_stereo[2] = { 4,2 };
    bool kneipInsteadBA_stereo = false;
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
	bool accumCorrs_enabled = false;
	string raiseSkipCnt = "00";
	int raiseSkipCntnr[2] = { 0,0 };
	double maxRat3DPtsFar = 0.5;
	double maxDist3DPtsZ = 50.0;
	int accumCorrs = 0;
    calibPars cp = calibPars();

    if(cmd.foundOption("addSequInfo")){
        addSequInfo = cmd.optionValue("addSequInfo");
    }
    if(cmd.foundOption("accumCorrs")){
        accumCorrs = stoi(cmd.optionValue("accumCorrs"));
        if(accumCorrs > 1) {
            accumCorrs_enabled = true;
        }
    }

	//Load basic matching information
    cp.sequ_path = cmd.optionValue("sequ_path");
    ovf_ext = cmd.optionValue("ovf_ext");
    size_t extspPos = ovf_ext.find_last_of('.');
    if(extspPos != string::npos){
        ovf_ext1part = ovf_ext.substr(0, extspPos);
    }else{
        ovf_ext1part = ovf_ext;
    }
    extspPos = ovf_ext1part.find('.');
    if(extspPos != string::npos){
        ovf_ext1part = ovf_ext1part.substr(extspPos + 1);
    }
    std::transform(ovf_ext1part.begin(), ovf_ext1part.end(), ovf_ext1part.begin(), ::tolower);
    if((ovf_ext1part != "yaml") && (ovf_ext1part != "yml") && (ovf_ext1part != "xml")){
        cerr << "Invalid extension of test data files" << endl;
        exit(1);
    }

    matchData_idx_str = cmd.optionValue("matchData_idx");
    cp.matchData_idx = stoi(matchData_idx_str);
    testing::internal::FilePath sequ_pathG(cp.sequ_path);
    if(!sequ_pathG.DirectoryExists()){
        cerr << "Given main directory does not exist." << endl;
        exit(1);
    }
    testing::internal::FilePath sequFullDirG =
            testing::internal::FilePath::MakeFileName(sequ_pathG,
                    testing::internal::FilePath("matchInfos"),
                                                                0,
                                                                ovf_ext1part.c_str());
    if(!sequFullDirG.FileOrDirectoryExists()){
        cerr << "Unable to locate file matchInfos" << endl;
        exit(1);
    }
    const string &matchesOVFile = sequFullDirG.string();
    if(!loadFeatureName(matchesOVFile,
                        cp.matchData_idx,
                        descrName,
                        kpNameM,
                        cp.hashMatchingPars)){
        cerr << "Unable to load matching information" << endl;
    }
    testing::internal::FilePath matchFullDirG =
            testing::internal::FilePath::ConcatPaths(sequ_pathG,testing::internal::FilePath(cp.hashMatchingPars));
    if(!matchFullDirG.DirectoryExists()){
        cerr << "Directory with matches does not exist." << endl;
        exit(1);
    }
    const string &matchesPath = matchFullDirG.string();


    cp.mfo.refineVFC = cmd.foundOption("refineVFC");
    cp.mfo.refineSOF = cmd.foundOption("refineSOF");
    cp.mfo.refineGMS = cmd.foundOption("refineGMS");

    useGTCamMat = cmd.foundOption("useGTCamMat");

    noPoseDiff = cmd.foundOption("noPoseDiff");
    cp.autoTH = cmd.foundOption("autoTH");

	compInitPose = cmd.foundOption("compInitPose");

	cp.stereoRef = cmd.foundOption("stereoRef");
    cp.useOnlyStablePose = cmd.foundOption("useOnlyStablePose");
    cp.useMostLikelyPose = cmd.foundOption("useMostLikelyPose");

	if (cmd.foundOption("evStepStereoStable"))
	{
		cp.evStepStereoStable = stoi(cmd.optionValue("evStepStereoStable"));
		if (cp.evStepStereoStable < 0 || cp.evStepStereoStable > 1000)
		{
			std::cout << "The number of image pairs skipped " << cp.evStepStereoStable <<
			" before estimating a new pose is out of range. Using default value of 0." << std::endl;
            cp.evStepStereoStable = 0;
		}
	}
	else {
        cp.evStepStereoStable = 0;
    }

    if(cmd.foundOption("Halign"))
    {
        cp.Halign = stoi(cmd.optionValue("Halign"));
        if((cp.Halign < 0) || (cp.Halign > 2))
        {
            std::cerr << "The specified option for homography alignment (Halign) is not available. Exiting." << endl;
            exit(1);
        }
    }
    else
    {
        cp.Halign = 0;
    }

    if(cmd.foundOption("BART"))
    {
        cp.BART = stoi(cmd.optionValue("BART"));
        if((cp.BART < 0) || (cp.BART > 2))
        {
            std::cerr << "The specified option for bundle adjustment (BART) is not available. Exiting." << endl;
            exit(1);
        }
    }
    else {
        cp.BART = 0;
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
        cp.RobMethod = cmd.optionValue("RobMethod");
    }
    else {
        cp.RobMethod = "USAC";
    }

	if (cmd.foundOption("USACInlratFilt"))
	{
		cp.USACInlratFilt = stoi(cmd.optionValue("USACInlratFilt"));
		if (cp.USACInlratFilt < 0 || cp.USACInlratFilt > 1)
		{
			std::cout << "The specified option forUSACInlratFilt is not available. Changing to default: VFC filtering" << std::endl;
            cp.USACInlratFilt = 1;
		}
	}
	else {
        cp.USACInlratFilt = 1;
    }

    if((cp.RobMethod != "ARRSAC") && cp.autoTH)
    {
        std::cout << "With option 'autoTH' only ARRSAC is supported. Using ARRSAC!" << endl;
    }

    if((cp.RobMethod != "ARRSAC") && cp.Halign)
    {
        std::cout << "With option 'Halign' only ARRSAC is supported. Using ARRSAC!" << endl;
    }

    if(cp.autoTH && cp.Halign)
    {
        std::cerr << "The options 'autoTH' and 'Halign' are mutually exclusive. Choose only one of them. Exiting." << endl;
        exit(1);
    }

    if (cmd.foundOption("th"))
    {
        cp.th = std::stod(cmd.optionValue("th"));
        if (cp.th < 0.1)
        {
            std::cout << "User specific threshold of " << cp.th << " is too small. Setting it to 0.1" << endl;
            cp.th = 0.1;
        }
        else if (cp.th > 5.0)
        {
            std::cout << "User specific threshold of " << cp.th << " is too large. Setting it to 5.0" << endl;
            cp.th = 5.0;
        }
    }
    else
        cp.th = PIX_MIN_GOOD_TH;

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
    cp.refineMethod = poselib::RefinePostAlg::PR_NO_REFINEMENT;
    if (refineRTnr[0])
    {
        switch (refineRTnr[0])
        {
        case(1):
            cp.refineRTold = true;
            break;
        case(2):
            cp.refineMethod = poselib::RefinePostAlg::PR_8PT;
            break;
        case(3):
            cp.refineMethod = poselib::RefinePostAlg::PR_NISTER;
            break;
        case(4):
            cp.refineMethod = poselib::RefinePostAlg::PR_STEWENIUS;
            break;
        case(5):
            cp.refineMethod = poselib::RefinePostAlg::PR_KNEIP;
            break;
        case(6):
            cp.refineMethod = poselib::RefinePostAlg::PR_KNEIP;
            cp.kneipInsteadBA = true;
            break;
        default:
            break;
        }

        switch (refineRTnr[1])
        {
        case(0):
            cp.refineMethod = (cp.refineMethod | poselib::RefinePostAlg::PR_NO_WEIGHTS);
            break;
        case(1):
            cp.refineMethod = (cp.refineMethod | poselib::RefinePostAlg::PR_TORR_WEIGHTS);
            break;
        case(2):
            cp.refineMethod = (cp.refineMethod | poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS);
            break;
        default:
            break;
        }
    }

	//For stereo refinement
	if (cmd.foundOption("refineRT_stereo"))
		refineRT_stereo = cmd.optionValue("refineRT_stereo");
	else
		refineRT_stereo = "42";
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
		std::cout << "Option refineRT_stereo is corrupt! Taking default values." << endl;
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
			std::cerr << "The specified option for bundle adjustment (BART_stereo) is not available. Exiting." << endl;
			exit(1);
		}
	}
	else
		BART_stereo = 0;

	if (cmd.foundOption("minStartAggInlRat"))
	{
		minStartAggInlRat = std::stod(cmd.optionValue("minStartAggInlRat"));
		if (minStartAggInlRat < 0.01 || minStartAggInlRat >= 1.0)
		{
			std::cout << "Value for minStartAggInlRat out of range. Using default value." << endl;
			minStartAggInlRat = 0.2;
		}
	}
	else
		minStartAggInlRat = 0.2;

	if (cmd.foundOption("relInlRatThLast"))
	{
		relInlRatThLast = std::stod(cmd.optionValue("relInlRatThLast"));
		if (relInlRatThLast < 0.01 || relInlRatThLast >= 1.0)
		{
			std::cout << "Value for relInlRatThLast out of range. Using default value." << endl;
			relInlRatThLast = 0.35;
		}
	}
	else
		relInlRatThLast = 0.35;

	if (cmd.foundOption("relInlRatThNew"))
	{
		relInlRatThNew = std::stod(cmd.optionValue("relInlRatThNew"));
		if (relInlRatThNew < 0.01 || relInlRatThNew >= 1.0)
		{
			std::cout << "Value for relInlRatThNew out of range. Using default value." << endl;
			relInlRatThNew = 0.2;
		}
	}
	else
		relInlRatThNew = 0.2;

	if (cmd.foundOption("minInlierRatSkip"))
	{
		minInlierRatSkip = std::stod(cmd.optionValue("minInlierRatSkip"));
		if (minInlierRatSkip < 0.01 || minInlierRatSkip >= 1.0)
		{
			std::cout << "Value for minInlierRatSkip out of range. Using default value." << endl;
			minInlierRatSkip = 0.38;
		}
	}
	else
		minInlierRatSkip = 0.38;

	if (cmd.foundOption("relMinInlierRatSkip"))
	{
		relMinInlierRatSkip = std::stod(cmd.optionValue("relMinInlierRatSkip"));
		if (relMinInlierRatSkip < 0.01 || relMinInlierRatSkip >= 1.0)
		{
			std::cout << "Value for relMinInlierRatSkip out of range. Using default value." << endl;
			relMinInlierRatSkip = 0.7;
		}
	}
	else
		relMinInlierRatSkip = 0.7;

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
	else
		maxSkipPairs = 5;

	if (cmd.foundOption("minInlierRatioReInit"))
	{
		minInlierRatioReInit = std::stod(cmd.optionValue("minInlierRatioReInit"));
		if (minInlierRatioReInit < 0.01 || minInlierRatioReInit >= 1.0)
		{
			std::cout << "Value for minInlierRatioReInit out of range. Using default value." << endl;
			minInlierRatioReInit = 0.6;
		}
	}
	else
		minInlierRatioReInit = 0.6;

	if (cmd.foundOption("minPtsDistance"))
	{
		minPtsDistance = std::stof(cmd.optionValue("minPtsDistance"));
		if (minPtsDistance < 1.42f || minPtsDistance > 16.f)
		{
			std::cout << "Value for minPtsDistance out of range. Using default value." << endl;
			minPtsDistance = 3.f;
		}
	}
	else
		minPtsDistance = 3.f;

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
	else
		maxPoolCorrespondences = 30000;

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
	else
		minContStablePoses = 3;

	if (cmd.foundOption("absThRankingStable"))
	{
		absThRankingStable = std::stod(cmd.optionValue("absThRankingStable"));
		if (absThRankingStable < 0.01 || absThRankingStable >= 1.0)
		{
			std::cout << "Value for absThRankingStable out of range. Using default value." << endl;
			absThRankingStable = 0.075;
		}
	}
	else
		absThRankingStable = 0.075;

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
	else
		checkPoolPoseRobust = 3;

	if (cmd.foundOption("minNormDistStable"))
	{
		minNormDistStable = std::stod(cmd.optionValue("minNormDistStable"));
		if (minNormDistStable < 0.01 || minNormDistStable >= 1.0)
		{
			std::cout << "Value for minNormDistStable out of range. Using default value." << endl;
			minNormDistStable = 0.5;
		}
	}
	else
		minNormDistStable = 0.5;

	if (cmd.foundOption("raiseSkipCnt"))
		raiseSkipCnt = cmd.optionValue("raiseSkipCnt");
	else
		raiseSkipCnt = "00";
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
	else
		maxRat3DPtsFar = 0.5;

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
        if (cfgUSACnr[0] > 1 && (cp.mfo.refineVFC || cp.mfo.refineGMS))
        {
            std::cout << "Impossible to estimate epsilon for SPRT if option refineVFC or refineGMS is enabled! "
                         "Disabling option refineVFC and refineGMS!" << endl;
            cp.mfo.refineVFC = false;
            cp.mfo.refineGMS = false;
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
        cp.USACdegenTh = std::stod(cmd.optionValue("USACdegenTh"));
    else
        cp.USACdegenTh = 0.85;

    if(cmd.foundOption("v"))
    {
        verbose = stoi(cmd.optionValue("v"));
    }
    else
        verbose = 0;

    output_path = cmd.optionValue("output_path");
    testing::internal::FilePath output_pathG(output_path);
    if(!output_pathG.DirectoryExists()){
        cerr << "Output directory does not exist" << endl;
        exit(1);
    }

    err = loadImageSequence(cp.sequ_path, "sequSingleFrameData_", filenamesRt);
    if(err || filenamesRt.empty())
    {
        std::cerr << "Could not find camera extrinsics! Exiting." << endl;
        exit(1);
    }
    err = loadImageSequence(matchesPath, "matchSingleFrameData_", filenamesMatches);
    if(err || filenamesMatches.empty())
    {
        std::cerr << "Could not find camera extrinsics! Exiting." << endl;
        exit(1);
    }
    if(filenamesRt.size() != filenamesMatches.size()){
        cerr << "Number of files for sequences and matches are not equal" << endl;
        exit(1);
    }

    testing::internal::FilePath sequParFG =
            testing::internal::FilePath::MakeFileName(sequ_pathG,
                                                                testing::internal::FilePath("sequPars"),
                                                                0,
                                                                ovf_ext.c_str());
    if(!sequParFG.FileOrDirectoryExists()){
        cerr << "Unable to locate file sequPars" << endl;
        exit(1);
    }
    cv::Size imgSize;
    if(!loadImgSize(sequParFG.string(), imgSize)){
        cerr << "Unable to load image size" << endl;
        exit(1);
    }

	cv::Mat R1, t1, K0, K1, dist0_8, dist1_8;
    dist0_8 = Mat::zeros(1, 8, CV_64FC1);
    dist1_8 = Mat::zeros(1, 8, CV_64FC1);

	std::unique_ptr<poselib::StereoRefine> stereoObj;
	if (cp.stereoRef)
	{
        sequMatches sm;
        if(!readCamParsFromDisk(filenamesRt[0], sm)){
            cerr << "Unable to read camera parameters" << endl;
            exit(1);
        }
        if(!readMatchesFromDisk(filenamesMatches[0], sm)){
            cerr << "Unable to read camera parameters" << endl;
            exit(1);
        }
        if(useGTCamMat){
            K0 = sm.K1.clone();
            K1 = sm.K2.clone();
        }else {
            K0 = sm.actKd1.clone();
            K1 = sm.actKd2.clone();
        }

		cp.cfg_stereo.dist0_8 = &dist0_8;
		cp.cfg_stereo.dist1_8 = &dist1_8;
		cp.cfg_stereo.K0 = &K0;
		cp.cfg_stereo.K1 = &K1;
		cp.cfg_stereo.keypointType = kpNameM;
		cp.cfg_stereo.descriptorType = descrName;
		cp.cfg_stereo.th_pix_user = cp.th;
		cp.cfg_stereo.verbose = (verbose > 0) ? (verbose + 2) : 0;
		cp.cfg_stereo.Halign = cp.Halign;
		cp.cfg_stereo.autoTH = cp.autoTH;
		cp.cfg_stereo.BART = cp.BART;
		cp.cfg_stereo.kneipInsteadBA = cp.kneipInsteadBA;
		cp.cfg_stereo.refineMethod = cp.refineMethod;
		cp.cfg_stereo.refineRTold = cp.refineRTold;
		cp.cfg_stereo.RobMethod = cp.RobMethod;

		cp.cfg_stereo.refineMethod_CorrPool = refineMethod_stereo;
		cp.cfg_stereo.refineRTold_CorrPool = refineRTold_stereo;
		cp.cfg_stereo.kneipInsteadBA_CorrPool = kneipInsteadBA_stereo;
		cp.cfg_stereo.BART_CorrPool = BART_stereo;
		cp.cfg_stereo.minStartAggInlRat = minStartAggInlRat;
		cp.cfg_stereo.relInlRatThLast = relInlRatThLast;
		cp.cfg_stereo.relInlRatThNew = relInlRatThNew;
		cp.cfg_stereo.minInlierRatSkip = minInlierRatSkip;
		cp.cfg_stereo.relMinInlierRatSkip = relMinInlierRatSkip;
		cp.cfg_stereo.maxSkipPairs = maxSkipPairs;
		cp.cfg_stereo.minInlierRatioReInit = minInlierRatioReInit;
		cp.cfg_stereo.minPtsDistance = minPtsDistance;
		cp.cfg_stereo.maxPoolCorrespondences = maxPoolCorrespondences;
		cp.cfg_stereo.minContStablePoses = minContStablePoses;
		cp.cfg_stereo.absThRankingStable = absThRankingStable;
		cp.cfg_stereo.useRANSAC_fewMatches = useRANSAC_fewMatches;
		cp.cfg_stereo.checkPoolPoseRobust = checkPoolPoseRobust;
		cp.cfg_stereo.minNormDistStable = minNormDistStable;
		cp.cfg_stereo.raiseSkipCnt = (raiseSkipCntnr[0] | (raiseSkipCntnr[1] << 4));
		cp.cfg_stereo.maxRat3DPtsFar = maxRat3DPtsFar;
		cp.cfg_stereo.maxDist3DPtsZ = maxDist3DPtsZ;

		stereoObj.reset(new poselib::StereoRefine(cp.cfg_stereo, verbose > 0));
        accumCorrs_enabled = false;
	}else{
        cp.cfg_stereo.keypointType = kpNameM;
        cp.cfg_stereo.descriptorType = descrName;
        if(accumCorrs_enabled){
            cp.accumCorrs = accumCorrs;
        }
	}

    int failNr = 0;
	const int evStepStereoStable_tmp = cp.evStepStereoStable + 1;
	int evStepStereoStable_cnt = evStepStereoStable_tmp;
	cv::Mat R_stable, t_stable;
    chrono::high_resolution_clock::time_point t_1, t_2;
    vector<algorithmResult> ar;//(filenamesMatches.size(), algorithmResult());
    ar.reserve(filenamesMatches.size());
    for(size_t i = 0; i < filenamesMatches.size(); ++i){
        ar.emplace_back(algorithmResult(addSequInfo));
    }
    std::list<std::vector<cv::KeyPoint>> kp1_accum, kp2_accum;
    std::list<vector<cv::Point2f>> points1_accum, points2_accum;
    std::list<std::pair<int, int>> nrFeatures;
    std::list<std::vector<cv::DMatch>> matches_accum;
    std::list<std::vector<bool>> frameInliers_accum;
    for(int i = 0; i < (int)filenamesMatches.size(); i++)
    {
        //Load stereo configuration
        sequMatches sm;
        if(!readCamParsFromDisk(filenamesRt[i], sm)){
            cerr << "Unable to read camera parameters" << endl;
            exit(1);
        }
        if(!readMatchesFromDisk(filenamesMatches[i], sm)){
            cerr << "Unable to read camera parameters" << endl;
            exit(1);
        }
        ar[i].calcInlRatGT(sm.frameInliers);
        std::vector<cv::DMatch> finalMatches;
        std::vector<cv::KeyPoint> kp1 = sm.frameKeypoints1;
        std::vector<cv::KeyPoint> kp2 = sm.frameKeypoints2;
        sm.actT.copyTo(t1);
        sm.actR.copyTo(R1);

        //Perform filtering
        t_1 = chrono::high_resolution_clock::now();
        if(cp.mfo.refineVFC){
            if(filterWithVFC(kp1, kp2, sm.frameMatches, finalMatches)){
                cout << "Unable to filter matches using VFC" << endl;
                finalMatches = sm.frameMatches;
            }
            t_2 = chrono::high_resolution_clock::now();
            ar[i].tm.filtering = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
            ar[i].nrCorrs_filtered = (int)finalMatches.size();
        }else if(cp.mfo.refineGMS){
            if(filterMatchesGMS(kp1, imgSize, kp2, imgSize, sm.frameMatches, finalMatches) < 10){
                cout << "Unable to filter matches using GMS" << endl;
                finalMatches = sm.frameMatches;
            }
            t_2 = chrono::high_resolution_clock::now();
            ar[i].tm.filtering = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
            ar[i].nrCorrs_filtered = (int)finalMatches.size();
        }else if(cp.mfo.refineSOF){
            finalMatches = sm.frameMatches;
            matchinglib::filterMatchesSOF(kp1, kp2, imgSize, finalMatches);
            t_2 = chrono::high_resolution_clock::now();
            ar[i].tm.filtering = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
            ar[i].nrCorrs_filtered = (int)finalMatches.size();
        }else{
            finalMatches = sm.frameMatches;
        }

        //Pose estimation
        //-------------------------------
        double t_mea = 0, t_oa = 0;
		cv::Mat R, t;
		if (!cp.stereoRef)
		{
            if(useGTCamMat){
                sm.K1.copyTo(K0);
                sm.K2.copyTo(K1);
            }else {
                sm.actKd1.copyTo(K0);
                sm.actKd2.copyTo(K1);
            }
			//Extract coordinates from keypoints
			vector<cv::Point2f> points1, points2;
			for (auto &j : finalMatches)
			{
				points1.push_back(kp1[j.queryIdx].pt);
				points2.push_back(kp2[j.trainIdx].pt);
			}

			if (verbose > 3)
			{
				t_mea = (double)getTickCount(); //Start time measurement
			}

			//Transfer into camera coordinates
			poselib::ImgToCamCoordTrans(points1, K0);
			poselib::ImgToCamCoordTrans(points2, K1);

			if (verbose > 3)
			{
				t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
				std::cout << "Time for coordinate conversion & undistortion (2 imgs): " << t_mea << "ms" << endl;
				t_oa = t_mea;
			}

			if(accumCorrs_enabled){
                kp1_accum.push_back(kp1);
                kp2_accum.push_back(kp2);
                points1_accum.push_back(points1);
                points2_accum.push_back(points2);
                matches_accum.push_back(finalMatches);
                nrFeatures.emplace_back(std::make_pair((int)kp1.size(), (int)kp2.size()));
                frameInliers_accum.push_back(sm.frameInliers);
			    if (i >= accumCorrs){
                    nrFeatures.pop_front();
                    kp1_accum.pop_front();
                    kp2_accum.pop_front();
                    points1_accum.pop_front();
                    points2_accum.pop_front();
                    matches_accum.pop_front();
                    frameInliers_accum.pop_front();
			    }
			    ar[i].ransac_agg = (int)frameInliers_accum.size();
			    if(kp1_accum.size() > 1) {
			        auto kp1_it = kp1_accum.begin();
                    auto kp2_it = kp2_accum.begin();
                    auto p1_it = points1_accum.begin();
                    auto p2_it = points2_accum.begin();
                    auto m_it = matches_accum.begin();
                    auto nrf_it = nrFeatures.begin();
                    auto finl_it = frameInliers_accum.begin();
                    std::pair<int, int> zero_idx = *nrf_it;
                    std::vector<bool> frame_inl_tmp = *finl_it;
                    kp1 = *kp1_it;
                    kp2 = *kp2_it;
                    points1 = *p1_it;
                    points2 = *p2_it;
                    finalMatches = *m_it;
                    for (size_t j = 1; j < kp1_accum.size(); j++) {
                        kp1_it++;
                        kp2_it++;
                        p1_it++;
                        p2_it++;
                        m_it++;
                        nrf_it++;
                        finl_it++;
                        kp1.insert(kp1.end(), kp1_it->begin(), kp1_it->end());
                        kp2.insert(kp2.end(), kp2_it->begin(), kp2_it->end());
                        points1.insert(points1.end(), p1_it->begin(), p1_it->end());
                        points2.insert(points2.end(), p2_it->begin(), p2_it->end());
                        std::vector<cv::DMatch> match_tmp = *m_it;
                        for(auto &m:match_tmp){
                            m.queryIdx += zero_idx.first;
                            m.trainIdx += zero_idx.second;
                        }
                        finalMatches.insert(finalMatches.end(), match_tmp.begin(), match_tmp.end());
                        zero_idx.first += nrf_it->first;
                        zero_idx.second += nrf_it->second;
                        frame_inl_tmp.insert(frame_inl_tmp.end(), finl_it->begin(), finl_it->end());
                    }
                    ar[i].calcInlRatGT(frame_inl_tmp);
                }
			}

			if (verbose > 1)
			{
				t_mea = (double)getTickCount(); //Start time measurement
			}

			//Set up USAC paramters
			if (cp.RobMethod == "USAC")
			{
				cinfigureUSAC(cp.cfg,
					cfgUSACnr,
                              cp.USACdegenTh,
					K0,
					K1,
					imgSize,
					&kp1,
					&kp2,
					&finalMatches,
                              cp.th,
                              cp.USACInlratFilt);
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
			double pixToCamFact = 4.0 / (std::sqrt(2.0) *
			        (K0.at<double>(0, 0) + K0.at<double>(1, 1) + K1.at<double>(0, 0) + K1.at<double>(1, 1)));
			double th = cp.th * pixToCamFact; //Inlier threshold
            t_1 = chrono::high_resolution_clock::now();
			if (cp.autoTH)
			{
				int inlierPoints;
				poselib::AutoThEpi Eautoth(pixToCamFact);
				if (Eautoth.estimateEVarTH(p1, p2, E, mask, &th, &inlierPoints) != 0)
				{
#if FAIL_SKIP_REST
					failNr++;
#endif
					if ((float)failNr / (float)filenamesRt.size() < 0.5f)
					{
						std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
                        t_2 = chrono::high_resolution_clock::now();
                        ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                        sm.actR.copyTo(ar[i].R_GT);
                        sm.actT.copyTo(ar[i].t_GT);
                        sm.K1.copyTo(ar[i].K1_GT);
                        sm.K2.copyTo(ar[i].K2_GT);
                        if(useGTCamMat){
                            sm.K1.copyTo(ar[i].K1_degenerate);
                            sm.K2.copyTo(ar[i].K2_degenerate);
                        }else {
                            sm.actKd1.copyTo(ar[i].K1_degenerate);
                            sm.actKd2.copyTo(ar[i].K2_degenerate);
                        }
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
			else if (cp.Halign)
			{
				int inliers;
				if (poselib::estimatePoseHomographies(p1, p2, R, t, E, th, inliers, mask, false, cp.Halign > 1) != 0)
				{
#if FAIL_SKIP_REST
                    failNr++;
#endif
					if ((float)failNr / (float)filenamesRt.size() < 0.5f)
					{
						std::cout << "Homography alignment failed! Trying next pair." << endl;
                        t_2 = chrono::high_resolution_clock::now();
                        ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                        sm.actR.copyTo(ar[i].R_GT);
                        sm.actT.copyTo(ar[i].t_GT);
                        sm.K1.copyTo(ar[i].K1_GT);
                        sm.K2.copyTo(ar[i].K2_GT);
                        if(useGTCamMat){
                            sm.K1.copyTo(ar[i].K1_degenerate);
                            sm.K2.copyTo(ar[i].K2_degenerate);
                        }else {
                            sm.actKd1.copyTo(ar[i].K1_degenerate);
                            sm.actKd2.copyTo(ar[i].K2_degenerate);
                        }
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
				if (cp.RobMethod == "USAC")
				{
#if TESTOUT
				    cout << "Entering USAC" << endl;
#endif
					bool isDegenerate = false;
					Mat R_degenerate, inliers_degenerate_R;
					bool usacerror = false;
					if (cp.cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP
					|| cp.cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS)
					{
						if (estimateEssentialOrPoseUSAC(p1,
							p2,
							E,
							th,
                                                        cp.cfg,
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
                                                        cp.cfg,
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
#if FAIL_SKIP_REST
                        failNr++;
#endif
						if ((float)failNr / (float)filenamesRt.size() < 0.5f)
						{
							std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
                            t_2 = chrono::high_resolution_clock::now();
                            ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                            sm.actR.copyTo(ar[i].R_GT);
                            sm.actT.copyTo(ar[i].t_GT);
                            sm.K1.copyTo(ar[i].K1_GT);
                            sm.K2.copyTo(ar[i].K2_GT);
                            if(useGTCamMat){
                                sm.K1.copyTo(ar[i].K1_degenerate);
                                sm.K2.copyTo(ar[i].K2_degenerate);
                            }else {
                                sm.actKd1.copyTo(ar[i].K1_degenerate);
                                sm.actKd2.copyTo(ar[i].K2_degenerate);
                            }
							continue;
						}
						else
						{
							std::cout << "Estimation of essential matrix or undistortion or matching failed for "
							<< failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
							exit(1);
						}
					}
#if TESTOUT
                    cout << "Leaving USAC" << endl;
#endif
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
                        t_2 = chrono::high_resolution_clock::now();
                        ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                        ar[i].calcInlRatEstimated(mask);
                        ar[i].calcRTDiff(R_degenerate,
                                sm.actR,
                                cv::Mat::zeros(3,1, CV_64FC1),
                                sm.actT,
                                false,
                                !noPoseDiff && compInitPose && (verbose > 0));
                        sm.K1.copyTo(ar[i].K1_GT);
                        sm.K2.copyTo(ar[i].K2_GT);
                        if(useGTCamMat){
                            sm.K1.copyTo(ar[i].K1_degenerate);
                            sm.K2.copyTo(ar[i].K2_degenerate);
                        }else {
                            sm.actKd1.copyTo(ar[i].K1_degenerate);
                            sm.actKd2.copyTo(ar[i].K2_degenerate);
                        }
						continue;
					}
				}
				else
				{
					if (!poselib::estimateEssentialMat(E, p1, p2, cp.RobMethod, th, cp.refineRTold, mask))
					{
#if FAIL_SKIP_REST
                        failNr++;
#endif
						if ((float)failNr / (float)filenamesRt.size() < 0.5f)
						{
							std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
                            t_2 = chrono::high_resolution_clock::now();
                            ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                            sm.actR.copyTo(ar[i].R_GT);
                            sm.actT.copyTo(ar[i].t_GT);
                            sm.K1.copyTo(ar[i].K1_GT);
                            sm.K2.copyTo(ar[i].K2_GT);
                            if(useGTCamMat){
                                sm.K1.copyTo(ar[i].K1_degenerate);
                                sm.K2.copyTo(ar[i].K2_degenerate);
                            }else {
                                sm.actKd1.copyTo(ar[i].K1_degenerate);
                                sm.actKd2.copyTo(ar[i].K2_degenerate);
                            }
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
            t_2 = chrono::high_resolution_clock::now();
            ar[i].tm.robEstimationAndRef = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
			size_t nr_inliers = (size_t)cv::countNonZero(mask);
            if (verbose) {
                std::cout << "Number of inliers after robust estimation of E: " << nr_inliers << endl;
            }
            if(nr_inliers < 5){
#if FAIL_SKIP_REST
                failNr++;
#endif
                if ((float)failNr / (float)filenamesRt.size() < 0.5f)
                {
                    std::cout << "Estimation of essential matrix failed! Trying next pair." << endl;
                    t_2 = chrono::high_resolution_clock::now();
                    ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                    sm.actR.copyTo(ar[i].R_GT);
                    sm.actT.copyTo(ar[i].t_GT);
                    sm.K1.copyTo(ar[i].K1_GT);
                    sm.K2.copyTo(ar[i].K2_GT);
                    if(useGTCamMat){
                        sm.K1.copyTo(ar[i].K1_degenerate);
                        sm.K2.copyTo(ar[i].K2_degenerate);
                    }else {
                        sm.actKd1.copyTo(ar[i].K1_degenerate);
                        sm.actKd2.copyTo(ar[i].K2_degenerate);
                    }
                    continue;
                }
                else
                {
                    std::cout << "Estimation of essential matrix or undistortion or matching failed for " << failNr << " image pairs. Something is wrong with your data! Exiting." << endl;
                    exit(1);
                }
            }

			//Get R & t
            t_1 = chrono::high_resolution_clock::now();
			bool availableRT = false;
			if (cp.Halign)
			{
				R_kneip = R;
				t_kneip = t;
			}
			if (cp.Halign ||
				((cp.RobMethod == "USAC") && (cp.cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP ||
                        cp.cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS)))
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
			cv::Mat Q, mask_E = mask.clone();
			if (cp.Halign && ((cp.refineMethod & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT))
			{
				if(poselib::triangPts3D(R, t, p1, p2, Q, mask, maxDist3DPtsZ) <= 0){
				    if(verbose){
				        cout << "Triangulating points not successful." << endl;
				    }
				}
			}
			else
			{
				if (cp.refineRTold)
				{
					poselib::robustEssentialRefine(p1, p2, E, E, th / 10.0, 0, true, nullptr, nullptr, cv::noArray(), mask, 0);
					availableRT = false;
				}
				else if (((cp.refineMethod & 0xF) != poselib::RefinePostAlg::PR_NO_REFINEMENT) && !cp.kneipInsteadBA)
				{
					cv::Mat R_tmp, t_tmp;
					if (availableRT)
					{
						R_kneip.copyTo(R_tmp);
						t_kneip.copyTo(t_tmp);

						if (poselib::refineEssentialLinear(p1, p2, E, mask, cp.refineMethod, nr_inliers, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
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
					else if ((cp.refineMethod & 0xF) == poselib::RefinePostAlg::PR_KNEIP)
					{

						if (poselib::refineEssentialLinear(p1, p2, E, mask, cp.refineMethod, nr_inliers, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
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
						        cp.refineMethod,
						        nr_inliers,
						        cv::noArray(),
						        cv::noArray(),
						        th,
						        4, 2.0, 0.1, 0.15)) {
                            std::cout << "Refinement failed!" << std::endl;
                        }
					}
				}
                mask_E = mask.clone();

				if (!availableRT) {
#if TESTOUT
                    cout << "Entering triangulation" << endl;
#endif
                    if (poselib::getPoseTriangPts(E, p1, p2, R, t, Q, mask, maxDist3DPtsZ) <= 0) {
                        if (verbose) {
                            cout << "Triangulating points not successful." << endl;
                        }
                    }
#if TESTOUT
                    cout << "Leaving triangulation" << endl;
#endif
                }
				else{
					R = R_kneip;
					t = t_kneip;
					//if ((cp.BART > 0) && !cp.kneipInsteadBA)
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
            t_2 = chrono::high_resolution_clock::now();
            ar[i].tm.linRefinement = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();

			if (verbose > 1)
			{
				t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
				std::cout << "Time for pose estimation (includes possible linear refinement): " << t_mea << "ms" << endl;
				t_oa += t_mea;
			}

			if (verbose > 2)
			{
				t_mea = (double)getTickCount(); //Start time measurement
			}

			//Bundle adjustment
			bool useBA = true;
            t_1 = chrono::high_resolution_clock::now();
			if (cp.kneipInsteadBA)
			{
				cv::Mat R_tmp, t_tmp;
				R.copyTo(R_tmp);
				t.copyTo(t_tmp);
				if (poselib::refineEssentialLinear(p1, p2, E, mask, cp.refineMethod, nr_inliers, R_tmp, t_tmp, th, 4, 2.0, 0.1, 0.15))
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
					if (cp.BART > 0)
					{
						std::cout << "Trying bundle adjustment instead!" << std::endl;
						if(poselib::triangPts3D(R, t, p1, p2, Q, mask, maxDist3DPtsZ) <= 0){
                            if (verbose) {
                                cout << "Triangulating points not successful." << endl;
                            }
						}
					}
				}
			}

			if (useBA)
			{
				if (cp.BART == 1)
				{
					poselib::refineStereoBA(p1, p2, R, t, Q, K0, K1, false, mask);
				}
				else if (cp.BART == 2)
				{
					poselib::CamToImgCoordTrans(p1, K0);
					poselib::CamToImgCoordTrans(p2, K1);
					poselib::refineStereoBA(p1, p2, R, t, Q, K0, K1, true, mask);
				}
			}
            t_2 = chrono::high_resolution_clock::now();
            ar[i].tm.bundleAdjust = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
            CV_Assert((p1.rows == mask.rows) || (p1.rows == mask.cols));
            CV_Assert((p2.rows == mask.rows) || (p2.rows == mask.cols));
            CV_Assert(mask_E.size() == mask.size());
            ar[i].calcInlRatEstimated(mask_E);
            ar[i].calcRTDiff(R, sm.actR, t, sm.actT, false, !noPoseDiff && compInitPose && (verbose > 0));

			if (verbose > 2)
			{
				t_mea = 1000 * ((double)getTickCount() - t_mea) / getTickFrequency(); //End time measurement
				std::cout << "Time for bundle adjustment: " << t_mea << "ms" << endl;
				t_oa += t_mea;
			}
			if (verbose > 3)
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
                t_1 = chrono::high_resolution_clock::now();
				if (cp.RobMethod == "USAC")
				{
					cinfigureUSAC(cp.cfg,
						cfgUSACnr,
                                  cp.USACdegenTh,
						K0,
						K1,
						imgSize,
						&kp1,
						&kp2,
						&finalMatches,
                                  cp.th,
                                  cp.USACInlratFilt);
				}

				static bool poseWasStable = false;
				if (stereoObj->addNewCorrespondences(finalMatches, kp1, kp2, cp.cfg) != -1)
				{
					R = stereoObj->R_new;
					t = stereoObj->t_new;
				}
				else
				{
					cout << "Pose estimation failed!" << endl;
                    t_2 = chrono::high_resolution_clock::now();
                    ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                    sm.actR.copyTo(ar[i].R_GT);
                    sm.actT.copyTo(ar[i].t_GT);
                    sm.K1.copyTo(ar[i].K1_GT);
                    sm.K2.copyTo(ar[i].K2_GT);
                    if(useGTCamMat){
                        sm.K1.copyTo(ar[i].K1_degenerate);
                        sm.K2.copyTo(ar[i].K2_degenerate);
                    }else {
                        sm.actKd1.copyTo(ar[i].K1_degenerate);
                        sm.actKd2.copyTo(ar[i].K2_degenerate);
                    }
					continue;
				}
                t_2 = chrono::high_resolution_clock::now();
                ar[i].tm.stereoRefine = chrono::duration_cast<chrono::microseconds>(t_2 - t_1).count();
                ar[i].calcRTDiff(R, sm.actR, t, sm.actT, false, !noPoseDiff && compInitPose && (verbose > 0));

				if(evStepStereoStable_cnt == 0)
					evStepStereoStable_cnt = evStepStereoStable_tmp;

				if (cp.useMostLikelyPose && stereoObj->mostLikelyPose_stable)
				{
					R = stereoObj->R_mostLikely;
					t = stereoObj->t_mostLikely;
				}
                ar[i].calcRTDiff(stereoObj->R_mostLikely,
                        sm.actR,
                        stereoObj->t_mostLikely,
                        sm.actT,
                        true,
                        !noPoseDiff && compInitPose && (verbose > 0));
				ar[i].mostLikelyPose_stable = stereoObj->mostLikelyPose_stable;
				ar[i].poseIsStable = stereoObj->poseIsStable;
				ar[i].poolSize = (int)stereoObj->getCorrespondencePoolSize();

				if (stereoObj->poseIsStable)
				{
					evStepStereoStable_cnt--;
					poseWasStable = true;
					R.copyTo(R_stable);
					t.copyTo(t_stable);
				}
				else if (poseWasStable && cp.useOnlyStablePose && !(cp.useMostLikelyPose && stereoObj->mostLikelyPose_stable))
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

        if(useGTCamMat){
            ar[i].enterCamMats(K0, K1, sm.K1, sm.K2, sm.K1, sm.K2);
        }else {
            ar[i].enterCamMats(K0, K1, sm.K1, sm.K2, sm.actKd1, sm.actKd2);
        }

        //Get the rotation angles in degrees and display the translation
        if(verbose) {
            double roll, pitch, yaw;
            if (compInitPose) {
                poselib::getAnglesRotMat(R1, roll, pitch, yaw);
                std::cout << "Angles of  original rotation: roll = " << setprecision(4) << roll << char(248)
                          << " pitch = " << pitch << char(248) << " yaw = " << yaw << char(248) << endl;
            }
            poselib::getAnglesRotMat(R, roll, pitch, yaw);
            std::cout << "Angles of estimated rotation: roll = " << setprecision(4) << roll << char(248) << " pitch = "
                      << pitch << char(248) << " yaw = " << yaw << char(248) << endl;
            std::cout << "Rotation matrix:" << std::endl;
            for (size_t m = 0; m < 3; m++) {
                for (size_t j = 0; j < 3; j++) {
                    std::cout << setprecision(6) << R.at<double>(m, j) << "  ";
                }
                std::cout << std::endl;
            }
            if (compInitPose) {
                //Normalize the original translation vector
                t1 = t1 / cv::norm(t1);
                std::cout << "Original  translation vector: [ " << setprecision(4) << t1.at<double>(0) << " "
                          << t1.at<double>(1) << " " << t1.at<double>(2) << " ]" << endl;
            }
            std::cout << "Estimated translation vector: [ " << setprecision(4) << t.at<double>(0) << " "
                      << t.at<double>(1) << " " << t.at<double>(2) << " ]" << endl;
            std::cout << std::endl << std::endl;
        }
    }

    //Calculate difference of GT extrinsics from frame to frame
    for (size_t l = 1; l < ar.size(); ++l) {
        ar[l].calcRTDiff(ar[l].R_GT, ar[l - 1].R_GT, ar[l].t_GT, ar[l - 1].t_GT, false, false, true);
    }

    //Write results to disk
    string outFileName;
    if(!genOutFileName(output_path, cp, outFileName)){
        cerr << "Unable to generate output file name." << endl;
        exit(1);
    }
    testing::internal::FilePath overviewFNameG =
            testing::internal::FilePath::MakeFileName(testing::internal::FilePath(output_path),
                                                                testing::internal::FilePath("allRunsOverview"),
                                                                0,
                                                                ovf_ext1part.c_str());
    string overviewFName = overviewFNameG.string();
    if(!writeResultsOverview(overviewFName, cp, outFileName)){
        cerr << "Unable to write algorithm parameters to output." << endl;
        exit(1);
    }
    if(!writeResultsDisk(ar, outFileName)){
        cerr << "Unable to write results to disk." << endl;
        exit(1);
    }

    return true;
}

//Generate an unique hash value for the used parameters
size_t getHashCalibPars(const calibPars &cp){
    std::stringstream ss;
    string strFromPars;

    ss << cp.useMostLikelyPose;
    ss << cp.useOnlyStablePose;
    ss << cp.evStepStereoStable;
    ss << cp.cfg_stereo.Halign;
    ss << cp.cfg_stereo.RobMethod;
    ss << cp.cfg_stereo.refineMethod;
    ss << cp.cfg_stereo.autoTH;
    ss << cp.cfg_stereo.descriptorType;
    ss << cp.cfg_stereo.absThRankingStable;
    ss << cp.cfg_stereo.BART;
    ss << cp.cfg_stereo.BART_CorrPool;
    ss << cp.cfg_stereo.checkPoolPoseRobust;
    ss << cp.cfg_stereo.keypointType;
    ss << cp.cfg_stereo.kneipInsteadBA;
    ss << cp.cfg_stereo.kneipInsteadBA_CorrPool;
    ss << cp.cfg_stereo.maxDist3DPtsZ;
    ss << cp.cfg_stereo.maxPoolCorrespondences;
    ss << cp.cfg_stereo.maxRat3DPtsFar;
    ss << cp.cfg_stereo.maxSkipPairs;
    ss << cp.cfg_stereo.minContStablePoses;
    ss << cp.cfg_stereo.minInlierRatioReInit;
    ss << cp.cfg_stereo.minInlierRatSkip;
    ss << cp.cfg_stereo.minNormDistStable;
    ss << cp.cfg_stereo.minPtsDistance;
    ss << cp.cfg_stereo.minStartAggInlRat;
    ss << cp.cfg_stereo.raiseSkipCnt;
    ss << cp.cfg_stereo.refineMethod_CorrPool;
    ss << cp.cfg_stereo.refineRTold;
    ss << cp.cfg_stereo.refineRTold_CorrPool;
    ss << cp.cfg_stereo.relInlRatThLast;
    ss << cp.cfg_stereo.relInlRatThNew;
    ss << cp.cfg_stereo.relMinInlierRatSkip;
    ss << cp.cfg_stereo.th_pix_user;
    ss << cp.cfg_stereo.useRANSAC_fewMatches;
    ss << cp.refineRTold;
    ss << cp.BART;
    ss << cp.autoTH;
    ss << cp.refineMethod;
    ss << cp.RobMethod;
    ss << cp.Halign;
    ss << cp.cfg.imgSize.height;
    ss << cp.cfg.imgSize.width;
    ss << cp.cfg.automaticSprtInit;
    ss << cp.cfg.degenDecisionTh;
    ss << cp.cfg.degeneracyCheck;
    ss << cp.cfg.estimator;
    ss << cp.cfg.focalLength;
    ss << cp.cfg.noAutomaticProsacParamters;
    ss << cp.cfg.prevalidateSample;
    ss << cp.cfg.refinealg;
    ss << cp.cfg.th_pixels;
    ss << cp.USACInlratFilt;
    ss << cp.USACdegenTh;
    ss << cp.hashMatchingPars;
    ss << cp.matchData_idx;
    ss << cp.sequ_path;
    ss << cp.mfo.refineGMS;
    ss << cp.mfo.refineSOF;
    ss << cp.mfo.refineVFC;
    ss << cp.stereoRef;
    ss << cp.th;
    ss << cp.kneipInsteadBA;

    strFromPars = ss.str();

    std::hash<std::string> hash_fn;
    return hash_fn(strFromPars);
}

//Generate an unique filename using a hash value for storing algorithm results
bool genOutFileName(const std::string &path,
        const calibPars &cp,
        std::string &filename){

    size_t hashVal = getHashCalibPars(cp);
    testing::internal::FilePath outFileNameG =
            testing::internal::FilePath::MakeFileName(testing::internal::FilePath(path),
                                                                testing::internal::FilePath(std::to_string(hashVal)),
                                                                0,
                                                                "txt");

    int idx = 0;
    while(outFileNameG.FileOrDirectoryExists()){
        outFileNameG =
                testing::internal::FilePath::MakeFileName(
                        testing::internal::FilePath(path),
                        testing::internal::FilePath(std::to_string(hashVal) + "_" + std::to_string(idx)),
                        0,
                        "txt");
        idx++;
        if(idx > 10000){
            cerr << "Too many output files with the same hashing value/parameters." << endl;
            return false;
        }
    }
    filename = outFileNameG.string();

    return true;
}

//Write an parameter overview to file that connects a hash number (in results file name) with the used parameters
bool writeResultsOverview(const string &filename,
        const calibPars &cp,
        const string &resultsFileName){
    using namespace boost::interprocess;
    testing::internal::FilePath filenameG(filename);
    FileStorage fs;
    int nrEntries = 0;
    string parSetNr = "parSetNr";
    /*FILE* fp = nullptr;
    int fd = 0, cnt = 0;*/
    //Open or create a named mutex
    try {
        named_mutex mutex(open_or_create, "write_yaml");
        scoped_lock<named_mutex> lock(mutex);
        if (filenameG.FileOrDirectoryExists()) {
            //Check if file is opened in an other process
/*#if defined(__linux__)
        fp = fopen(filename.c_str(), "a");
        if(fp) {
            fd = fileno(fp);
            int res = 0;
            do {
                if (cnt) {
                    std::this_thread::sleep_for(chrono::seconds(1));
                }
                res = lockf(fd, F_LOCK, 0);
                cnt += 1;
            } while ((res == -1) && (cnt < 20));
            if (cnt == 20) {
                cerr << "Unable to lock file " << filename << endl;
            }
        }else{
            cerr << "Unable to open file " << filename << endl;
        }
#endif*/
            //Check number of entries first
            if (!getNrEntriesYAML(filename, parSetNr, nrEntries)) {
                return false;
            }
            fs = FileStorage(filename, FileStorage::APPEND);
            if (!fs.isOpened()) {
                cerr << "Failed to open " << filename << endl;
                return false;
            }
            fs.writeComment("\n\nNext parameters:\n", 0);
            parSetNr += std::to_string(nrEntries);
        } else {
            fs = FileStorage(filename, FileStorage::WRITE);
            if (!fs.isOpened()) {
                cerr << "Failed to open " << filename << endl;
                return false;
            }
            fs.writeComment("This file contains the file name and its corresponding parameters for "
                                "tested calibration runs.\n\n", 0);
            parSetNr += "0";
        }
        fs << parSetNr;
        fs << "{";

        fs.writeComment("File name (within the path containing this file) which holds results from testing "
                            "the autocalibration SW.", 0);
        size_t posLastSl = resultsFileName.rfind('/');
        string resDirName = resultsFileName.substr(posLastSl + 1);
        fs << "hashTestingPars" << resDirName;

        writeTestingParameters(fs, cp);
        fs << "}";

        fs.release();
    }catch(interprocess_exception &ex){
        cerr << ex.what() << std::endl;
        return false;
    }

/*#if defined(__linux__)
    if(fp && (nrEntries > 0)){
        if(lockf(fd, F_ULOCK, 0) == -1){
            cerr << "Unable to unlock file " << filename << endl;
        }
        fclose(fp);
    }
#endif*/

    return true;
}

void writeTestingParameters(cv::FileStorage &fs,
                            const calibPars &cp){
    fs << "sequ_path" << cp.sequ_path;
    fs << "matchData_idx" << cp.matchData_idx;
    fs << "hashMatchingPars" << cp.hashMatchingPars;

    fs << "stereoRef" << cp.stereoRef;
    fs << "th" << cp.th;
    fs << "RobMethod" << cp.RobMethod;
    fs << "USAC_parameters" << "{";
    fs << "th_pixels" << cp.cfg.th_pixels;
    fs << "USACInlratFilt" << ((cp.USACInlratFilt == 0) ? "GMS":"VFC");
    if(cp.cfg.automaticSprtInit == poselib::SprtInit::SPRT_DEFAULT_INIT) {
        fs << "automaticSprtInit" << "SPRT_DEFAULT_INIT";
    }else if(cp.cfg.automaticSprtInit == poselib::SprtInit::SPRT_DELTA_AUTOM_INIT){
        fs << "automaticSprtInit" << "SPRT_DELTA_AUTOM_INIT";
    }else if(cp.cfg.automaticSprtInit == poselib::SprtInit::SPRT_EPSILON_AUTOM_INIT){
        fs << "automaticSprtInit" << "SPRT_EPSILON_AUTOM_INIT";
    }else{
        fs << "automaticSprtInit" << "SPRT_DELTA_AND_EPSILON_AUTOM_INIT";
    }
    fs << "automaticProsacParameters" << (cp.cfg.noAutomaticProsacParamters ? "disabled":"enabled");
    fs << "prevalidateSample" << (cp.cfg.prevalidateSample ? "enabled":"disabled");
    if(cp.cfg.estimator == poselib::PoseEstimator::POSE_NISTER) {
        fs << "estimator" << "POSE_NISTER";
    }else if(cp.cfg.estimator == poselib::PoseEstimator::POSE_EIG_KNEIP){
        fs << "estimator" << "POSE_EIG_KNEIP";
    }else{
        fs << "estimator" << "POSE_STEWENIUS";
    }
    if(cp.cfg.refinealg == poselib::RefineAlg::REF_WEIGHTS) {
        fs << "refinealg" << "REF_WEIGHTS";
    }else if(cp.cfg.refinealg == poselib::RefineAlg::REF_8PT_PSEUDOHUBER){
        fs << "refinealg" << "REF_8PT_PSEUDOHUBER";
    }else if(cp.cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP){
        fs << "refinealg" << "REF_EIG_KNEIP";
    }else if(cp.cfg.refinealg == poselib::RefineAlg::REF_EIG_KNEIP_WEIGHTS){
        fs << "refinealg" << "REF_EIG_KNEIP_WEIGHTS";
    }else if(cp.cfg.refinealg == poselib::RefineAlg::REF_STEWENIUS){
        fs << "refinealg" << "REF_STEWENIUS";
    }else if(cp.cfg.refinealg == poselib::RefineAlg::REF_STEWENIUS_WEIGHTS){
        fs << "refinealg" << "REF_STEWENIUS_WEIGHTS";
    }else if(cp.cfg.refinealg == poselib::RefineAlg::REF_NISTER){
        fs << "refinealg" << "REF_NISTER";
    }else{
        fs << "refinealg" << "REF_NISTER_WEIGHTS";
    }
    fs << "degenDecisionTh" << cp.cfg.degenDecisionTh;
    if(cp.cfg.degeneracyCheck == poselib::UsacChkDegenType::DEGEN_NO_CHECK) {
        fs << "degeneracyCheck" << "DEGEN_NO_CHECK";
    }else if(cp.cfg.degeneracyCheck == poselib::UsacChkDegenType::DEGEN_QDEGSAC){
        fs << "degeneracyCheck" << "DEGEN_QDEGSAC";
    }else{
        fs << "degeneracyCheck" << "DEGEN_USAC_INTERNAL";
    }
    fs << "}";

    fs << "refineMethod" << "{";
    if((cp.refineMethod & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT) {
        fs << "algorithm" << "PR_NO_REFINEMENT";
    }else if((cp.refineMethod & 0xF) == poselib::RefinePostAlg::PR_8PT){
        fs << "algorithm" << "PR_8PT";
    }else if((cp.refineMethod & 0xF) == poselib::RefinePostAlg::PR_NISTER){
        fs << "algorithm" << "PR_NISTER";
    }else if((cp.refineMethod & 0xF) == poselib::RefinePostAlg::PR_STEWENIUS){
        fs << "algorithm" << "PR_STEWENIUS";
    }else {
        fs << "algorithm" << "PR_KNEIP";
    }
    if((cp.refineMethod & 0xF0) == poselib::RefinePostAlg::PR_TORR_WEIGHTS) {
        fs << "costFunction" << "PR_TORR_WEIGHTS";
    }else if((cp.refineMethod & 0xF0) == poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS) {
        fs << "costFunction" << "PR_PSEUDOHUBER_WEIGHTS";
    }else{
        fs << "costFunction" << "PR_NO_WEIGHTS";
    }
    fs << "}";

    fs << "kneipInsteadBA" << cp.kneipInsteadBA;
    fs << "refineRTold" << cp.refineRTold;
    if(cp.BART == 0) {
        fs << "BART" << "disabled";
    }else if(cp.BART == 1) {
        fs << "BART" << "extr_only";
    }else{
        fs << "BART" << "extr_intr";
    }
    fs << "matchesFilter" << "{";
    fs << "refineGMS" << (cp.mfo.refineGMS ? "enabled":"disabled");
    fs << "refineVFC" << (cp.mfo.refineVFC ? "enabled":"disabled");
    fs << "refineSOF" << (cp.mfo.refineSOF ? "enabled":"disabled") << "}";

    fs << "stereoParameters" << "{";
    fs << "th_pix_user" << cp.cfg_stereo.th_pix_user;
    fs << "keypointType" << cp.cfg_stereo.keypointType;
    fs << "descriptorType" << cp.cfg_stereo.descriptorType;
    fs << "RobMethod" << cp.cfg_stereo.RobMethod;
    fs << "refineMethod" << "{";
    if((cp.cfg_stereo.refineMethod & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT) {
        fs << "algorithm" << "PR_NO_REFINEMENT";
    }else if((cp.cfg_stereo.refineMethod & 0xF) == poselib::RefinePostAlg::PR_8PT){
        fs << "algorithm" << "PR_8PT";
    }else if((cp.cfg_stereo.refineMethod & 0xF) == poselib::RefinePostAlg::PR_NISTER){
        fs << "algorithm" << "PR_NISTER";
    }else if((cp.cfg_stereo.refineMethod & 0xF) == poselib::RefinePostAlg::PR_STEWENIUS){
        fs << "algorithm" << "PR_STEWENIUS";
    }else {
        fs << "algorithm" << "PR_KNEIP";
    }
    if((cp.cfg_stereo.refineMethod & 0xF0) == poselib::RefinePostAlg::PR_TORR_WEIGHTS) {
        fs << "costFunction" << "PR_TORR_WEIGHTS";
    }else if((cp.cfg_stereo.refineMethod & 0xF0) == poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS) {
        fs << "costFunction" << "PR_PSEUDOHUBER_WEIGHTS";
    }else{
        fs << "costFunction" << "PR_NO_WEIGHTS";
    }
    fs << "}";
    fs << "refineMethod_CorrPool" << "{";
    if((cp.cfg_stereo.refineMethod_CorrPool & 0xF) == poselib::RefinePostAlg::PR_NO_REFINEMENT) {
        fs << "algorithm" << "PR_NO_REFINEMENT";
    }else if((cp.cfg_stereo.refineMethod_CorrPool & 0xF) == poselib::RefinePostAlg::PR_8PT){
        fs << "algorithm" << "PR_8PT";
    }else if((cp.cfg_stereo.refineMethod_CorrPool & 0xF) == poselib::RefinePostAlg::PR_NISTER){
        fs << "algorithm" << "PR_NISTER";
    }else if((cp.cfg_stereo.refineMethod_CorrPool & 0xF) == poselib::RefinePostAlg::PR_STEWENIUS){
        fs << "algorithm" << "PR_STEWENIUS";
    }else {
        fs << "algorithm" << "PR_KNEIP";
    }
    if((cp.cfg_stereo.refineMethod_CorrPool & 0xF0) == poselib::RefinePostAlg::PR_TORR_WEIGHTS) {
        fs << "costFunction" << "PR_TORR_WEIGHTS";
    }else if((cp.cfg_stereo.refineMethod_CorrPool & 0xF0) == poselib::RefinePostAlg::PR_PSEUDOHUBER_WEIGHTS) {
        fs << "costFunction" << "PR_PSEUDOHUBER_WEIGHTS";
    }else{
        fs << "costFunction" << "PR_NO_WEIGHTS";
    }
    fs << "}";
    fs << "kneipInsteadBA" << cp.cfg_stereo.kneipInsteadBA;
    fs << "kneipInsteadBA_CorrPool" << cp.cfg_stereo.kneipInsteadBA_CorrPool;
    fs << "refineRTold" << cp.cfg_stereo.refineRTold;
    fs << "refineRTold_CorrPool" << cp.cfg_stereo.refineRTold_CorrPool;
    if(cp.cfg_stereo.BART == 0) {
        fs << "BART" << "disabled";
    }else if(cp.cfg_stereo.BART == 1) {
        fs << "BART" << "extr_only";
    }else{
        fs << "BART" << "extr_intr";
    }
    if(cp.cfg_stereo.BART_CorrPool == 0) {
        fs << "BART_CorrPool" << "disabled";
    }else if(cp.cfg_stereo.BART_CorrPool == 1) {
        fs << "BART_CorrPool" << "extr_only";
    }else{
        fs << "BART_CorrPool" << "extr_intr";
    }
    fs << "checkPoolPoseRobust" << (int)cp.cfg_stereo.checkPoolPoseRobust;
    fs << "useRANSAC_fewMatches" << (cp.cfg_stereo.useRANSAC_fewMatches ? "enabled":"disabled");
    fs << "maxPoolCorrespondences" << (int)cp.cfg_stereo.maxPoolCorrespondences;
    fs << "maxDist3DPtsZ" << cp.cfg_stereo.maxDist3DPtsZ;
    fs << "maxRat3DPtsFar" << cp.cfg_stereo.maxRat3DPtsFar;
    fs << "minStartAggInlRat" << cp.cfg_stereo.minStartAggInlRat;
    fs << "minInlierRatSkip" << cp.cfg_stereo.minInlierRatSkip;
    fs << "relInlRatThLast" << cp.cfg_stereo.relInlRatThLast;
    fs << "relInlRatThNew" << cp.cfg_stereo.relInlRatThNew;
    fs << "relMinInlierRatSkip" << cp.cfg_stereo.relMinInlierRatSkip;
    fs << "minInlierRatioReInit" << cp.cfg_stereo.minInlierRatioReInit;
    fs << "maxSkipPairs" << (int)cp.cfg_stereo.maxSkipPairs;
    fs << "minNormDistStable" << cp.cfg_stereo.minNormDistStable;
    fs << "absThRankingStable" << cp.cfg_stereo.absThRankingStable;
    fs << "minContStablePoses" << (int)cp.cfg_stereo.minContStablePoses;
    fs << "raiseSkipCnt" << cp.cfg_stereo.raiseSkipCnt;
    fs << "minPtsDistance" << cp.cfg_stereo.minPtsDistance;
    fs << "Halign" << cp.cfg_stereo.Halign;
    fs << "autoTH" << cp.cfg_stereo.autoTH << "}";

    fs << "Halign" << cp.Halign;
    fs << "autoTH" << cp.autoTH;

    fs << "useMostLikelyPose" << cp.useMostLikelyPose;
    fs << "useOnlyStablePose" << cp.useOnlyStablePose;
    fs << "evStepStereoStable" << cp.evStepStereoStable;
    fs << "accumCorrs" << cp.accumCorrs;
}

bool getNrEntriesYAML(const std::string &filename, const string &buzzword, int &nrEntries){
    FileStorage fs = FileStorage(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    nrEntries = 0;
    while(true) {
        cv::FileNode fn = fs[buzzword + std::to_string(nrEntries)];
        if (fn.empty()) {
            break;
        }
        nrEntries++;
    }

    fs.release();

    return true;
}

//Write the results from testing to disk
bool writeResultsDisk(const std::vector<algorithmResult> &ar, const string &filename){
    ofstream evalsToFile(filename);
    if(!evalsToFile.is_open()){
        cerr << "Error creating output file for writing results." << endl;
        return false;
    }

    //Print header
    evalsToFile << "Nr;";
    ar[0].print(evalsToFile, true);

    //Print data
    int idx = 0;
    for(auto &i: ar){
        evalsToFile << idx << ";";
        i.print(evalsToFile, false);
        idx++;
    }

    evalsToFile.close();

    return true;
}

void printCVMat(const cv::Mat &m, std::ofstream &os, const std::string &name){
    CV_Assert(m.type() == CV_64FC1);
    int rows = m.rows;
    int cols = m.cols;
    int rows1 = rows - 1;
    int cols1 = cols - 1;

    if(name.empty()){
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if((y == rows1) && (x == cols1)){
                    os << m.at<double>(y,x);
                }else {
                    os << m.at<double>(y, x) << ";";
                }
            }
        }
    }else{
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if((y == rows1) && (x == cols1)){
                    os << name << "(" << y << "," << x << ")";
                }else {
                    os << name << "(" << y << "," << x << ")" << ";";
                }
            }
        }
    }
}

void cinfigureUSAC(poselib::ConfigUSAC &cfg, 
	int cfgUSACnr[6], 
	double USACdegenTh, 
	const cv::Mat &K0,
	const cv::Mat &K1,
	cv::Size imgSize, 
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

bool loadFeatureName(const string &filename,
        const int parSetNr,
        string &descriptorName,
        string &keyPName,
        string &matchesSubFolder){
    FileStorage fs = FileStorage(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    cv::FileNode fn = fs["parSetNr" + std::to_string(parSetNr)];
    if (fn.empty()) {
        return false;
    }

    fn["keyPointType"] >> keyPName;
    fn["descriptorType"] >> descriptorName;
    fn["hashMatchingPars"] >> matchesSubFolder;

    fs.release();

    return true;
}

bool loadImgSize(const string &filename,
                     cv::Size &imgSize){
    FileStorage fs = FileStorage(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    cv::FileNode n = fs["imgSize"];
    int first_int, second_int;
    n["width"] >> first_int;
    n["height"] >> second_int;
    imgSize = cv::Size(first_int, second_int);

    fs.release();

    return true;
}

FileStorage& operator << (FileStorage& fs, bool &value)
{
    if(value){
        return (fs << 1);
    }

    return (fs << 0);
}

/** @function main */
int main( int argc, char* argv[])
{
    ArgvParser cmd;
    if(SetupCommandlineParser(cmd, argc, argv) != 0){
        return EXIT_FAILURE;
    }
    if(!startEvaluation(cmd)){
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

//#endif
