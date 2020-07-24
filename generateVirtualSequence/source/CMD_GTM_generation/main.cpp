//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2020 Josef Maier
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

#include "argvparser.h"

#include "io_data.h"
#include "glob_includes.h"

#include "getStereoCameraExtr.h"
#include "generateSequence.h"
#include "generateMatches.h"
#include "helper_funcs.h"
#include "GTM/base_matcher.h"
#include <chrono>
#include <iomanip>

using namespace std;
using namespace CommandLineProcessing;

int startCalculation(ArgvParser& cmd);

void SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
{
	cmd.setIntroductoryDescription("Test interface for generating randomized 3D scenes and matches");
	//define error codes
	cmd.addErrorCode(0, "Success");
	cmd.addErrorCode(1, "Error");

	cmd.setHelpOption("h", "help","Generation of ground truth matches (GTM) and "
                               "optical flow from the MegaDepth dataset.");
	cmd.defineOption("img_path", "<Path to the image folder which includes folders "
                              "MegaDepth, Oxford, and KITTI.>", ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("kitti", "<If provided, GTM for KITTI are calculated.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("oxford", "<If provided, GTM for Oxford are calculated.>", ArgvParser::NoOptionAttribute);
    cmd.defineOption("mega", "<If provided, GTM and flow for the MegaDepth dataset are calculated.>",
            ArgvParser::NoOptionAttribute);
    cmd.defineOption("nr_kitti", "<If provided, only as many matches as specified here are calculated from the KITTI dataset. "
                                 "Otherwise, the full dataset is processed.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("nr_oxford", "<If provided, only as many matches as specified here are calculated from the Oxford dataset. "
                                 "Otherwise, the full dataset is processed.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("nr_mega", "<If provided, only as many matches as specified here are calculated from the MegaDepth dataset. "
                                 "Otherwise, the full dataset is processed.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("f_detect", "<The name of the feature detector (FAST, MSER, ORB, BRISK, KAZE, AKAZE, STAR, MSD)"
                                 "(For SIFT & SURF, OpenCV must be built with option -DOPENCV_ENABLE_NONFREE). "
                                 "[Default=ORB]>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("d_extr", "<The name of the descriptor extractor (BRISK, ORB, KAZE, AKAZE, FREAK, "
                               "DAISY, LATCH, BGM, BGM_HARD, BGM_BILINEAR, LBGM, BINBOOST_64, BINBOOST_128, BINBOOST_256, "
                               "VGG_120, VGG_80, VGG_64, VGG_48, RIFF, BOLD )"
                               "(For SIFT & SURF, OpenCV must be built with option -DOPENCV_ENABLE_NONFREE). "
                               "[Default=ORB]>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("v", "<Verbose in HEX [Default: 0] (combinations using bit-wise or operator can be provided):\n"
                          "0x0\t no output\n"
                          "0x1\t show initial GTM keypoints\n"
                          "0x2\t show generation process of GTM\n"
                          "0x4\t show interpolated flow\n"
                          "0x8\t print Ceres progress for refining MegaDepth poses and intrinsics\n"
                          "0x10\t print Ceres result summaries of refined MegaDepth poses and intrinsics\n"
                          "0x20\t show matches on distorted image pairs of the MegaDepth dataset\n"
                          "0x40\t show matches on undistorted image pairs of the MegaDepth dataset\n"
                          "0x80\t show matches on undistorted image pairs of the MegaDepth dataset with refined poses and intrinsics.>",
                          ArgvParser::OptionRequiresValue);
    cmd.defineOption("only_mega_flow", "<If provided, flow for the MegaDepth dataset is calculated and "
                                       "calculation of GTM is skipped.>",ArgvParser::NoOptionAttribute);
    cmd.defineOption("del_pool", "<If provided, all internal variables holding GTM information are "
                                 "cleared after each iteration to save memory (RAM).>",ArgvParser::NoOptionAttribute);
    cmd.defineOption("skip_gtm_refine", "<If provided, GTM refinement or annotation step is skipped.>",ArgvParser::NoOptionAttribute);

	/// finally parse and handle return codes (display help etc...)
	int result = -1;
	result = cmd.parse(argc, argv);
	if (result != ArgvParser::NoParserError)
	{
		cout << cmd.parseErrorDescription(result);
		exit(1);
	}
}

int startCalculation(ArgvParser& cmd)
{
    bool use_kitti = cmd.foundOption("kitti");
    bool use_oxf = cmd.foundOption("oxford");
    bool use_mega = cmd.foundOption("mega");
    bool only_mega_flow = cmd.foundOption("only_mega_flow");
	if(!(use_kitti | use_oxf | use_mega | only_mega_flow)){
		cerr << "At least one of options kitti, oxford, mega, or only_mega_flow must be provided." << endl;
		return -1;
	}
	size_t nr_kitti = 0, nr_oxford = 0, nr_mega = 0;
	if(cmd.foundOption("nr_kitti")){
        nr_kitti = stoul(cmd.optionValue("nr_kitti"));
        if(nr_kitti < 50){
            cerr << "Number of matches to calculate for KITTI too small. Disabling." << endl;
            nr_kitti = 0;
            use_kitti = false;
        }
	}else{
        nr_kitti = numeric_limits<size_t>::max();
	}
    if(cmd.foundOption("nr_oxford")){
        nr_oxford = stoul(cmd.optionValue("nr_oxford"));
        if(nr_oxford < 50){
            cerr << "Number of matches to calculate for Oxford too small. Disabling." << endl;
            nr_oxford = 0;
            use_oxf = false;
        }
    }else{
        nr_oxford = numeric_limits<size_t>::max();
    }
    if(cmd.foundOption("nr_mega")){
        nr_mega = stoul(cmd.optionValue("nr_mega"));
        if(nr_mega < 50){
            cerr << "Number of matches to calculate for KITTI too small. Disabling." << endl;
            nr_mega = 0;
            use_mega = false;
        }
    }else{
        nr_mega = numeric_limits<size_t>::max();
    }

    string img_path = cmd.optionValue("img_path");
    if(!checkPathExists(img_path)){
        cerr << "Path " << img_path << " does not exist." << endl;
    }
    string keyPointType = "ORB";
    string descriptorType = "ORB";
    if(cmd.foundOption("f_detect")) {
        keyPointType = cmd.optionValue("f_detect");
    }
    if(cmd.foundOption("d_extr")) {
        descriptorType = cmd.optionValue("d_extr");
    }
    uint32_t verbose = 0;
    if(cmd.foundOption("v")) {
        size_t verbose0 = stoul(cmd.optionValue("v"), nullptr, 16);
        if(verbose0 & 0x1) verbose |= vorboseType::SHOW_GTM_KEYPOINTS;
        if(verbose0 & 0x2) verbose |= vorboseType::SHOW_GTM_GEN_PROCESS;
        if(verbose0 & 0x4) verbose |= vorboseType::SHOW_GTM_INTERPOL_FLOW;
        if(verbose0 & 0x8) verbose |= vorboseType::PRINT_MEGADEPTH_CERES_PROGRESS;
        if(verbose0 & 0x10) verbose |= vorboseType::PRINT_MEGADEPTH_CERES_RESULTS;
        if(verbose0 & 0x20) verbose |= vorboseType::SHOW_MEGADEPTH_MATCHES_DISTORTED;
        if(verbose0 & 0x40) verbose |= vorboseType::SHOW_MEGADEPTH_MATCHES_UNDISTORTED;
        if(verbose0 & 0x80) verbose |= vorboseType::SHOW_MEGADEPTH_MATCHES_REFINED;
    }
    bool del_pool = cmd.foundOption("del_pool");
    bool skip_gtm_refine = cmd.foundOption("skip_gtm_refine");

    std::random_device rd;
    std::mt19937 rand2(rd());
    baseMatcher bm(keyPointType, img_path, descriptorType, false, verbose, &rand2, !skip_gtm_refine, only_mega_flow, del_pool);
    int err = 0;
    if(use_oxf){
        if(!bm.calcGTM_Oxford(nr_oxford)){
            cerr << "Error at Oxford GTM calculation" << endl;
            err = 1;
        }
    }
    if(use_kitti){
        if(!bm.calcGTM_KITTI(nr_kitti)){
            cerr << "Error at KITTI GTM calculation" << endl;
            err += 2;
        }
    }
    if(use_mega || only_mega_flow){
        if(!bm.calcGTM_MegaDepth(nr_mega)){
            cerr << "Error at MegaDepth GTM calculation" << endl;
            err += 4;
        }
    }

	return err;
}

/** @function main */
int main( int argc, char* argv[])
{
	ArgvParser cmd;
	SetupCommandlineParser(cmd, argc, argv);
	int err = startCalculation(cmd);
	if(err){
	    return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}