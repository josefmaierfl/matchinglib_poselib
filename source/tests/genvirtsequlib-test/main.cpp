#include "eval_start.h"
#include "argvparser.h"

#include "io_data.h"

#include "loadGTMfiles.h"

//#include <opencv2/imgproc/imgproc.hpp>

//#include "PfeImgFileIO.h"

#include "getStereoCameraExtr.h"
#include "generateSequence.h"
#include "helper_funcs.h"
#include <time.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace CommandLineProcessing;

double getRandDoubleVal(std::default_random_engine rand_generator, double lowerBound, double upperBound);
void initStarVal(default_random_engine rand_generator, double *range, vector<double>& startVec);
void initRangeVals(default_random_engine rand_generator, double *range, double *relMinMaxCh, vector<double> startVec, vector<double>& actVec);
void testStereoCamGeneration(int verbose = 0, bool genSequence = true);
int genNewSequence(std::vector<cv::Mat>& Rv, std::vector<cv::Mat>& tv, cv::Mat& K_1, cv::Mat& K_2, cv::Size &imgSize);

depthClass getRandDepthClass();

void SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
{
	cmd.setIntroductoryDescription("Evaluation of matching algorithms on ground truth");
	//define error codes
	cmd.addErrorCode(0, "Success");
	cmd.addErrorCode(1, "Error");

	cmd.setHelpOption("h", "help","");
	cmd.defineOption("img_path", "<Path to the images (all required in one folder)>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("gt_path", "<Path to the ground truth data (flow-files, disparity-files, homographies)>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("gt_type", "<Specifies the type of ground truth. 0 for flow, 1 for disparity, 2 for homography>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("l_img_pref", "<The prefix of the left or first image. The whole prefix until the start of the number is needed>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("r_img_pref", "<The prefix of the right or second image. The whole prefix until the start of the number is needed. Can be empty for homographies>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("gt_pref", "<The prefix of the ground truth flow, disparity or homography files. The whole prefix until the start of the number is needed>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("f_detect", "<The name of the feature detector in OpenCV 2.4.9 style>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("d_extr", "<The name of the descriptor extractor in OpenCV 2.4.9 style>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("matcher", "<The short form of the matcher:\n  CASCHASH: hierarchical k-means tree matcher from the FLANN library\n  GEOMAWARE: Geometry-aware Feature matching algorithm\n  GMBSOF: Guided matching based on statistical optical flow\n  HIRCLUIDX: Hirarchical Clustering Index Matching\n  HIRKMEANS: hierarchical k-means tree matcher from the FLANN library\n  VFCKNN: Vector field consensus (VFC) algorithm with k nearest neighbor matches provided from the Hirarchical Clustering Index Matching algorithm from the FLANN library\n  LIBVISO: matcher from the libviso2 library\n  LINEAR: linear Matching algorithm (Brute force) from the FLANN library\n  LINEAR: linear Matching algorithm (Brute force) from the FLANN library\n  LINEAR: linear Matching algorithm (Brute force) from the FLANN library\n  LSHIDX: LSH Index Matching algorithm from the FLANN library\n  RANDKDTREE: randomized KD-trees matcher from the FLANN library>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("ratiot", "<If provided, ratio test is enabled for the matchers for which it is possible.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("res_path", "<Path, where the results should be written to. If empty, a new folder is generated in the image path.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("refine", "<If provided, the result from the matching algorithm is refined with VFC>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("inl_rat", "<If provided, the specified inlier ratio is generated from the ground truth data. Default = 1.0>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("show", "<If provided, the result is shown. It must be specified if the result images should be written to disk. Options:\n  0 = draw only true positives\n  1 = draw true and false positives\n  2 = draw true and false positives + false negatives>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("show_ref", "<If provided, the refined result is shown. It must be specified if the refined result images should be written to disk. Options:\n  0 = draw only true positives\n  1 = draw true and false positives\n  2 = draw true and false positives + false negatives>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("img_res_path", "<If provided, it specifies the path were the result images should be stored. If a wrong path is entered or only a few letters, a default directory is generated in the provided input image directory>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("img_ref_path", "<If provided, it specifies the path were the refined result images should be stored. If a wrong path is entered or only a few letters, a default directory is generated in the provided input image directory>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("t_meas", "<Specifies if the runtime of the algorithm should be evaluated>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("t_meas_inlr", "<Specifies if the runtime of the algorithm should be evaluated in cunjunction with different inlier ratios.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("inl_rat_test", "<Specifies if the quality parameters of the matching algorithm should be evaluated on a single image pair with many different inlier ratios. The index (or indexes for homography images) must be provided>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("inl_rat_test_all", "<Specifies if the quality parameters statistics of the matching algorithm should be evaluated on all image pairs of a scene with many different inlier ratios.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("idx1", "<Used for the inlier ratio test. Specifies the image index>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("idx2", "<Used for the inlier ratio test with homographies. Specifies the image index of the second image>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("qual_meas", "<Starts the calculation of quality parameters of the specified matcher for the given dataset>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("th_eval", "<Calculates the quality parameters of the GMbSOF algorithm for different thresholds of the statistical optical flow>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("rad_eval", "<Tries different multiplication factors for the standard deviation used to calculate the search range of the GMbSOF algorithm. Outputs of this test are the execution time and quality parameters (TPR, ACC, ...) at different inlier ratios (if no specific was given) and different search range multiplication factors.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("valid_th", "<Threshold value for verifying the statistical optical flow (Only used with --rad_eval). If not provided, a default value of 0.3 is used. Possible values are between 0.2 and 1.0.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("initM_eval", "<Starts the evaluation of the estimated inlier ratio as well as on the precision of the initial and filtered (with SOF) initial matches. This option is only possible to choos, if this software was compiled with defines INITMATCHQUALEVAL_O and INITMATCHQUALEVAL set to 1 - otherwise the program exits.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("CDrat_eval", "<Generates a cumulative distribution for the matching cost and match-SOF-distance ratios. The first ratio is a ratio between the descriptor distance of each match and the median of the descriptor distances of each matche's neighbors. The latter one is a ratio between the spatial distance of each matching right keypoint to its SOF-estimated value and the median of the spatial distances of each matche's neighbors.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("s_key_size", "<Specifies if the same number of keypoints should be used over all inlier ratios.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("testGT", "<If specified, the testing of the GT matching is started. If a different descriptor than FREAK is wanted for filtering the GT matches, it must be specified using option 'd_extr'>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("threshhTh", "<Parameter for annotating the GT matches: Threshold for thresholding the difference of matching image patches, to decide if a match should be annotated manually or automatically. A value of 64.0 has proofen to be a good value for normal images, whereas 20.0 should be chosen for synthetic images. Default = 64.0>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("nmsIdx",
		"<Index parameters for matchers of the NMSLIB. See manual of NMSLIB for details. Instead of '=' in the string you have to use '+'. If you are using a NMSLIB matcher but no parameters are given, the default parameters are used which may leed to unsatisfactory results.>",
		ArgvParser::OptionRequiresValue);
	cmd.defineOption("nmsQry",
		"<Query-time parameters for matchers of the NMSLIB. See manual of NMSLIB for details. Instead of '=' in the string you have to use '+'. If you are using a NMSLIB matcher but no parameters are given, the default parameters are used which may leed to unsatisfactory results.>",
		ArgvParser::OptionRequiresValue);
	cmd.defineOption("timeDescr", "<If specified, the runtime of the given descriptor is analyzed>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("loadGTM", "<If specified, the GTM are loaded and displayed. The following options must be specified as well: img_path, gtm_path, gtm_pref, l_img_pref, (r_img_pref).>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("gtm_path", "<Only used with option loadGTM. It specifies the path to the GTM.>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("gtm_postf", "<Only used with option loadGTM. It specifies the postfix of GTM files. It must include the intended inlier ratio (10 * inlier ratio in percent) and keypoint type. E.g. 'inlRat950FAST.gtm'. Specifying an additional folder is also possible: e.g. 'folder/*inlRat950FAST.gtm'>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("initGTMs", "<If specified, the missing initial GTMs (no specific inlier ratio) of a dataset are generated>", ArgvParser::NoOptionAttribute);
	cmd.defineOption("genGTMs", "<If specified, the GTMs are generated with various inlier ratios for a dataset>", ArgvParser::NoOptionAttribute);
	
	/// finally parse and handle return codes (display help etc...)
	int result = -1;
	result = cmd.parse(argc, argv);
	if (result != ArgvParser::NoParserError)
	{
		cout << cmd.parseErrorDescription(result);
		exit(1);
	}
}

void startEvaluation(ArgvParser& cmd)
{
	bool t_meas, t_meas_inlr, inl_rat_test, inl_rat_test_all, qual_meas, th_eval, rad_eval, initM_eval, CDrat_eval;
	string img_path, gt_path, gt_type_str, l_img_pref, gt_pref, r_img_pref, f_detect, d_extr, matcher, nmsIdx, nmsQry;
	string res_path, inl_rat_str, show_str, show_ref_str, img_res_path, img_ref_path, idx1_str, idx2_str, valid_th_str, threshhTh_str;
	int gt_type, show, show_ref, idx1, idx2;
	bool ratiot, refine, s_key_size;
	double inl_rat = -1.0, valid_th = 0.3;
	bool testGT, timeDescr;
	double threshhTh = 64.0;
	bool loadGTM, initGTMs, genGTMs;
	string gtm_path, gtm_postf;

	t_meas = cmd.foundOption("t_meas");
	t_meas_inlr = cmd.foundOption("t_meas_inlr");
	inl_rat_test = cmd.foundOption("inl_rat_test");
	inl_rat_test_all = cmd.foundOption("inl_rat_test_all");
	th_eval = cmd.foundOption("th_eval");
	qual_meas = cmd.foundOption("qual_meas");
	rad_eval = cmd.foundOption("rad_eval");
	initM_eval = cmd.foundOption("initM_eval");
	CDrat_eval = cmd.foundOption("CDrat_eval");
	testGT = cmd.foundOption("testGT");
	timeDescr = cmd.foundOption("timeDescr");
	loadGTM = cmd.foundOption("loadGTM");
	initGTMs = cmd.foundOption("initGTMs");
	genGTMs = cmd.foundOption("genGTMs");

	ratiot = cmd.foundOption("ratiot");
	refine = cmd.foundOption("refine");
	s_key_size = cmd.foundOption("s_key_size");

	if (cmd.foundOption("img_path"))
		img_path = cmd.optionValue("img_path");
	if (cmd.foundOption("gt_path"))
		gt_path = cmd.optionValue("gt_path");
	if (cmd.foundOption("l_img_pref"))
		l_img_pref = cmd.optionValue("l_img_pref");
	if (cmd.foundOption("gt_pref"))
		gt_pref = cmd.optionValue("gt_pref");
	if(cmd.foundOption("r_img_pref"))
		r_img_pref = cmd.optionValue("r_img_pref");
	if (cmd.foundOption("f_detect"))
		f_detect = cmd.optionValue("f_detect");
	if (cmd.foundOption("d_extr"))
		d_extr = cmd.optionValue("d_extr");
	if(cmd.foundOption("matcher"))
		matcher = cmd.optionValue("matcher");
	if(cmd.foundOption("res_path"))
		res_path = cmd.optionValue("res_path");
	if(cmd.foundOption("img_res_path"))
		img_res_path = cmd.optionValue("img_res_path");
	if(cmd.foundOption("img_ref_path"))
		img_ref_path = cmd.optionValue("img_ref_path");

	if (cmd.foundOption("gt_type"))
		gt_type_str = cmd.optionValue("gt_type");
	if(cmd.foundOption("show"))
		show_str = cmd.optionValue("show");
	if(cmd.foundOption("show_ref"))
		show_ref_str = cmd.optionValue("show_ref");
	if(cmd.foundOption("idx1"))
		idx1_str = cmd.optionValue("idx1");
	if(cmd.foundOption("idx2"))
		idx2_str = cmd.optionValue("idx2");
	if(cmd.foundOption("valid_th"))
	{
		valid_th_str = cmd.optionValue("valid_th");
		valid_th = atof(valid_th_str.c_str());
		if((valid_th > 1.0) || (valid_th < 0.2))
		{
			cout << "Validation treshold out of range. Must be between 0.2 and 1.0. Exiting." << endl;
			exit(1);
		}
	}
	if(cmd.foundOption("inl_rat"))
	{
		inl_rat_str = cmd.optionValue("inl_rat");
		inl_rat = atof(inl_rat_str.c_str());
		if((inl_rat > 1.0) || (inl_rat < 0.01))
		{
			cout << "Inlier ratio out of range. Must be between 0.01 and 1.0. Exiting." << endl;
			exit(1);
		}
	}

	if(cmd.foundOption("threshhTh"))
	{
		threshhTh_str = cmd.optionValue("threshhTh");
		threshhTh = (double)std::atof(threshhTh_str.c_str());
		if((threshhTh < 5.0) || (threshhTh > 255))
		{
			cout << "Threshold for tresholding out of range. Must be between 5.0 and 255.0. Exiting." << endl;
			exit(1);
		}
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

	if (cmd.foundOption("gtm_path"))
		gtm_path = cmd.optionValue("gtm_path");

	if (cmd.foundOption("gtm_postf"))
		gtm_postf = cmd.optionValue("gtm_postf");

	gt_type = atoi(gt_type_str.c_str());

	if(!show_str.empty())
	{
		show = atoi(show_str.c_str());
	}
	else
	{
		show = -1;
	}
	if(!show_ref_str.empty())
	{
		show_ref = atoi(show_ref_str.c_str());
	}
	else
	{
		show_ref = -1;
	}
	if(!idx1_str.empty())
	{
		idx1 = atoi(idx1_str.c_str());
	}
	else
	{
		idx1 = INT_MAX;
		if(inl_rat_test)
		{
			cout << "Image index 1 must be provided. Exiting." << endl;
			exit(1);
		}
	}
	if(!idx2_str.empty())
	{
		idx2 = atoi(idx2_str.c_str());
	}
	else
	{
		idx2 = INT_MAX;
		if(inl_rat_test && (gt_type == 2))
		{
			cout << "Image index 2 must be provided. Exiting." << endl;
			exit(1);
		}
	}

	/*if(t_meas)
	{
		startTimeMeasurement(img_path, gt_path, gt_type, 
						 l_img_pref, r_img_pref, gt_pref,
						 f_detect, d_extr, matcher,
						 ratiot, res_path, refine, inl_rat, show, show_ref, img_res_path, img_ref_path, nmsIdx, nmsQry);
	}

	if(t_meas_inlr)
	{
		startTimeMeasurementDiffInlRats(img_path, gt_path, gt_type, 
								l_img_pref, r_img_pref, gt_pref,
								f_detect, d_extr, matcher,
								ratiot, res_path, s_key_size,
								show, img_res_path);
	}

	if(inl_rat_test)
	{
		startInlierRatioMeasurement(img_path, gt_path, gt_type, 
								l_img_pref, r_img_pref, gt_pref,
								f_detect, d_extr, matcher,
								ratiot, res_path, idx1, idx2, refine, 
								show, show_ref, img_res_path, img_ref_path);
	}

	if(inl_rat_test_all)
	{
		startInlierRatioMeasurementWholeSzene(img_path, gt_path, gt_type, 
								l_img_pref, r_img_pref, gt_pref,
								f_detect, d_extr, matcher,
								ratiot, res_path, refine, 
								show, show_ref, img_res_path, img_ref_path, nmsIdx, nmsQry);
	}

	if(th_eval)
	{
		testGMbSOFthreshold(img_path, gt_path, gt_type, 
						 l_img_pref, r_img_pref, gt_pref,
						 f_detect, d_extr,
						 res_path, show, img_res_path);
	}

	if(qual_meas)
	{
		startQualPMeasurement(img_path, gt_path, gt_type, 
						 l_img_pref, r_img_pref, gt_pref,
						 f_detect, d_extr, matcher,
						 ratiot, res_path, refine, inl_rat, show, show_ref, img_res_path, img_ref_path);
	}

	if(rad_eval)
	{
		testGMbSOFsearchRange(img_path, gt_path, gt_type, 
						 l_img_pref, r_img_pref, gt_pref,
						 f_detect, d_extr,
						 res_path, inl_rat, valid_th, show, 
						 img_res_path);
	}

	if(initM_eval)
	{
		testGMbSOFinitMatching(img_path, gt_path, gt_type, 
						 l_img_pref, r_img_pref, gt_pref,
						 f_detect, d_extr,
						 res_path, show, img_res_path);
	}

	if(CDrat_eval)
	{
		testGMbSOF_CDratios(img_path, gt_path, gt_type, 
						 l_img_pref, r_img_pref, gt_pref,
						 f_detect, d_extr,
						 res_path, show, img_res_path);
	}

	if(testGT)
	{
		testGTmatches(img_path, gt_path, gt_type, 
				  l_img_pref, r_img_pref, gt_pref,
				  f_detect, res_path, d_extr.empty() ? "FREAK":d_extr, threshhTh);
	}

	if (timeDescr)
	{
		startDescriptorTimeMeasurement(img_path, gt_path, gt_type,
			l_img_pref, r_img_pref, gt_pref,
			f_detect, d_extr, res_path);
	}

	if (loadGTM)
	{
		if (gtm_path.empty() || gtm_postf.empty() || img_path.empty() || l_img_pref.empty())
		{
			cout << "Paramters for showing GTM missing! Exiting." << endl;
			exit(1);
		}
		else
		{
			showGTM(img_path, l_img_pref, r_img_pref, gtm_path, gtm_postf);
		}
	}

	if (initGTMs)
	{
		generateMissingInitialGTMs(img_path, gt_path, gt_type,
			l_img_pref, r_img_pref, gt_pref,
			f_detect, d_extr.empty() ? "FREAK" : d_extr);
	}

	if (genGTMs)
	{
		generateGTMs(img_path, gt_path, gt_type,
			l_img_pref, r_img_pref, gt_pref,
			f_detect, d_extr.empty() ? "FREAK" : d_extr);
	}*/
}

/** @function main */
int main( int argc, char* argv[])
{
	/*ArgvParser cmd;
	SetupCommandlineParser(cmd, argc, argv);
	startEvaluation(cmd);*/

	testStereoCamGeneration(0);


	return 0;
}

void testStereoCamGeneration(int verbose, bool genSequence)
{
	srand(time(NULL));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rand_generator(seed);

	const bool testGenericAlignment = false;

	//Parameter ranges for tx and ty are down in the for-loop
	double roll_minmax[2] = { -20.0, 20.0 };
	double pitch_minmax[2] = { -20.0, 20.0 };
	double yaw_minmax[2] = { -10.0, 10.0 };
	double tx_minmax_change[2] = { 0.7, 1.3 };
	double ty_minmax_change[2] = { 0.7, 1.3 };
	double tz_minmax_change[2] = { 0.7, 1.3 };
	double roll_minmax_change[2] = { 0.7, 1.3 };
	double pitch_minmax_change[2] = { 0.7, 1.3 };
	double yaw_minmax_change[2] = { 0.7, 1.3 };
	double txy_relmax = 0.25;// 0.7; //must be between 0 and 1
	double tz_relxymax = 0.1;// 0.5; //should be between 0 and 1
	double hlp = std::max(tx_minmax_change[1] * txy_relmax, ty_minmax_change[1] * txy_relmax);
	txy_relmax = hlp >= 1.0 ? (0.99 / std::max(tx_minmax_change[1], ty_minmax_change[1])) : txy_relmax;

	CV_Assert(txy_relmax < 1.0);

	for (size_t j = 0; j < 1000; j++) //Number of performed test runs
	{
		double tx_minmax[2] = { 0.1, 5.0 };
		double ty_minmax[2] = { 0, 5.0 };

		int nrCams = 2;// std::rand() % 10 + 1;//1 //Number of different stereo camera configurations (extrinsic) per test

		//Calculate, for which extrinsic (tx, ty, tz, roll, ...) its value or range should remain the same for all nrCams different stereo camera configuratuions
		int roll_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int pitch_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int yaw_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int tx_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int ty_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int tz_equRanges = (std::rand() % 2) & (std::rand() % 2);

		vector<double> txi_start, tyi_start, tzi_start, rolli_start, pitchi_start, yawi_start;
		if (!testGenericAlignment)//If restrictions apply for the relative distances between cameras in x and y in a specific camera alignment (horizontal or vertical)
		{
			int align = 0;// rand() % 2;//0 //Choose either vertical or horizontal camera alignment
			if (align)//Vertical camera alignment
			{
				initStarVal(rand_generator, ty_minmax, tyi_start);//Generate a random value or range (chosen ramdomly) between ranges for the distance in y
				tx_minmax[0] = -txy_relmax * tyi_start.back();//Calculate the range of the x-distance based on the maximum y-distance
				tx_minmax[1] = txy_relmax * tyi_start.back();
				double tmp = -ty_minmax[0];//Negate to make sure that the reference camera is the top camera (y-component of translation vector must be negative)
				ty_minmax[0] = -ty_minmax[1];
				ty_minmax[1] = tmp;
				initStarVal(rand_generator, tx_minmax, txi_start);//Generate a random value or range (chosen ramdomly) between ranges (smaller or equal than for y) for the distance in x
				if (tyi_start.size() == 1)
				{
					tyi_start[0] = -tyi_start[0];
				}
				else
				{
					tmp = -tyi_start[0];
					tyi_start[0] = -tyi_start[1];
					tyi_start[1] = tmp;
				}
			}
			else //Horizontal camera alignment
			{
				initStarVal(rand_generator, tx_minmax, txi_start);//Generate a random value or range (chosen ramdomly) between ranges for the distance in x
				//txi_start.push_back(0.9);//REMOVE

				ty_minmax[0] = -txy_relmax * txi_start.back();//Calculate the range of the y-distance based on the maximum x-distance
				ty_minmax[1] = txy_relmax * txi_start.back();
				double tmp = -tx_minmax[0];//Negate to make sure that the reference camera is the left camera (x-component of translation vector must be negative)
				tx_minmax[0] = -tx_minmax[1];
				tx_minmax[1] = tmp;
				initStarVal(rand_generator, ty_minmax, tyi_start);//Generate a random value or range (chosen ramdomly) between ranges (smaller or equal than for x) for the distance in y
				//tyi_start.push_back(-0.5);//REMOVE
				//tyi_start.push_back(0.5);//REMOVE

				if (txi_start.size() == 1)
				{
					txi_start[0] = -txi_start[0];
				}
				else
				{
					tmp = -txi_start[0];
					txi_start[0] = -txi_start[1];
					txi_start[1] = tmp;
				}
			}
		}
		else //Complete random alignment (x- and y-direction of translation vector) of the stereo cameras without any restrictions about relations between x and y translation
		{
			tx_minmax[0] = -tx_minmax[1];
			ty_minmax[0] = -ty_minmax[1];
			initStarVal(rand_generator, tx_minmax, txi_start);
			initStarVal(rand_generator, ty_minmax, tyi_start);
		}

		//Calculate maximum z-distance between cameras based on maximum x- and y-distance (z-distance must be smaller)
		double tzr = tz_relxymax * std::max(abs(txi_start.back()), abs(tyi_start.back()));
		double tz_minmax[2];
		tz_minmax[0] = -tzr;
		tz_minmax[1] = tzr;
		initStarVal(rand_generator, tz_minmax, tzi_start); //Calculate value or range (chosen ramdomly) for the z-distance

		initStarVal(rand_generator, roll_minmax, rolli_start); //Calculate value or range (chosen ramdomly) for the roll
		initStarVal(rand_generator, pitch_minmax, pitchi_start); //Calculate value or range (chosen ramdomly) for the pitch
		initStarVal(rand_generator, yaw_minmax, yawi_start); //Calculate value or range (chosen ramdomly) for the yaw

		//tzi_start.push_back(0.02);//REMOVE
		//rolli_start.push_back(-15.0);//REMOVE
		//rolli_start.push_back(15.0);//REMOVE
		//pitchi_start.push_back(-15.0);//REMOVE
		//pitchi_start.push_back(15.0);//REMOVE
		//yawi_start.push_back(-0.2);//REMOVE

		std::vector<std::vector<double>> tx;
		std::vector<std::vector<double>> ty;
		std::vector<std::vector<double>> tz;
		std::vector<std::vector<double>> roll;
		std::vector<std::vector<double>> pitch;
		std::vector<std::vector<double>> yaw;

		//Store the initial values or ranges of the extrinsics in the vector that holds the configurations of every of the nrCams stereo configurations
		tx.push_back(txi_start);
		ty.push_back(tyi_start);
		tz.push_back(tzi_start);
		roll.push_back(rolli_start);
		pitch.push_back(pitchi_start);
		yaw.push_back(yawi_start);

		//Generate the remaining ranges (or values) of extrinsics for the other stereo camera configurations
		//The other ranges must be within the maximum specified ranges and are only allowed to deviate from the initial calculated ranges (..._start) by a relative value (..._minmax_change)
		for (int i = 1; i < nrCams; i++)
		{
			vector<double> txi, tyi, tzi, rolli, pitchi, yawi;
			size_t wrongsignCnt = 0;
			bool cngTxy = false;
			while (txi.empty() || ((wrongsignCnt < 100) && cngTxy))
			{
				if (tx_equRanges)//If true, the tx value or range stays the same for all stereo camera configurations
					txi = txi_start;
				else
					initRangeVals(rand_generator, tx_minmax, tx_minmax_change, txi_start, txi);//Calculate the new range or value based on the first range (within maximum specified range tx_minmax)
				if (ty_equRanges)//If true, the ty value or range stays the same for all stereo camera configurations
					tyi = tyi_start;
				else
					initRangeVals(rand_generator, ty_minmax, ty_minmax_change, tyi_start, tyi);//Calculate the new range or value based on the first range (within maximum specified range ty_minmax)

				//Check if the largest absolute x- or y-value (from x-,y-values and/or ranges) is negative to ensure that the reference camera is the top or left camera
				//This is necessary, as all optimization algorithms (im LM) are based on this assumption
				double maxX, maxY;
				if (txi.size() > 1)
				{
					if (((txi[0] >= 0) && (txi[1] > 0)) ||
						(abs(txi[1]) > abs(txi[0])))
					{
						maxX = txi[1];
					}
					else
					{
						maxX = txi[0];
					}
				}
				else
				{
					maxX = txi[0];
				}
				if (tyi.size() > 1)
				{
					if (((tyi[0] >= 0) && (tyi[1] > 0)) ||
						(abs(tyi[1]) > abs(tyi[0])))
					{
						maxY = tyi[1];
					}
					else
					{
						maxY = tyi[0];
					}
				}
				else
				{
					maxY = tyi[0];
				}

				if (((abs(maxX) >= abs(maxY)) && (maxX > 0)) ||
					((abs(maxY) >= abs(maxX)) && (maxY > 0)))
				{
					cngTxy = true;//Try again if the reference camera is not the top or left camera
					txi.clear();
					tyi.clear();
				}
				else
				{
					cngTxy = false;
				}
				wrongsignCnt++;
			}
			if (cngTxy)//Abort if no valid extrinsic values or ranges were created for this stereo configuration (reference camera must be top or left camera)
				break;

			if (tz_equRanges)//If true, the tz value or range stays the same for all stereo camera configurations
				tzi = tzi_start;
			else
				initRangeVals(rand_generator, tz_minmax, tz_minmax_change, tzi_start, tzi);//Calculate the new range or value based on the first range (within maximum specified range tz_minmax)
			if (roll_equRanges)//If true, the roll value or range stays the same for all stereo camera configurations
				rolli = rolli_start;
			else
				initRangeVals(rand_generator, roll_minmax, roll_minmax_change, rolli_start, rolli);//Calculate the new range or value based on the first range (within maximum specified range roll_minmax)
			if (pitch_equRanges) //If true, the pitch value or range stays the same for all stereo camera configurations
				pitchi = pitchi_start;
			else
				initRangeVals(rand_generator, pitch_minmax, pitch_minmax_change, pitchi_start, pitchi);//Calculate the new range or value based on the first range (within maximum specified range pitch_minmax)
			if (yaw_equRanges) //If true, the yaw value or range stays the same for all stereo camera configurations
				yawi = yawi_start;
			else
				initRangeVals(rand_generator, yaw_minmax, yaw_minmax_change, yawi_start, yawi);//Calculate the new range or value based on the first range (within maximum specified range yaw_minmax)

			//Add the new camera configuration to the vector that should hold all of them in the end
			tx.push_back(txi);
			ty.push_back(tyi);
			tz.push_back(tzi);
			roll.push_back(rolli);
			pitch.push_back(pitchi);
			yaw.push_back(yawi);
		}

		//If for one or more stereo configurations no valid configuration was found (reference camera must be top or left camera), skip this test
		if (tx.size() < nrCams)
			continue;

		//Generate a random target image overlap between the stereo cameras that sould be present after LM optimization of the extrinsic parameters between the given ranges (that were generated above)
		double approxImgOverlap = getRandDoubleVal(rand_generator, 0.2, 1.0);//0.1;
		cv::Size imgSize = cv::Size(1280, 720);//Select an image size (can be arbitrary)

		//Calculate the extrinsic parameters for all given stereo configurations within the given ranges to achieve an image overlap nearest to the above generated value (in addition to a few other constraints)
		GenStereoPars newStereoPars(tx, ty, tz, roll, pitch, yaw, approxImgOverlap, imgSize);
		int err = newStereoPars.optimizeRtf(verbose);

		//Write the results to std out
		vector<cv::Mat> t_new;
		Mat K;
		vector<double> roll_new, pitch_new, yaw_new;
		newStereoPars.getEulerAngles(roll_new, pitch_new, yaw_new);
		t_new = newStereoPars.tis;
		K = newStereoPars.K1;
		cout << endl;
		cout << "**************************************************************************" << endl;
		cout << "User specified image overlap = " << std::setprecision(3) << approxImgOverlap << endl;
		cout << "f= " << std::setprecision(2) << K.at<double>(0, 0) << " cx= " << K.at<double>(0, 2) << " cy= " << K.at<double>(1, 2) << endl;
		for (size_t i = 0; i < roll_new.size(); i++)
		{
			cout << "tx= " << std::setprecision(6) << t_new[i].at<double>(0) << " ty= " << t_new[i].at<double>(1) << " tz= " << t_new[i].at<double>(2) << endl;
			cout << "roll= " << std::setprecision(3) << roll_new[i] << " pitch= " << pitch_new[i] << " yaw= " << yaw_new[i] << endl;
		}
		cout << "**************************************************************************" << endl << endl;

		//Generate a trajectory/image sequence using the above calculated stereo camera configurations
		if ((err == 0) && genSequence)
		{
			std::vector<cv::Mat> Rv, tv;
			cv::Mat K_1, K_2;
			if (newStereoPars.getCamPars(Rv, tv, K_1, K_2))
			{
				for (size_t i = 0; i < 10; i++)
				{
					genNewSequence(Rv, tv, K_1, K_2, imgSize);
				}
			}
		}
	}
}

int genNewSequence(std::vector<cv::Mat>& Rv, std::vector<cv::Mat>& tv, cv::Mat& K_1, cv::Mat& K_2, cv::Size &imgSize)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rand_generator(seed);

	const int maxTrackElements = 100;
	double closedLoopMaxXYRatioRange[2] = { 1.0, 3.0 }; //Multiplier range that specifies how much larger the expension of a track in x-direction can be compared to y (y = x / value from this range).
	double closedLoopMaxXZRatioRange[2] = { 0.1, 1.0 }; //Multiplier range that specifies how much smaller/larger the expension of a track in z-direction can be compared to x.
	double closedLoopMaxElevationAngleRange[2] = { 0, 3.14 / 16.0}; //Only for closed loop. Angle range for the z-component (y in the camera coordinate system) of the ellipsoide (must be in the range -pi/2 <= angle <= pi/2). For a fixed angle, it defines the ellipse on the ellipsoide with a fixed value of z (y in the camera coordinate system).
	const bool enableFlightMode = false; //Only for closed loop. If enabled, the elevation angle teta of the ellipsoide is continuously changed within the range closedLoopMaxElevationAngleRange to get different height values (z-component of ellipsoide (y in the camera coordinate system)) along the track

	size_t nFramesPerCamConf = 5;//Number of consecutive frames on a track with the same stereo configuration

	double minInlierRange[2] = { 0.1, 0.5 };//Range of the minimum inlier ratio
	double maxInlierRange[2] = { 0.55, 1.0 };//Range of the maximum inlier ratio bounded by minInlierRange

	size_t minNrInlierRange[2] = { 1, 50 };//Range of the minimum number of true positive correspondences
	size_t maxNrInlierRange[2] = { 100, 10000 };//Range of the maximum number of true positive correspondences

	double minKeypDistRange[2] = { 1.0, 10.0 };//Range of the minimum distance between keypoints

	size_t MaxNrDepthAreasPReg = 30;//Maximum number of depth areas per image region

	double relCamVelocityRange[2] = { 0.1, 5.0 };//Relative camera velocity compared to the basline length

	double rollRange[2] = { -3.0, 3.0 };//Roll angle (x-axis) for the first camera centre. This rotation can change the camera orientation for which without rotation the z - component of the relative movement vector coincides with the principal axis of the camera
	double pitchRange[2] = { -10.0, 10.0 };//Pitch angle (y-axis) for the first camera centre. This rotation can change the camera orientation for which without rotation the z - component of the relative movement vector coincides with the principal axis of the camera
	double yawRange[2] = { -2.0, 2.0 };//Yaw angle (z-axis) for the first camera centre. This rotation can change the camera orientation for which without rotation the z - component of the relative movement vector coincides with the principal axis of the camera

	double minRelAreaRangeMovObjsRange[2] = { 0, 0.1 };//Range of the lower border of the relative area range of moving objects. Minimum area range of moving objects relative to the image area at the beginning (when a moving object is first introduced).
	double maxRelAreaRangeMovObjsRange[2] = { 0.15, 0.99 };//Range of the upper border of the relative area range of moving objects. Maximum area range of moving objects relative to the image area at the beginning (when a moving object is first introduced).

	double minRelVelocityMovObj = 0.1;//Minimum relative object velocity compared to camera velocity
	double maxRelVelocityMovObj = 20.0;//Maximum relative object velocity compared to camera velocity

	double minPortionMovObj = 0.1;//Minimal portion of used correspondences by moving objects
	double maxPortionMovObj = 0.85;//Maximal portion of used correspondences by moving objects

	//Generate a random camera track
	int closedLoop = rand() % 2;
	int staticCamMovement = rand() % 2;
	int nrTrackElements = rand() % maxTrackElements + 1;
	if (nrTrackElements == 1)
		staticCamMovement = 1;
	std::vector<cv::Mat> camTrack;
	if ((closedLoop == 0) || (staticCamMovement == 1))
	{
		double xy, xz;
		Mat tr1 = Mat(3, 1, CV_64FC1);
		//Generate a random camera movement vector based on the ranges given. The scaling is not important. The scaling and track segments for every image pair are caluclated within the class genStereoSequ 
		tr1.at<double>(0) = 1.0;
		xy = getRandDoubleVal(rand_generator, closedLoopMaxXYRatioRange[0], closedLoopMaxXYRatioRange[1]);
		xz = getRandDoubleVal(rand_generator, closedLoopMaxXZRatioRange[0], closedLoopMaxXZRatioRange[1]);
		tr1.at<double>(1) = 1.0 / xy;
		tr1.at<double>(2) = xz;
		camTrack.push_back(tr1.clone());
		if (staticCamMovement != 1)//Add more track segments (ignoring scale of the whole track). Otherwise, the first generated vector defines the not changing direction of the track.
		{
			std::normal_distribution<double> distributionNX2(1.0, 0.25);
			for (int i = 0; i < nrTrackElements; i++)
			{
				xy = getRandDoubleVal(rand_generator, closedLoopMaxXYRatioRange[0], closedLoopMaxXYRatioRange[1]);
				xz = getRandDoubleVal(rand_generator, closedLoopMaxXZRatioRange[0], closedLoopMaxXZRatioRange[1]);
				tr1.at<double>(0) *= distributionNX2(rand_generator);//New x-direction depends on old x-direction
				tr1.at<double>(1) = (2.0 * (tr1.at<double>(1) * distributionNX2(rand_generator)) + tr1.at<double>(0) / xy) / 3.0;//New y-direction mainly depends on old y-direction but also on a random part which depends on the x-direction
				tr1.at<double>(2) = (2.0 * (tr1.at<double>(2) * distributionNX2(rand_generator)) + tr1.at<double>(0) * xz) / 3.0;//New z-direction mainly depends on old z-direction but also on a random part which depends on the x-direction
				camTrack.push_back(camTrack.back() + tr1);
			}
		}
	}
	else //Track with loop closure (last track position is near the first but not the same)
	{
		//Take an ellipsoide with random parameters
		double b, a = 1.0, c, ab, ac;
		b = a / getRandDoubleVal(rand_generator, closedLoopMaxXYRatioRange[0], closedLoopMaxXYRatioRange[1]);
		c = a * getRandDoubleVal(rand_generator, closedLoopMaxXZRatioRange[0], closedLoopMaxXZRatioRange[1]);
		double tmp1 = getRandDoubleVal(rand_generator, closedLoopMaxElevationAngleRange[0], closedLoopMaxElevationAngleRange[1]);
		double tmp2 = getRandDoubleVal(rand_generator, closedLoopMaxElevationAngleRange[0], closedLoopMaxElevationAngleRange[1]);
		double theta = min(tmp1, tmp2);
		double theta2 = max(tmp1, tmp2);
		double thetaPiece = (theta2 - theta) / (double)(nrTrackElements - 1);//Variation in height from one track element to the next based on the angle theta. Only used in enableFlightMode.
		double phiPiece = (2.0 * M_PI - M_PI / (double)(10 * maxTrackElements))/ (double)(nrTrackElements - 1);//The track is nearly a loop closure (phi from 0 to nearly 2 pi)
		//camTrack.push_back(Mat::zeros(3, 1, CV_64FC1));
		double phi = 0;// phiPiece;
		for (int i = 0; i < nrTrackElements; i++)
		{
			Mat tr1 = Mat(3, 1, CV_64FC1);
			tr1.at<double>(0) = a * cos(theta) * cos(phi);
			tr1.at<double>(2) = b * cos(theta) * sin(phi);
			tr1.at<double>(1) = c * sin(theta);
			phi += phiPiece;
			if (enableFlightMode)
				theta += thetaPiece;
			camTrack.push_back(tr1.clone());
		}
	}

	int onlyOneInlierRat = rand() % 2;
	//Inlier ratio range
	std::pair<double, double> inlRatRange;
	if (onlyOneInlierRat == 0)//Generate a random inlier ratio range within a given range for the image pairs
	{
		inlRatRange = std::make_pair(getRandDoubleVal(rand_generator, minInlierRange[0], minInlierRange[1]), 1.0);
		inlRatRange.second = getRandDoubleVal(rand_generator, max(inlRatRange.first + 0.01, maxInlierRange[0]), max(min(inlRatRange.first + 0.02, 1.0), maxInlierRange[1]));
	}
	else//Only 1 fixed inlier ratio over all image pairs
	{
		inlRatRange = std::make_pair(getRandDoubleVal(rand_generator, minInlierRange[0], maxInlierRange[1]), 1.0);
		inlRatRange.second = inlRatRange.first;
	}

	int inlChangeRate = rand() % 3;
	//Inlier ratio change rate from pair to pair. If 0, the inlier ratio within the given range is always the same for every image pair. 
	//If 100, the inlier ratio is chosen completely random within the given range.
	//For values between 0 and 100, the inlier ratio selected is not allowed to change more than this factor (not percentage) from the last inlier ratio.
	double inlRatChanges = 0;
	if (inlChangeRate == 1)
	{
		inlRatChanges = getRandDoubleVal(rand_generator, 0, 100.0);
	}
	else if (inlChangeRate == 2)
	{
		inlRatChanges = 100.0;
	}

	int onlyOneTPNr = rand() % 2;
	//# true positives range
	std::pair<size_t, size_t> truePosRange;
	if (onlyOneTPNr == 0)
	{
		std::uniform_int_distribution<size_t> distribution1(minNrInlierRange[0], minNrInlierRange[1]);
		truePosRange = std::make_pair(distribution1(rand_generator), 1);
		std::uniform_int_distribution<size_t> distribution2(max(truePosRange.first, maxNrInlierRange[0]), max(truePosRange.first + 1, maxNrInlierRange[1]));
		truePosRange.second = distribution2(rand_generator);
	}
	else
	{
		std::uniform_int_distribution<size_t> distribution1(minNrInlierRange[0], maxNrInlierRange[1]);
		truePosRange = std::make_pair(distribution1(rand_generator), 1);
		truePosRange.second = truePosRange.first;
	}

	int tpChangeRate = rand() % 3;
	//True positives change rate from pair to pair. If 0, the true positives within the given range are always the same for every image pair. 
	//If 100, the true positives are chosen completely random within the given range.
	//For values between 0 and 100, the true positives selected are not allowed to change more than this factor from the true positives.
	double truePosChanges = 0;
	if (tpChangeRate == 1)
	{
		truePosChanges = getRandDoubleVal(rand_generator, 0, 100.0);
	}
	else if (tpChangeRate == 2)
	{
		truePosChanges = 100.0;
	}

	//Functionality for the next few values is not implemented yet!!!!!!!!!!!
	bool keypPosErrType = false;
	std::pair<double, double> keypErrDistr = std::make_pair(0, 0.5);
	std::pair<double, double> imgIntNoise = std::make_pair(0, 5.0);
	//ends here

	//min. distance between keypoints
	double minKeypDist = getRandDoubleVal(rand_generator, minKeypDistRange[0], minKeypDistRange[1]);
	
	//portion of correspondences at depths
	depthPortion corrsPerDepth = depthPortion(getRandDoubleVal(rand_generator, 0, 1.0), getRandDoubleVal(rand_generator, 0, 1.0), getRandDoubleVal(rand_generator, 0, 1.0));

	int useMultcorrsPerRegion = rand() % 3;
	//List of portions of image correspondences at regions (Matrix must be 3x3). This is the desired distribution of features in the image. Maybe doesnt hold: Also depends on 3D-points from prior frames.
	std::vector<cv::Mat> corrsPerRegion;
	//If useMultcorrsPerRegion==0, the portions are randomly initialized. There are as many matrices generated as the number of image pairs divided by corrsPerRegRepRate
	if (useMultcorrsPerRegion == 1)//The same correspondence distribution over the image is used for all image pairs
	{
		cv::Mat corrsPReg(3, 3, CV_64FC1);
		cv::randu(corrsPReg, Scalar(0), Scalar(1.0));
		corrsPReg /= sum(corrsPReg)[0];
		corrsPerRegion.push_back(corrsPReg.clone());
	}
	else if (useMultcorrsPerRegion == 2)//Generates a random number (between 1 and the maximum number of image pairs) of different correspondence distributions
	{
		int nr_elems = rand() % (int)(nFramesPerCamConf * Rv.size() - 1) + 1;
		for (int i = 0; i < nr_elems; i++)
		{
			cv::Mat corrsPReg(3, 3, CV_64FC1);
			cv::randu(corrsPReg, Scalar(0), Scalar(1.0));
			corrsPReg /= sum(corrsPReg)[0];
			corrsPerRegion.push_back(corrsPReg.clone());
		}
	}

	int setcorrsPerRegRepRate0 = rand() % 2;
	//Repeat rate of portion of correspondences at regions. 
	//If more than one matrix of portions of correspondences at regions is provided, this number specifies the number of frames for which such a matrix is valid. 
	//After all matrices are used, the first one is used again. If 0 and no matrix of portions of correspondences at regions is provided, as many random matrizes as frames are randomly generated.
	size_t corrsPerRegRepRate = 0;
	if (setcorrsPerRegRepRate0 == 0)
	{
		corrsPerRegRepRate = (size_t)rand() % (nFramesPerCamConf * Rv.size() - 1);
	}
	
	int depthsPerRegionEmpty = rand() % 2;
	//Portion of depths per region (must be 3x3). For each of the 3x3=9 image regions, the portion of near, mid, and far depths can be specified. If the overall depth definition is not met, this tensor is adapted.Maybe doesnt hold: Also depends on 3D - points from prior frames.
	std::vector<std::vector<depthPortion>> depthsPerRegion;
	//If depthsPerRegionEmpty==1, depthsPerRegion is initialized randomly within class genStereoSequ
	if (depthsPerRegionEmpty == 0)
	{
		depthsPerRegion.resize(3, std::vector<depthPortion>(3));
		for (size_t y = 0; y < 3; y++)
		{
			for (size_t x = 0; x < 3; x++)
			{
				depthsPerRegion[y][x] = depthPortion(getRandDoubleVal(rand_generator, 0, 1.0), getRandDoubleVal(rand_generator, 0, 1.0), getRandDoubleVal(rand_generator, 0, 1.0));
			}
		}
	}

	int nrDepthAreasPRegEmpty = rand() % 2;
	//Min and Max number of connected depth areas per region (must be 3x3). The minimum number (first) must be larger 0. 
	//The maximum number is bounded by a minimum area of 16 pixels. Maybe doesnt hold: Also depends on 3D - points from prior frames.
	std::vector<std::vector<std::pair<size_t, size_t>>> nrDepthAreasPReg;
	//if nrDepthAreasPRegEmpty==1, nrDepthAreasPReg is initialized randomly within class genStereoSequ
	if (nrDepthAreasPRegEmpty == 0)
	{
		nrDepthAreasPReg.resize(3, std::vector<std::pair<size_t, size_t>>(3));
		for (size_t y = 0; y < 3; y++)
		{
			for (size_t x = 0; x < 3; x++)
			{
				nrDepthAreasPReg[y][x] = std::pair<size_t, size_t>(1, (size_t)rand() % MaxNrDepthAreasPReg + 1);
				int nrDepthAreasPRegEqu = rand() % 2;
				if (nrDepthAreasPRegEqu == 1)
				{
					nrDepthAreasPReg[y][x].first = nrDepthAreasPReg[y][x].second;
				}
			}
		}
	}

	//Functionality not implemented:
	double lostCorrPor = 0;
	//end

	//Relative velocity of the camera movement (value between 0 and 10; must be larger 0). The velocity is relative to the baseline length between the stereo cameras
	double relCamVelocity = getRandDoubleVal(rand_generator, relCamVelocityRange[0], relCamVelocityRange[1]);

	//Rotation matrix of the first camera centre.
	//This rotation can change the camera orientation for which without rotation the z - component of the relative movement vector coincides with the principal axis of the camera
	cv::Mat R = eulerAnglesToRotationMatrix(getRandDoubleVal(rand_generator, rollRange[0], rollRange[1]) * M_PI / 180.0,
		getRandDoubleVal(rand_generator, pitchRange[0], pitchRange[1]) * M_PI / 180.0,
		getRandDoubleVal(rand_generator, yawRange[0], yawRange[1]) * M_PI / 180.0);
    /*cv::Mat R = eulerAnglesToRotationMatrix(0,
                                            -90.0 * M_PI / 180.0,
                                            0);*/

	//Number of moving objects in the scene
	size_t nrMovObjs = (size_t)rand() % 20;

	int startPosMovObjsEmpty = rand() % 2;
	//Possible starting positions of moving objects in the image (must be 3x3 boolean (CV_8UC1))
	cv::Mat startPosMovObjs;
	//if startPosMovObjsEmpty==1, nrDepthAreasPReg is initialized randomly within class genStereoSequ
	if (startPosMovObjsEmpty == 0)
	{
		startPosMovObjs = Mat(3, 3, CV_8UC1);
		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				startPosMovObjs.at<unsigned char>(y, x) = (unsigned char)(rand() % 2);
			}
		}
	}

	//Relative area range of moving objects. Area range relative to the image area at the beginning.
	std::pair<double, double> relAreaRangeMovObjs = std::make_pair(getRandDoubleVal(rand_generator, minRelAreaRangeMovObjsRange[0], minRelAreaRangeMovObjsRange[1]), 0.1);
	relAreaRangeMovObjs.second = getRandDoubleVal(rand_generator, max(maxRelAreaRangeMovObjsRange[0], relAreaRangeMovObjs.first), maxRelAreaRangeMovObjsRange[1]);

	int movObjDepthVersion = rand() % 4;
	//Depth of moving objects. Moving objects are always visible and not covered by other static objects. 
	//If the number of paramters is 1, this depth class (near, mid, or far) is used for every object. 
	//If the number of paramters is equal "nrMovObjs", the corresponding depth is used for every object. 
	//If the number of parameters is smaller and between 2 and 3, the depths for the moving objects are selected uniformly distributed from the given depths. 
	//For a number of paramters larger 3 and unequal to "nrMovObjs", a portion for every depth that should be used can be defined 
	//(e.g. 3 x far, 2 x near, 1 x mid -> 3 / 6 x far, 2 / 6 x near, 1 / 6 x mid).
	std::vector<depthClass> movObjDepth;
	if (movObjDepthVersion == 0)//Use the same random depth for every moving object
	{
		movObjDepth.push_back(getRandDepthClass());
	}
	if (movObjDepthVersion == 1)//Use a random depth for every moving object
	{
		movObjDepth.resize(nrMovObjs);
		for (size_t i = 0; i < nrMovObjs; i++)
		{
			movObjDepth[i] = getRandDepthClass();
		}
	}
	else if (movObjDepthVersion == 2)//Specify 2 or 3 random depth classes that are used by every moving object (uniformly distributed)
	{
		int nrClasses = max(min((rand() % 2) + 2, (int)nrMovObjs - 1), 1);
		for (int i = 0; i < nrClasses; i++)
		{
			movObjDepth.push_back(getRandDepthClass());
		}
	}
	else//Generates a random distribution of the 3 depth classes which is used to select a depth class for a moving object
	{
		int nrClasses = rand() % 1000 + 1;
		for (int i = 0; i < nrClasses; i++)
		{
			movObjDepth.push_back(getRandDepthClass());
		}
	}

	int movObjDirEmpty = rand() % 2;
	//Movement direction of moving objects relative to camera movement (must be 3x1). 
	//The movement direction is linear and does not change if the movement direction of the camera changes. The moving object is removed, if it is no longer visible in both stereo cameras.
	cv::Mat movObjDir;
	if (movObjDirEmpty == 0)
	{
		movObjDir = Mat(3, 1, CV_64FC1);
		cv::randu(movObjDir, Scalar(0), Scalar(1.0));
	}

	//Relative velocity range of moving objects based on relative camera velocity. Values between 0 and 100; Must be larger 0;
	std::pair<double, double> relMovObjVelRange = std::make_pair(minRelVelocityMovObj, getRandDoubleVal(rand_generator, minRelVelocityMovObj, maxRelVelocityMovObj));

	int minMovObjCorrPortionType = rand() % 3;
	int minMovObjCorrPortionType1 = ((rand() % 3) | minMovObjCorrPortionType) & 2;
	minMovObjCorrPortionType = minMovObjCorrPortionType1 == 2 ? minMovObjCorrPortionType1 : minMovObjCorrPortionType;//Higher propability to get a 2
	//Minimal portion of correspondences on moving objects for removing them. 
	//If the portion of visible correspondences drops below this value, the whole moving object is removed. 
	//Zero means, that the moving object is only removed if there is no visible correspondence in the stereo pair. 
	//One means, that a single missing correspondence leads to deletion. Values between 0 and 1;
	double minMovObjCorrPortion = 0;
	if (minMovObjCorrPortionType == 1)
	{
		minMovObjCorrPortion = 1.0;
	}
	else if (minMovObjCorrPortionType == 2)
	{
		minMovObjCorrPortion = getRandDoubleVal(rand_generator, 0, 1.0);
	}

	//Portion of correspondences on moving object (compared to static objects). It is limited by the size of the objects visible in the images and the minimal distance between correspondences.
	double CorrMovObjPort = getRandDoubleVal(rand_generator, minPortionMovObj, maxPortionMovObj);

	//Minimum number of moving objects over the whole track. 
	//If the number of moving obects drops below this number during camera movement, as many new moving objects are inserted until "nrMovObjs" is reached. 
	//If 0, no new moving objects are inserted if every preceding object is out of sight.
	size_t minNrMovObjs = (size_t)rand() % (nrMovObjs + 1);

	//Set parameters
	StereoSequParameters stereoSequPars(camTrack,
		nFramesPerCamConf,
		inlRatRange,
		inlRatChanges,
		truePosRange,
		truePosChanges,
		keypPosErrType,
		keypErrDistr,
		imgIntNoise,
		minKeypDist,
		corrsPerDepth,
		corrsPerRegion,
		corrsPerRegRepRate,
		depthsPerRegion,
		nrDepthAreasPReg,
		//lostCorrPor,
		relCamVelocity,
		R.clone(),
		nrMovObjs,
		startPosMovObjs.clone(),
		relAreaRangeMovObjs,
		movObjDepth,
		movObjDir.clone(),
		relMovObjVelRange,
		minMovObjCorrPortion,
		CorrMovObjPort,
		minNrMovObjs
	);

	//Initialize system
	int32_t verbose = SHOW_INIT_CAM_PATH | SHOW_BUILD_PROC_STATIC_OBJ | SHOW_STATIC_OBJ_DISTANCES | SHOW_STATIC_OBJ_CORRS_GEN |
            SHOW_STATIC_OBJ_3D_PTS | SHOW_MOV_OBJ_3D_PTS;
	genStereoSequ stereoSequ(imgSize, K_1, K_2, Rv, tv, stereoSequPars, verbose);

	//Generate Sequence
	stereoSequ.startCalc();

	return 0;
}

depthClass getRandDepthClass()
{
	int selDepthClass = rand() % 3;
	depthClass res = depthClass::MID;
	switch (selDepthClass)
	{
	case 0:
		res = depthClass::NEAR;
		break;
	case 1:
		res = depthClass::MID;
		break;
	case 2:
		res = depthClass::FAR;
		break;
	default:
		break;
	}
	return res;
}

double getRandDoubleVal(std::default_random_engine rand_generator, double lowerBound, double upperBound)
{
	rand_generator = std::default_random_engine((unsigned int)std::rand());
	std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
	return distribution(rand_generator);
}

void initStarVal(default_random_engine rand_generator, double *range, vector<double>& startVec)
{
	int isRange = std::rand() % 2;
	for (size_t i = 0; i < 1; i++)
	{
		isRange |= std::rand() % 2;
	}
	startVec.push_back(getRandDoubleVal(rand_generator, range[0], range[1]));
	if (isRange)
		startVec.push_back(getRandDoubleVal(rand_generator, startVec.back(), range[1]));
}

void initRangeVals(default_random_engine rand_generator, double *range, double *relMinMaxCh, vector<double> startVec, vector<double>& actVec)
{
	double newRange[2];
	if (startVec.size() == 1)
	{
		if (startVec[0] >= 0)
		{
			newRange[0] = std::max(startVec[0] * relMinMaxCh[0], range[0]);
			newRange[1] = std::min(startVec[0] * relMinMaxCh[1], range[1]);
		}
		else
		{
			newRange[0] = std::max(startVec[0] * relMinMaxCh[1], range[0]);
			newRange[1] = std::min(startVec[0] * relMinMaxCh[0], range[1]);
		}
	}
	else//2
	{
		if (startVec[0] >= 0)
			newRange[0] = std::max(startVec[0] * relMinMaxCh[0], range[0]);
		else
			newRange[0] = std::max(startVec[0] * relMinMaxCh[1], range[0]);
		if (startVec[1] >= 0)
			newRange[1] = std::min(startVec[1] * relMinMaxCh[1], range[1]);
		else
			newRange[1] = std::min(startVec[1] * relMinMaxCh[0], range[1]);
	}
	initStarVal(rand_generator, newRange, actVec);
}


/** @function main */
//void main( int argc, char* argv[])
//{
//	//IMPORTANT: When calling this program only one descriptor type (binary or vector) is allowed for all function calls. Moreover this program is design for 64Bit machines.
//	double maxInliers = 0;
//	int idxMaxInliers, idxMaxInliers1;
//	string filepath, fileprefl, fileprefr, filepathflow, fileprefflow;
//	Mat     src[2];
//
//	/*filepath = "C:\\work\\work_applications\\matching_GMbSOF\\Test_data\\Test_data_homography\\wall";
//	filepathflow = filepath;*/
//	filepath = "C:\\work\\work_applications\\matching_GMbSOF\\Test_data\\KITTI\\training\\images_bmp";
//	filepathflow = "C:\\work\\work_applications\\matching_GMbSOF\\Test_data\\KITTI\\training\\res_flow_filled";//"C:\\work";
//
//	/*filepath = "C:\\work\\work_applications\\matching_GMbSOF\\Test_data\\KITTI\\training\\disp_for_test\\imgs_first_bmp";
//	filepathflow = "C:\\work\\work_applications\\matching_GMbSOF\\Test_data\\KITTI\\training\\disp_for_test\\results\\res";*/
//
//	/*filepath = "C:\\work\\work_applications\\matching_GMbSOF\\Test_data\\KITTI\\training\\only_few_imgs_test";
//	filepathflow = "C:\\work\\work_applications\\matching_GMbSOF\\Test_data\\KITTI\\training\\only_few_imgs_test\\flow";*/
//
//	/*std::vector<std::string> fnames;
//	readHomographyFiles(filepath, "H1to", fnames);
//	Mat H;*/
//	
//
//	fileprefl = "left_";//"left_cam_"; //letztes Zeichen MUSS "_" sein (im Filename vor der Zahl)
//	fileprefr = "right_";//"right_cam_"; //letztes Zeichen MUSS "_" sein (im Filename vor der Zahl)
//	fileprefflow = "disp_filled_";
//
//	//fileprefl = "img_";//"left_cam_"; //letztes Zeichen MUSS "_" sein (im Filename vor der Zahl)
//	//fileprefr = "right_";//"right_cam_"; //letztes Zeichen MUSS "_" sein (im Filename vor der Zahl)
//	//fileprefflow = "H1to";
//
//	/*startTimeMeasurement(filepath, filepathflow, 2, 
//						 fileprefl, fileprefr, fileprefflow,
//						 "FAST", "FREAK", "CASCHASH",
//						 true, "abc",true,1.0,2,2,"abc","abc");*/
//
//	/*startInlierRatioMeasurement(filepath, filepathflow, 1, 
//								fileprefl, fileprefr, fileprefflow,
//								"FAST", "FREAK", "CASCHASH",
//								true, "abc", 1, 2, true, 
//								2, 2, "abc","abc");*/
//
//	/*startQualPMeasurement(filepath, filepathflow, 2, 
//						 fileprefl, fileprefr, fileprefflow,
//						 "FAST", "FREAK", "CASCHASH",
//						 true, "abc", true,0.6,2,2,"abc","abc");*/
//
//	/*testGMbSOFthreshold(filepath, filepathflow, 0, 
//						 fileprefl, fileprefr, fileprefflow,
//						 "FAST", "FREAK",
//						 "abc", 0.55, 2, "abc");*/
//
//	vector<string> filenamesl,filenamesr,filenamesflow;
//	loadStereoSequence(filepath, fileprefl, fileprefr, filenamesl, filenamesr);
//	loadImageSequence(filepathflow, fileprefflow, filenamesflow);
//	//loadImageSequence(filepath, "img_", filenamesl);
//	//
//	//std::vector<std::string> fnames;
//	//readHomographyFiles(filepath, "H1to", fnames);
//	//std::vector<cv::Mat> Hs(fnames.size());
//	//cv::Mat H;
//	//int err;
//	//for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//	//{
//	//	readHomographyFromFile(filepath, fnames[idx1], &(Hs[idx1]));
//	//}
//	//if(filenamesl.size() != (fnames.size() + 1))
//	//{
//	//	cout << "Wrong number of provided images or homographies in the specified path!" << endl;
//	//	exit(0);
//	//}
//	//if(fnames.size() < 30)
//	//{
//	//	src[0] = imread(filepath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//	//	for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//	//	{
//	//		H = Hs[idx1];
//	//		src[1] = imread(filepath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//	//		GMbSOF_matcher mymatcher(src[0], src[1], "FAST", "FREAK", H, false, 0.3);
//	//		err = mymatcher.performMatching(1.0);
//	//		if(!err)
//	//		{
//	//			mymatcher.showMatches(2);
//	//			if(maxInliers < mymatcher.positivesGT)
//	//			{
//	//				maxInliers = mymatcher.positivesGT;
//	//				idxMaxInliers = 0;
//	//				idxMaxInliers1 = idx1 + 1;
//	//			}
//	//		}
//	//	}
//	//	for(int idx1 = 0; idx1 < (int)fnames.size() - 1; idx1++)
//	//	{
//	//		for(int idx2 = idx1 + 1; idx2 < (int)fnames.size(); idx2++)
//	//		{
//	//			H = (Hs[idx2].inv() * Hs[idx1]).inv();
//	//			src[0] = imread(filepath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//	//			src[1] = imread(filepath + "\\" + filenamesl[idx2 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//	//			GMbSOF_matcher mymatcher(src[0], src[1], "FAST", "FREAK", H, false, 0.3);
//	//			err = mymatcher.performMatching(1.0);
//	//			if(!err)
//	//			{
//	//				//mymatcher.showMatches(2);
//	//				if(maxInliers < mymatcher.positivesGT)
//	//				{
//	//					maxInliers = mymatcher.positivesGT;
//	//					idxMaxInliers = idx1 + 1;
//	//					idxMaxInliers1 = idx2 + 1;
//	//				}
//	//			}
//	//		}
//	//	}
//	//}
//	//else
//	//{
//	//	src[0] = imread(filepath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//	//	for(int idx1 = 0; idx1 < (int)fnames.size(); idx1++)
//	//	{
//	//		H = Hs[idx1];
//	//		src[1] = imread(filepath + "\\" + filenamesl[idx1 + 1],CV_LOAD_IMAGE_GRAYSCALE);
//	//		GMbSOF_matcher mymatcher(src[0], src[1], "FAST", "FREAK", H, false, 0.3);
//	//		err = mymatcher.performMatching(1.0);
//	//		if(!err)
//	//		{
//	//			if(maxInliers < mymatcher.positivesGT)
//	//			{
//	//				maxInliers = mymatcher.positivesGT;
//	//				idxMaxInliers = 0;
//	//				idxMaxInliers1 = idx1 + 1;
//	//			}
//	//		}
//	//	}
//	//}
//
//	//src[0] = imread(filepath + "\\" + filenamesl[0],CV_LOAD_IMAGE_GRAYSCALE);
//	for(size_t i = 143; i < filenamesl.size();i++)
//	{
//		int err;
//		Mat flowimg;
//		src[0] = imread(filepath + "\\" + filenamesl[i],CV_LOAD_IMAGE_GRAYSCALE);
//		src[1] = imread(filepath + "\\" + filenamesr[i],CV_LOAD_IMAGE_GRAYSCALE);
//		convertImageFlowFile(filepathflow, filenamesflow[i], &flowimg);//, 32.0f);
//		//convertImageDisparityFile(filepathflow, filenamesflow[i], &flowimg);
//		/*src[1] = imread(filepath + "\\" + filenamesl[i+1],CV_LOAD_IMAGE_GRAYSCALE);
//		readHomographyFromFile(filepath, fnames[i], &H);*/
//
//		//GMbSOF_matcher mymatcher(src[0], src[1], "FAST", "FREAK", H, false, 0.3);
//		GMbSOF_matcher mymatcher(src[0], src[1], "FAST", "FREAK", flowimg, true, 0.1);
//		//HirClustIdx_matcher mymatcher(src[0], src[1], "FAST", "FREAK", flowimg, true, true);
//		//LSHidx_matcher mymatcher(src[0], src[1], "FAST", "FREAK", flowimg, true, true);
//		//Linear_matcher mymatcher(src[0], src[1], "FAST", "FREAK", flowimg, true, true);
//		//CascadeHashing_matcher mymatcher(src[0], src[1], "FAST", "FREAK", flowimg, true);
//		//GeometryAware_matcher mymatcher(src[0], src[1], "SIFT", "FREAK", flowimg, true);
//		//VFCknn_matcher mymatcher(src[0], src[1], "FAST", "FREAK", flowimg, 3);
//		err = mymatcher.performMatching(0.55);
//		/*if(!err)
//		{
//			mymatcher.showMatches(2);
//			err = mymatcher.refineMatches();
//			if(!err)
//			{
//				mymatcher.showMatches(2, true);
//			}
//		}*/
//
//		if(maxInliers < mymatcher.positivesGT)
//		{
//			maxInliers = mymatcher.positivesGT;
//			idxMaxInliers = (int)i;
//		}
//	}
//
//	//cout << "Number of max. Inliers: " << maxInliers << " img idx1: " << idxMaxInliers << " img idx2: "<< idxMaxInliers1 << endl;
//	//src[0] = imread(filepath + "\\" + filenamesl[idxMaxInliers],CV_LOAD_IMAGE_GRAYSCALE);
//	//namedWindow( "Channel 1", WINDOW_AUTOSIZE );// Create a window for display.
//	//imshow( "Channel 1", src[0] );
//	//cv::waitKey(0);
//}

