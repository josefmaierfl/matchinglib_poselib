#include "eval_start.h"
#include "argvparser.h"

#include "io_data.h"

#include "loadGTMfiles.h"

//#include <opencv2/imgproc/imgproc.hpp>

//#include "PfeImgFileIO.h"

#include "getStereoCameraExtr.h"
#include <time.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace CommandLineProcessing;

double getRandDoubleVal(std::default_random_engine rand_generator, double lowerBound, double upperBound);
void initStarVal(default_random_engine rand_generator, double *range, vector<double>& startVec);
void initRangeVals(default_random_engine rand_generator, double *range, double *relMinMaxCh, vector<double> startVec, vector<double>& actVec);
void testStereoCamGeneration(int verbose = 0);

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

	testStereoCamGeneration(-100);


	return 0;
}

void testStereoCamGeneration(int verbose)
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
	double txy_relmax = 0.7; //must be between 0 and 1
	double tz_relxymax = 0.5; //should be between 0 and 1
	double hlp = std::max(tx_minmax_change[1] * txy_relmax, ty_minmax_change[1] * txy_relmax);
	txy_relmax = hlp >= 1.0 ? (0.99 / std::max(tx_minmax_change[1], ty_minmax_change[1])) : txy_relmax;

	CV_Assert(txy_relmax < 1.0);

	for (size_t j = 0; j < 1000; j++)
	{
		double tx_minmax[2] = { 0.1, 5.0 };
		double ty_minmax[2] = { 0, 5.0 };

		int nrCams = std::rand() % 10 + 1;//1

		int roll_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int pitch_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int yaw_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int tx_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int ty_equRanges = (std::rand() % 2) & (std::rand() % 2);
		int tz_equRanges = (std::rand() % 2) & (std::rand() % 2);

		vector<double> txi_start, tyi_start, tzi_start, rolli_start, pitchi_start, yawi_start;
		if (!testGenericAlignment)
		{
			int align = rand() % 2;//0
			if (align)
			{
				initStarVal(rand_generator, ty_minmax, tyi_start);
				tx_minmax[0] = -txy_relmax * tyi_start.back();
				tx_minmax[1] = txy_relmax * tyi_start.back();
				double tmp = -ty_minmax[0];
				ty_minmax[0] = -ty_minmax[1];
				ty_minmax[1] = tmp;
				initStarVal(rand_generator, tx_minmax, txi_start);
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
			else
			{
				initStarVal(rand_generator, tx_minmax, txi_start);
				//txi_start.push_back(0.9);//REMOVE

				ty_minmax[0] = -txy_relmax * txi_start.back();
				ty_minmax[1] = txy_relmax * txi_start.back();
				double tmp = -tx_minmax[0];
				tx_minmax[0] = -tx_minmax[1];
				tx_minmax[1] = tmp;
				initStarVal(rand_generator, ty_minmax, tyi_start);
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
		else
		{
			tx_minmax[0] = -tx_minmax[1];
			ty_minmax[0] = -ty_minmax[1];
			initStarVal(rand_generator, tx_minmax, txi_start);
			initStarVal(rand_generator, ty_minmax, tyi_start);
		}

		double tzr = tz_relxymax * std::max(abs(txi_start.back()), abs(tyi_start.back()));
		double tz_minmax[2];
		tz_minmax[0] = -tzr;
		tz_minmax[1] = tzr;
		initStarVal(rand_generator, tz_minmax, tzi_start);
		initStarVal(rand_generator, roll_minmax, rolli_start);
		initStarVal(rand_generator, pitch_minmax, pitchi_start);
		initStarVal(rand_generator, yaw_minmax, yawi_start);

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

		tx.push_back(txi_start);
		ty.push_back(tyi_start);
		tz.push_back(tzi_start);
		roll.push_back(rolli_start);
		pitch.push_back(pitchi_start);
		yaw.push_back(yawi_start);

		for (int i = 1; i < nrCams; i++)
		{
			vector<double> txi, tyi, tzi, rolli, pitchi, yawi;
			size_t wrongsignCnt = 0;
			bool cngTxy = false;
			while (txi.empty() || ((wrongsignCnt < 100) && cngTxy))
			{
				if (tx_equRanges)
					txi = txi_start;
				else
					initRangeVals(rand_generator, tx_minmax, tx_minmax_change, txi_start, txi);
				if (ty_equRanges)
					tyi = tyi_start;
				else
					initRangeVals(rand_generator, ty_minmax, ty_minmax_change, tyi_start, tyi);

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
					cngTxy = true;
					txi.clear();
					tyi.clear();
				}
				else
				{
					cngTxy = false;
				}
				wrongsignCnt++;
			}
			if (cngTxy)
				break;
			if (tz_equRanges)
				tzi = tzi_start;
			else
				initRangeVals(rand_generator, tz_minmax, tz_minmax_change, tzi_start, tzi);
			if (roll_equRanges)
				rolli = rolli_start;
			else
				initRangeVals(rand_generator, roll_minmax, roll_minmax_change, rolli_start, rolli);
			if (pitch_equRanges)
				pitchi = pitchi_start;
			else
				initRangeVals(rand_generator, pitch_minmax, pitch_minmax_change, pitchi_start, pitchi);
			if (yaw_equRanges)
				yawi = yawi_start;
			else
				initRangeVals(rand_generator, yaw_minmax, yaw_minmax_change, yawi_start, yawi);

			tx.push_back(txi);
			ty.push_back(tyi);
			tz.push_back(tzi);
			roll.push_back(rolli);
			pitch.push_back(pitchi);
			yaw.push_back(yawi);
		}

		if (tx.size() < nrCams)
			continue;

		double approxImgOverlap = getRandDoubleVal(rand_generator, 0.2, 1.0);//0.1;
		cv::Size imgSize = cv::Size(1280, 720);

		GenStereoPars newStereoPars(tx, ty, tz, roll, pitch, yaw, approxImgOverlap, imgSize);
		newStereoPars.optimizeRtf(verbose);

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
	}
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

