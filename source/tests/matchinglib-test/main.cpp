
#if 1

#include <gmock/gmock.h>

int main(int argc, char* argv[])
{
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}

#else

// ideal case
#include "matchinglib.h"
// ---------------------

#include "..\include\eval_start.h"
#include "..\include\argvparser.h"

//#include "..\include\match_statOptFlow.h"
#include "..\include\io_data.h"
#include "..\include\test_GMbSOF.h"
#include "..\include\test_hirachClustIdx.h"
#include "..\include\test_LSHidx.h"
#include "..\include\test_LinearMatch.h"
#include "..\include\test_RandomKDTree.h"
#include "..\include\test_hirarchK-meansTree.h"
#include "..\include\test_CascadeHashing.h"
#include "..\include\test_libvisio2.h"
#include "..\include\test_GeometryAwareMatch.h"
#include "..\include\test_knnHirarchClustIdxVFC.h"

//#include <opencv2/imgproc/imgproc.hpp>

//#include "PfeImgFileIO.h"

using namespace std;
using namespace cv;
using namespace CommandLineProcessing;

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
	string img_path, gt_path, gt_type_str, l_img_pref, gt_pref, r_img_pref, f_detect, d_extr, matcher;
	string res_path, inl_rat_str, show_str, show_ref_str, img_res_path, img_ref_path, idx1_str, idx2_str, valid_th_str;
	int gt_type, show, show_ref, idx1, idx2;
	bool ratiot, refine, s_key_size;
	double inl_rat = -1.0, valid_th = 0.3;

	t_meas = cmd.foundOption("t_meas");
	t_meas_inlr = cmd.foundOption("t_meas_inlr");
	inl_rat_test = cmd.foundOption("inl_rat_test");
	inl_rat_test_all = cmd.foundOption("inl_rat_test_all");
	th_eval = cmd.foundOption("th_eval");
	qual_meas = cmd.foundOption("qual_meas");
	rad_eval = cmd.foundOption("rad_eval");
	initM_eval = cmd.foundOption("initM_eval");
	CDrat_eval = cmd.foundOption("CDrat_eval");

	ratiot = cmd.foundOption("ratiot");
	refine = cmd.foundOption("refine");
	s_key_size = cmd.foundOption("s_key_size");

	img_path = cmd.optionValue("img_path");
	gt_path = cmd.optionValue("gt_path");
	l_img_pref = cmd.optionValue("l_img_pref");
	gt_pref = cmd.optionValue("gt_pref");
	if(cmd.foundOption("r_img_pref"))
		r_img_pref = cmd.optionValue("r_img_pref");
	f_detect = cmd.optionValue("f_detect");
	d_extr = cmd.optionValue("d_extr");
	if(cmd.foundOption("matcher"))
		matcher = cmd.optionValue("matcher");
	if(cmd.foundOption("res_path"))
		res_path = cmd.optionValue("res_path");
	if(cmd.foundOption("img_res_path"))
		img_res_path = cmd.optionValue("img_res_path");
	if(cmd.foundOption("img_ref_path"))
		img_ref_path = cmd.optionValue("img_ref_path");

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

	if(t_meas)
	{
		startTimeMeasurement(img_path, gt_path, gt_type, 
						 l_img_pref, r_img_pref, gt_pref,
						 f_detect, d_extr, matcher,
						 ratiot, res_path, refine, inl_rat, show, show_ref, img_res_path, img_ref_path);
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
								show, show_ref, img_res_path, img_ref_path);
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
}

/** @function main */
int main( int argc, char* argv[])
{
	ArgvParser cmd;
	SetupCommandlineParser(cmd, argc, argv);
	startEvaluation(cmd);

	return 0;
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

#endif