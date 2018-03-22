/**********************************************************************************************************
FILE: loadGTMfiles.cpp

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: May 2017

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for loading and showing the GTMs.
**********************************************************************************************************/

#include "loadGTMfiles.h"
#include "io_helper.h"
#include "readGTM.h"

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#define WRITEMATCHESDISK 1
#if WRITEMATCHESDISK
#include <opencv2/imgproc/imgproc.hpp>
#endif

using namespace std;
using namespace cv;

#if WRITEMATCHESDISK
//from CV, but slightly changed:
void drawMatchesCV1(const Mat& img1, const vector<KeyPoint>& keypoints1,
	const Mat& img2, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& matches1to2, Mat& outImg,
	const Scalar& matchColor = Scalar::all(-1), const Scalar& singlePointColor = Scalar::all(-1),
	const vector<char>& matchesMask = vector<char>(), int flags = DrawMatchesFlags::DEFAULT, bool drawvertical = false);
static void _prepareImgAndDrawKeypointsCV1(const Mat& img1, const vector<KeyPoint>& keypoints1,
	const Mat& img2, const vector<KeyPoint>& keypoints2,
	Mat& outImg, Mat& outImg1, Mat& outImg2,
	const Scalar& singlePointColor, int flags, bool drawvertical = false);
static inline void _drawMatchCV1(Mat& outImg, Mat& outImg1, Mat& outImg2,
	const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, int flags, bool drawvertical = false);
static inline void _drawKeypointCV1(Mat& img, const KeyPoint& p, const Scalar& color, int flags);
#endif

//Show a fraction of the matches
int showMatches(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypL, std::vector<cv::KeyPoint> keypR, cv::Mat imgs[2], size_t keepNMatches = 20);

int showGTM(std::string img_path, std::string l_img_pref, std::string r_img_pref,
	std::string gtm_path, std::string gtm_postfix)
{
	int err;
	cv::Mat src[2];
	string fileprefl, fileprefr;
	vector<string> filenamesl, filenamesr, filenamesgtm;

	fileprefl = l_img_pref;
	if (r_img_pref.empty())
		fileprefr = fileprefl;
	else
		fileprefr = r_img_pref;

	//Load corresponding image names
	err = loadImgStereoSequence(img_path, fileprefl, fileprefr, filenamesl, filenamesr);
	if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
	{
		cout << "Could not find flow images! Exiting." << endl;
		exit(0);
	}

	//load GTM names
	err = loadGTMSequence(gtm_path, gtm_postfix, filenamesgtm);
	if (err || filenamesgtm.empty() || (filenamesgtm.size() != filenamesl.size()))
	{
		cout << "Could not find GTM files! Exiting." << endl;
		exit(0);
	}

	// read images and GTMs
	std::vector<bool> leftInlier;
	std::vector<cv::DMatch> matchesGT;
	std::vector<cv::KeyPoint> keypL;
	std::vector<cv::KeyPoint> keypR;
	double inlRatioL, inlRatioR, inlRatioO, positivesGT, negativesGTl, negativesGTr, usedMatchTH;
	for (int k = 0; k < (int)filenamesl.size(); k++)
	{
		cv::Mat flowimg;
		src[0] = cv::imread(filenamesl[k]);
		src[1] = cv::imread(filenamesr[k]);
		if (readGTMatchesDisk(filenamesgtm[k],
			leftInlier,
			matchesGT,
			keypL,
			keypR,
			&inlRatioL,
			&inlRatioR,
			&inlRatioO,
			&positivesGT,
			&negativesGTl,
			&negativesGTr,
			&usedMatchTH))
		{
			cout << "Succesfully read GTM file " << filenamesgtm[k] << endl;
			cout << "Inlier ratio in first/left image: " << inlRatioL << endl;
			cout << "Inlier ratio in second/right image: " << inlRatioR << endl;
			cout << "Mean inlier ratio of both images: " << inlRatioO << endl;
			cout << "Number of true positive matches: " << positivesGT << endl;
			cout << "Number of left negatives (having no corresponding right match): " << negativesGTl << endl;
			cout << "Number of right negatives (having no corresponding left match): " << negativesGTr << endl;
			cout << "Threshold used to generate GTM: " << usedMatchTH << endl << endl;
			showMatches(matchesGT, keypL, keypR, src, 20);
		}
		else
		{
			cout << "Error while reading GTM file " << filenamesgtm[k] << endl << endl;
		}
	}
}

/* Shows the matches and optionally stores the matched keypoints as image to the disk. It can be specified if only true positives (blue),
* all matches (true positives in blue, false positives in red) or all matches with false negatives (true positives in blue, false
* positives in red, false negatives in green) should be shown.
*
* int drawflags				Input  -> The following matches are drawn in different colors for different flags:
*										  0:	Only true positives
*										  1:	True positives and false positives [DEFAULT]
*										  2:	True positives, false positives, and false negatives
* string path					Input  -> If not empty, this string specifies the path where the image with drawn matches should be stored.
*										  If empty [DEFAULT], no image is stored to disk.
* string file					Input  -> If not empty, this string specifies the file name including the file extension.
*										  If empty [DEFAULT], no image is stored to disk.
* bool storeOnly				Input  -> If true, the output image is only stored to disk but not displayed [DEFAULT = false]
*
* Return value:				 0:		  Everything ok
*								-1:		  No refined matches available
*/
int showMatches(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> keypL, std::vector<cv::KeyPoint> keypR, cv::Mat imgs[2], size_t keepNMatches)
{
	//vector<cv::DMatch> matches_tmp;
	//vector<KeyPoint> kps1_tp, kps2_tp, kps1_fp, kps2_fp, kps1_fn, kps2_fn;
	//vector<KeyPoint> kps1_tp_reduced, kps2_tp_reduced, kps1_fp_reduced, kps2_fp_reduced, kps1_fn_reduced, kps2_fn_reduced;
	vector<char> matchesMask(matches.size(), false);;
	Mat img_correctMatches;
	//size_t keepNMatches = 20; //The overall number of matches that should be displayed
	float keepXthMatch;
	float oldremainder, newremainder = 0;
	//int j = 0;

		//matches_tmp = matches;

	//Reduce number of displayed matches
			//keepNMatches_tp = keepNMatches;



	//Get only true positives without true negatives
	/*j = 0;
	for (unsigned int i = 0; i < matches_tmp.size(); i++)
	{
		int idx = matches_tmp[i].queryIdx;
		kps1_tp.push_back(keypL[idx]);
		matches_tp.push_back(matches_tmp[i]);
		matches_tp.back().queryIdx = j;
		kps2_tp.push_back(keypR[matches_tp.back().trainIdx]);
		matches_tp.back().trainIdx = j;
		j++;
	}*/
	//Reduce number of displayed matches
	keepXthMatch = 1.0f;
	if (matches.size() > keepNMatches)
		keepXthMatch = (float)matches.size() / (float)keepNMatches;
	//j = 0;
	oldremainder = 0;
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		newremainder = fmod((float)i, keepXthMatch);
		//if((i % (int)keepXthMatch) == 0)
		if (oldremainder >= newremainder)
		{
			//kps1_tp.push_back(keypL[i]);
			//matches_tmp.push_back(matches[i]);
			matchesMask[i] = true;
			//matches_tmp.back().queryIdx = j;
			//kps2_tp.push_back(keypR[i]);
			//matches_tmp.back().trainIdx = j;
			//j++;
		}
		oldremainder = newremainder;
	}
	//Draw true positive matches
		//drawMatchesCV(imgs[0], kps1_tp, imgs[1], kps2_tp, matches_tmp, img_correctMatches, CV_RGB(0, 0, 255));
#if WRITEMATCHESDISK
	drawMatchesCV1(imgs[0], keypL, imgs[1], keypR, matches, img_correctMatches, Scalar::all(-1), Scalar::all(-1), matchesMask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS, true);
#else
	drawMatches(imgs[0], keypL, imgs[1], keypR, matches, img_correctMatches, Scalar::all(-1), Scalar::all(-1), matchesMask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
#endif

#if WRITEMATCHESDISK
		string writepath = "C:/work/tmp/GTM_images/gtm_img.jpg";
		cv::imwrite(writepath, img_correctMatches);
#endif

		//Show result
		cvNamedWindow("Ground Truth Matches");
		imshow("Ground Truth Matches", img_correctMatches);
		waitKey(0);
		cv::destroyWindow("Ground Truth Matches");

	return 0;
}

#if WRITEMATCHESDISK
//from CV and additional option to align images vertical:
void drawMatchesCV1(const Mat& img1, const vector<KeyPoint>& keypoints1,
	const Mat& img2, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& matches1to2, Mat& outImg,
	const Scalar& matchColor, const Scalar& singlePointColor,
	const vector<char>& matchesMask, int flags, bool drawvertical)
{
	if (!matchesMask.empty() && matchesMask.size() != matches1to2.size())
		CV_Error(CV_StsBadSize, "matchesMask must have the same size as matches1to2");

	Mat outImg1, outImg2;
	_prepareImgAndDrawKeypointsCV1(img1, keypoints1, img2, keypoints2,
		outImg, outImg1, outImg2, singlePointColor, flags, drawvertical);

	// draw matches
	for (size_t m = 0; m < matches1to2.size(); m++)
	{
		if (matchesMask.empty() || matchesMask[m])
		{
			int i1 = matches1to2[m].queryIdx;
			int i2 = matches1to2[m].trainIdx;
			CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints1.size()));
			CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints2.size()));

			const KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
			_drawMatchCV1(outImg, outImg1, outImg2, kp1, kp2, matchColor, flags, drawvertical);
		}
	}
}

//from CV and additional option to align images vertical:
static void _prepareImgAndDrawKeypointsCV1(const Mat& img1, const vector<KeyPoint>& keypoints1,
	const Mat& img2, const vector<KeyPoint>& keypoints2,
	Mat& outImg, Mat& outImg1, Mat& outImg2,
	const Scalar& singlePointColor, int flags, bool drawvertical)
{
	Size size;
	if(drawvertical)
		size = Size(MAX(img1.cols, img2.cols), img1.rows + img2.rows);
	else
		size = Size(img1.cols + img2.cols, MAX(img1.rows, img2.rows));

	if (flags & DrawMatchesFlags::DRAW_OVER_OUTIMG)
	{
		if (size.width > outImg.cols || size.height > outImg.rows)
			CV_Error(CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together");
		outImg1 = outImg(Rect(0, 0, img1.cols, img1.rows));
		if (drawvertical)
			outImg2 = outImg(Rect(0, img1.rows, img2.cols, img2.rows));
		else
			outImg2 = outImg(Rect(img1.cols, 0, img2.cols, img2.rows));
	}
	else
	{
		outImg.create(size, CV_MAKETYPE(img1.depth(), 3));
		outImg = Scalar::all(0);
		outImg1 = outImg(Rect(0, 0, img1.cols, img1.rows));
		if (drawvertical)
			outImg2 = outImg(Rect(0, img1.rows, img2.cols, img2.rows));
		else
			outImg2 = outImg(Rect(img1.cols, 0, img2.cols, img2.rows));

		if (img1.type() == CV_8U)
			cvtColor(img1, outImg1, CV_GRAY2BGR);
		else
			img1.copyTo(outImg1);

		if (img2.type() == CV_8U)
			cvtColor(img2, outImg2, CV_GRAY2BGR);
		else
			img2.copyTo(outImg2);
	}

	// draw keypoints
	if (!(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS))
	{
		Mat _outImg1 = outImg(Rect(0, 0, img1.cols, img1.rows));
		drawKeypoints(_outImg1, keypoints1, _outImg1, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG);

		Mat _outImg2 = outImg(Rect(img1.cols, 0, img2.cols, img2.rows));
		drawKeypoints(_outImg2, keypoints2, _outImg2, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG);
	}
}
//from CV with changed line width and additional option to align images vertical:
static inline void _drawMatchCV1(Mat& outImg, Mat& outImg1, Mat& outImg2,
	const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, int flags, bool drawvertical)
{
	const int draw_shift_bits = 4;
	const int draw_multiplier = 1 << draw_shift_bits;
	RNG& rng = theRNG();
	bool isRandMatchColor = matchColor == Scalar::all(-1);
	Scalar color = isRandMatchColor ? Scalar(rng(256), rng(256), rng(256)) : matchColor;

	_drawKeypointCV1(outImg1, kp1, color, flags);
	_drawKeypointCV1(outImg2, kp2, color, flags);

	Point2f pt1 = kp1.pt,
		pt2 = kp2.pt,
		dpt2;
	if (drawvertical)
		dpt2 = Point2f(pt2.x, min(pt2.y + (float)outImg1.rows, (float)(outImg.rows - 1)));
	else
		dpt2 = Point2f(min(pt2.x + (float)outImg1.cols, (float)(outImg.cols - 1)), pt2.y);

	line(outImg,
		Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
		Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
		color, 2, CV_AA, draw_shift_bits);
}
//from CV:
static inline void _drawKeypointCV1(Mat& img, const KeyPoint& p, const Scalar& color, int flags)
{
	CV_Assert(!img.empty());
	const int draw_shift_bits = 4;
	const int draw_multiplier = 1 << draw_shift_bits;
	Point center(cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier));

	if (flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS)
	{
		int radius = cvRound(p.size / 2 * draw_multiplier); // KeyPoint::size is a diameter

															// draw the circles around keypoints with the keypoints size
		circle(img, center, radius, color, 1, CV_AA, draw_shift_bits);

		// draw orientation of the keypoint, if it is applicable
		if (p.angle != -1)
		{
			float srcAngleRad = p.angle*(float)CV_PI / 180.f;
			Point orient(cvRound(cos(srcAngleRad)*radius),
				cvRound(sin(srcAngleRad)*radius)
			);
			line(img, center, center + orient, color, 1, CV_AA, draw_shift_bits);
		}
#if 0
		else
		{
			// draw center with R=1
			int radius = 1 * draw_multiplier;
			circle(img, center, radius, color, 1, CV_AA, draw_shift_bits);
		}
#endif
	}
	else
	{
		// draw center with R=3
		int radius = 3 * draw_multiplier;
		circle(img, center, radius, color, 1, CV_AA, draw_shift_bits);
	}
}
#endif
