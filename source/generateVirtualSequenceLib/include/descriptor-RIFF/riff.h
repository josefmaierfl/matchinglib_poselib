#ifndef _RIFFH
#define _RIFFH

#pragma once

//#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/ml/ml.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

class RIFFDescriptor
{

public:
	RIFFDescriptor(	){};
	~RIFFDescriptor( ){};

public:
	//void  Keypoints_Detection(cv::Mat& image, std::vector<cv::KeyPoint>& Key_Points);//this function is used to detect the key points in the image
	void  Retinapattern_Bulid();//this function is used establish the retina sampling pattern
	float Orentation_Calculate(cv::Mat& image, cv::Mat& integral, float keypoint_x, float keypoint_y, int k_point);//this function is used to calculate the orientation of the key points
	void  Scale_Update(cv::Mat& image,std::vector<cv::KeyPoint>& Key_Points);//this function is used to update the scale of detected key points
	void  Descriptor_Generation(cv::Mat& image, cv::Mat& descriptors, std::vector<cv::KeyPoint>& Key_Points);//this function is used to generate the retina features
	float Mean_Compute(cv::Mat& image, cv::Mat& integral, float keypoint_x, float keypoint_y, int scale, int rotation, int point);//this function is to compute the mean value
	void  Top_Salientpoints(cv::Mat& detected_descriptors, std::vector<cv::KeyPoint>& detected_keypoinits, cv::Mat& topdescriptors, std::vector<cv::KeyPoint>& topkeypoinits);//this function is used to computer the top salient points 
	void  Descriptor_Match(cv::Mat& image_a, cv::Mat& image_b, cv::Mat& image_c, cv::Mat& image_d, std::vector<cv::KeyPoint>& keypoints_a, std::vector<cv::KeyPoint> keypoints_b, cv::Mat& descriptor_a, cv::Mat& descriptor_b);//this function is used to match two calculated descriptors and draw the line between the correspondences 
   
protected:
	int   patternSizes[64]; 
	std::vector<int> Keypoint_Scale_Index;

	struct PatternPoint
	{
		float x; // x coordinate
		float y; // y coordinate
		float radius; //Radius of the point 
	};

	struct OrientationPair
	{
		uchar i; // index of the first point
		uchar j; // index of the second point
		int weight_dx; // dx/(norm_sq))*4096
		int weight_dy; // dy/(norm_sq))*4096
	};

	std::vector<PatternPoint> patternLookup;
	OrientationPair orientationPairs[45];

private:
};
#endif