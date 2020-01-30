/*
Discriminate Salient Point Retina Descriptor
This is based on Opencv library 
LIACS Media Lab, Leiden University,
Song Wu (s.wu@liace.leidenuniv.nl)
Michael S. Lew (mlew@liacs.nl)
*/

#include "descriptor-RIFF/riff.h"
//#include <opencv2/legacy/legacy.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/contrib/contrib.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

static const int     NB_PATTERN_POINTS = 43;
static const int     NB_SCALES         = 64;
static const int     NB_ORIENTATION    = 360;
static const int     NB_OCTAVES        = 4;
static const float   DESC_SIGMA        = 0.5f;
static const float   PATTERN_SCALE     = 22.0f;
static const double  LOG2_VALUE        = 0.693147180559945;
static const int     NEAREST_NUMBER    = 50;

//x correspondence to image cols
//y correspondence to image rows
/*
this function is used to detect the key points in the image
cv::Mat& image: input image
std::vector<cv::KeyPoint>& Key_Points: vector to storage the detected key points
use SURF detector to locate key points in image
*/
//void RIFFDescriptor::Keypoints_Detection( cv::Mat& image, std::vector<cv::KeyPoint>& Key_Points )
//{
//	cv::FeatureDetector *detector = new cv::SURF(200,4,3,0,1);
//	detector->detect(image,Key_Points);
//	delete detector;
//}

/*this function is used establish the retina sampling pattern*/
void RIFFDescriptor::Retinapattern_Bulid()
{
	patternLookup.resize(NB_SCALES*NB_ORIENTATION*NB_PATTERN_POINTS);//the size is 64*300*43
	double scaleStep = pow( 2.0, (double)(NB_OCTAVES)/NB_SCALES ); 
	double scalingFactor, alpha, beta, theta = 0;
	const int    n[8]      = {6,6,6,6,6,6,6,1}; // number of points on each concentric circle
	const double outer_R   = 2.0/3.0; // radius of outer circle
	const double inter_R   = 2.0/24.0; // radius of the inter circle
	const double unitSpace = (outer_R-inter_R)/21.0; // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)	
	const double Circle_Radius[8] = {outer_R,              outer_R-6*unitSpace, 
		                       outer_R-11*unitSpace, outer_R-15*unitSpace, 
		                       outer_R-18*unitSpace, outer_R-20*unitSpace, 
		                       inter_R, 0.0 };// radius of the concentric circles (from outer to inner)
	const double Point_Radius[8]  = {Circle_Radius[0]/2.0, Circle_Radius[1]/2.0, 
		                       Circle_Radius[2]/2.0, Circle_Radius[3]/2.0,
		                       Circle_Radius[4]/2.0, Circle_Radius[5]/2.0, 
		                       Circle_Radius[6]/2.0, Circle_Radius[7]/2.0};//radius of the pattern points
	//establish the lookup table
	for( int scaleIdx=0; scaleIdx < NB_SCALES; ++scaleIdx ) 
	{
		patternSizes[scaleIdx] = 0; // proper initialization
		scalingFactor = pow(scaleStep,scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

		for( int orientationIdx = 0; orientationIdx < NB_ORIENTATION; ++orientationIdx ) 
		{
			theta = double(orientationIdx)* 2*CV_PI/double(NB_ORIENTATION); // orientation of the pattern
			int pointIdx = 0;

			PatternPoint* patternLookupPtr = &patternLookup[0];
			for( int i = 0; i < 8; ++i )
			{
				for( int k = 0 ; k < n[i]; ++k ) 
				{
					beta = CV_PI/n[i] * (i%2); // orientation offset so that groups of points on each circles are staggered
					alpha = double(k)* 2*CV_PI/double(n[i])+beta+theta;

					// add the point to the lookup table
					PatternPoint &point = patternLookupPtr[ scaleIdx*NB_ORIENTATION*NB_PATTERN_POINTS+orientationIdx*NB_PATTERN_POINTS+pointIdx ];
					point.x = static_cast<float>(Circle_Radius[i] * cos(alpha) * scalingFactor * PATTERN_SCALE);
					point.y = static_cast<float>(Circle_Radius[i] * sin(alpha) * scalingFactor * PATTERN_SCALE);
					point.radius = static_cast<float>(Point_Radius[i] * scalingFactor * PATTERN_SCALE);

					// adapt the sizeList if necessary
					const int sizeMax = static_cast<int>(ceil((Circle_Radius[i]+Point_Radius[i])*scalingFactor*PATTERN_SCALE)) + 1;
					if( patternSizes[scaleIdx] < sizeMax )
						patternSizes[scaleIdx] = sizeMax;
					++pointIdx;
				}
			}
		}
	}
}

/*
this function is used to calculate the orientation of the key points
cv::Mat& image: the input image
cv::Mat& integral: the intergral_image of input image
float keypoint_x, keypoint_y: the coordinates of key point
int k_point: the k-th point in the detected key points
*/
float RIFFDescriptor::Orentation_Calculate(cv::Mat& image, cv::Mat& integral, float keypoint_x, float keypoint_y, int k_point)
{
	float angle         = 0.f;
	float orientation_x = 0.f;
	float orientation_y = 0.f;
	float Points_Mean[NB_PATTERN_POINTS];
	// the list of orientation pairs which is established by FREAK
	orientationPairs[0].i=0;   orientationPairs[0].j=3;   orientationPairs[1].i=1;   orientationPairs[1].j=4;   orientationPairs[2].i=2;   orientationPairs[2].j=5;
	orientationPairs[3].i=0;   orientationPairs[3].j=2;   orientationPairs[4].i=1;   orientationPairs[4].j=3;   orientationPairs[5].i=2;   orientationPairs[5].j=4;
	orientationPairs[6].i=3;   orientationPairs[6].j=5;   orientationPairs[7].i=4;   orientationPairs[7].j=0;   orientationPairs[8].i=5;   orientationPairs[8].j=1;

	orientationPairs[9].i=6;   orientationPairs[9].j=9;   orientationPairs[10].i=7;  orientationPairs[10].j=10; orientationPairs[11].i=8;  orientationPairs[11].j=11;
	orientationPairs[12].i=6;  orientationPairs[12].j=8;  orientationPairs[13].i=7;  orientationPairs[13].j=9;  orientationPairs[14].i=8;  orientationPairs[14].j=10;
	orientationPairs[15].i=9;  orientationPairs[15].j=11; orientationPairs[16].i=10; orientationPairs[16].j=6;  orientationPairs[17].i=11; orientationPairs[17].j=7;

	orientationPairs[18].i=12; orientationPairs[18].j=15; orientationPairs[19].i=13; orientationPairs[19].j=16; orientationPairs[20].i=14; orientationPairs[20].j=17;
	orientationPairs[21].i=12; orientationPairs[21].j=14; orientationPairs[22].i=13; orientationPairs[22].j=15; orientationPairs[23].i=14; orientationPairs[23].j=16;
	orientationPairs[24].i=15; orientationPairs[24].j=17; orientationPairs[25].i=16; orientationPairs[25].j=12; orientationPairs[26].i=17; orientationPairs[26].j=13;

	orientationPairs[27].i=18; orientationPairs[27].j=21; orientationPairs[28].i=19; orientationPairs[28].j=22; orientationPairs[29].i=20; orientationPairs[29].j=23;
	orientationPairs[30].i=18; orientationPairs[30].j=20; orientationPairs[31].i=19; orientationPairs[31].j=21; orientationPairs[32].i=20; orientationPairs[32].j=22;
	orientationPairs[33].i=21; orientationPairs[33].j=23; orientationPairs[34].i=22; orientationPairs[34].j=18; orientationPairs[35].i=23; orientationPairs[35].j=19;

	orientationPairs[36].i=24; orientationPairs[36].j=27; orientationPairs[37].i=25; orientationPairs[37].j=28; orientationPairs[38].i=26; orientationPairs[38].j=29;
	orientationPairs[39].i=30; orientationPairs[39].j=33; orientationPairs[40].i=31; orientationPairs[40].j=34; orientationPairs[41].i=32; orientationPairs[41].j=35;
	orientationPairs[42].i=36; orientationPairs[42].j=39; orientationPairs[43].i=37; orientationPairs[43].j=40; orientationPairs[44].i=38; orientationPairs[44].j=41;

	for( int m = 45; m--; )
	{
		const float dx = patternLookup[orientationPairs[m].i].x-patternLookup[orientationPairs[m].j].x;
		const float dy = patternLookup[orientationPairs[m].i].y-patternLookup[orientationPairs[m].j].y;
		const float norm_sq = (dx*dx+dy*dy);
		orientationPairs[m].weight_dx = int((dx/(norm_sq))*4096.0+0.5);
		orientationPairs[m].weight_dy = int((dy/(norm_sq))*4096.0+0.5);
	}

	for( int i = NB_PATTERN_POINTS; i--; )
	{
		Points_Mean[i] = Mean_Compute(image, integral, keypoint_x,keypoint_y, Keypoint_Scale_Index[k_point], 0, i);
	}

	for( int m = 45; m--; ) 
	{
		//iterate through the orientation pairs
		int delta = (int)(Points_Mean[ orientationPairs[m].i ]-Points_Mean[ orientationPairs[m].j ]);
		orientation_x += delta*(orientationPairs[m].weight_dx)/2048;
		orientation_y += delta*(orientationPairs[m].weight_dy)/2048;
	}

	return angle = static_cast<float>(atan2((float)orientation_y,(float)orientation_x)*(180.0/CV_PI));
}

/*
This function update the Scale of the keypoints and remove the keypoints near the border
cv::Mat image: input image
std::vector<cv::KeyPoint>& Key_Points: vector to storage the detected key points
*/
void RIFFDescriptor::Scale_Update(cv::Mat& image, std::vector<cv::KeyPoint>& Key_Points)
{
	int SMALLEST_KP_SIZE=7;
	Keypoint_Scale_Index.resize(Key_Points.size());
	const std::vector<int>::iterator Scale_iterator = Keypoint_Scale_Index.begin();
	const std::vector<cv::KeyPoint>::iterator KeyPoint_iterator = Key_Points.begin(); 
	const float sizeCst = static_cast<float>(NB_SCALES/(LOG2_VALUE* NB_OCTAVES));

	for( size_t k = Key_Points.size(); k--; )
	{
		Keypoint_Scale_Index[k] = max( (int)(log(Key_Points[k].size/SMALLEST_KP_SIZE)*sizeCst+0.5) ,0);
		if( Keypoint_Scale_Index[k] >= NB_SCALES )
			Keypoint_Scale_Index[k]  = NB_SCALES-1;

		if( Key_Points[k].pt.x <= patternSizes[Keypoint_Scale_Index[k]] || 
			Key_Points[k].pt.y <= patternSizes[Keypoint_Scale_Index[k]] ||
			Key_Points[k].pt.x >= image.cols-patternSizes[Keypoint_Scale_Index[k]] || 
			Key_Points[k].pt.y >= image.rows-patternSizes[Keypoint_Scale_Index[k]] )
		{
			Key_Points.erase(KeyPoint_iterator+k);
			Keypoint_Scale_Index.erase(Scale_iterator+k);
		}
	}
}

/*
this function is to compute the mean value
cv::Mat& image: input image
cv::Mat& integral: the intergral_image of input image
float keypoint_x, keypoint_y: the coordinates of key point
int scale: the update scale of key point
int rotation: the rotation of key point
int point: the point-th point in the retina sampling pattern
*/
float RIFFDescriptor::Mean_Compute(cv::Mat& image, cv::Mat& integral, float keypoint_x, float keypoint_y, int scale, int rotation, int point)
{
	PatternPoint& Point = patternLookup[scale*NB_ORIENTATION*NB_PATTERN_POINTS + rotation*NB_PATTERN_POINTS + point];
	float xf = Point.x+keypoint_x;
	float yf = Point.y+keypoint_y;
	int   x = int(xf);
	int   y = int(yf);
	int   &imagecols = image.cols;

	int ret_val;
	const float Radius = Point.radius;
	const float area = 4.0f * Radius * Radius;
  	
	if( Radius < 0.5 )
	{	
		// interpolation multipliers:
		const int r_x = static_cast<int>((xf-x)*1024);
		const int r_y = static_cast<int>((yf-y)*1024);
		const int r_x_1 = (1024-r_x);
		const int r_y_1 = (1024-r_y);
		const uchar* ptr = &image.at<uchar>(y, x);
		size_t step = image.step;
		//interpolate:
		ret_val = r_x_1 * r_y_1 * ptr[0] + r_x * r_y_1 * ptr[1] +r_x * r_y * ptr[step] + r_x_1 * r_y * ptr[step+1];		 
		return (ret_val + 512) / 1024;
	}

	// this is the standard case :
	// scaling:
	const int scaling = (int)(4194304.0 / area);
	const int scaling2 = int(float(scaling) * area / 1024.0);

	// the integral image is one pixel larger:
	const int integralcols = imagecols + 1;

	// calculate borders
	const float x_1 = xf - Radius;
	const float x1  = xf + Radius;
	const float y_1 = yf - Radius;
	const float y1  = yf + Radius;

	const int x_left   = int(x_1 + 0.5);
	const int y_top    = int(y_1 + 0.5);
	const int x_right  = int(x1  + 0.5);
	const int y_bottom = int(y1  + 0.5);

	// overlap area - multiplication factors:
	const float r_x_1   = float(x_left) - x_1 + 0.5f;
	const float r_y_1   = float(y_top)  - y_1 + 0.5f;
	const float r_x1    = x1 - float(x_right) + 0.5f;
	const float r_y1    = y1 - float(y_bottom)+ 0.5f;
	const int   dx      = x_right  - x_left - 1;
	const int   dy      = y_bottom - y_top  - 1;
	const int   A       = (int)((r_x_1 * r_y_1)* scaling);
	const int   B       = (int)((r_x1  * r_y_1)* scaling);
	const int   C       = (int)((r_x1  * r_y1) * scaling);
	const int   D       = (int)((r_x_1 * r_y1) * scaling);
	const int   r_x_1_i = (int)(r_x_1  * scaling);
	const int   r_y_1_i = (int)(r_y_1  * scaling);
	const int   r_x1_i  = (int)(r_x1   * scaling);
	const int   r_y1_i  = (int)(r_y1   * scaling);

	if (dx + dy > 2)
	{
		// now the calculation:
		uchar* ptr = image.data + x_left + imagecols * y_top;
		// first the corners:
		ret_val    = A * int(*ptr);
		ptr += dx + 1;
		ret_val += B * int(*ptr);
		ptr += dy * imagecols + 1;
		ret_val += C * int(*ptr);
		ptr -= dx + 1;
		ret_val += D * int(*ptr);

		// next the edges:
		int* ptr_integral = (int*) integral.data + x_left + integralcols * y_top + 1;
		// find a simple path through the different surface corners
		const int tmp1 = (*ptr_integral);
		ptr_integral += dx;
		const int tmp2 = (*ptr_integral);
		ptr_integral += integralcols;
		const int tmp3 = (*ptr_integral);
		ptr_integral++;
		const int tmp4 = (*ptr_integral);
		ptr_integral += dy * integralcols;
		const int tmp5 = (*ptr_integral);
		ptr_integral--;
		const int tmp6 = (*ptr_integral);
		ptr_integral += integralcols;
		const int tmp7 = (*ptr_integral);
		ptr_integral -= dx;
		const int tmp8 = (*ptr_integral);
		ptr_integral -= integralcols;
		const int tmp9 = (*ptr_integral);
		ptr_integral--;
		const int tmp10 = (*ptr_integral);
		ptr_integral -= dy * integralcols;
		const int tmp11 = (*ptr_integral);
		ptr_integral++;
		const int tmp12 = (*ptr_integral);

		// assign the weighted surface integrals:
		const int upper  = (tmp3 - tmp2  + tmp1  - tmp12) * r_y_1_i;
		const int middle = (tmp6 - tmp3  + tmp12 - tmp9 ) * scaling;
		const int left   = (tmp9 - tmp12 + tmp11 - tmp10) * r_x_1_i;
		const int right  = (tmp5 - tmp4  + tmp3  - tmp6 ) * r_x1_i;
		const int bottom = (tmp7 - tmp6  + tmp9  - tmp8 ) * r_y1_i;

		return (ret_val + upper + middle + left + right + bottom + scaling2 / 2) / scaling2;
	}

	// now the calculation:
	uchar* ptr = image.data + x_left + imagecols * y_top;
	// first row:
	ret_val = A * int(*ptr);
	ptr++;
	const uchar* end1 = ptr + dx;
	for (; ptr < end1; ptr++)
	{
		ret_val += r_y_1_i * int(*ptr);
	}
	ret_val += B * int(*ptr);
	// middle ones:
	ptr += imagecols - dx - 1;
	uchar* end_j = ptr + dy * imagecols;
	for (; ptr < end_j; ptr += imagecols - dx - 1)
	{
		ret_val += r_x_1_i * int(*ptr);
		ptr++;
		const uchar* end2 = ptr + dx;
		for (; ptr < end2; ptr++)
		{
			ret_val += int(*ptr) * scaling;
		}
		ret_val += r_x1_i * int(*ptr);
	}
	// last row:
	ret_val += D * int(*ptr);
	ptr++;
	const uchar* end3 = ptr + dx;
	for (; ptr < end3; ptr++)
	{
		ret_val += r_y1_i * int(*ptr);
	}
	ret_val += C * int(*ptr);

	return (ret_val + scaling2 / 2) / scaling2;
}

/*
this function obtain the top keypoints and filter the unstable keypoints 
this is based on the discriminate score of salient point features
cv::Mat& detected_descriptors: the already generated descriptors
std::vector<cv::KeyPoint>& detected_keypoinits: the already detected key points
cv::Mat& topdescriptors: used for the top salient points features
std::vector<cv::KeyPoint>& topkeypoinits: used for the top salient points
*/
void RIFFDescriptor::Top_Salientpoints(cv::Mat& detected_descriptors, std::vector<cv::KeyPoint>& detected_keypoinits, 
									   cv::Mat& topdescriptors,       std::vector<cv::KeyPoint>& topkeypoinits)
{
  	cv::Mat results_b;
  	cv::Mat distans_b;
  	cv::Mat index_descriptors( detected_descriptors );
 
 	cv::flann::Index flannIndex( index_descriptors,  cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN );
  	results_b  = cv::Mat( detected_descriptors.rows, NEAREST_NUMBER, CV_32SC1 );
  	distans_b  = cv::Mat( detected_descriptors.rows, NEAREST_NUMBER, CV_32FC1 );
  	flannIndex.knnSearch( detected_descriptors, results_b, distans_b, NEAREST_NUMBER, cv::flann::SearchParams( ) );
  
 	std::vector<float>sum_distance( detected_descriptors.rows );	
  	float max_value=FLT_MIN;
  	float min_value=FLT_MAX;
  	for (int i=0; i<detected_descriptors.rows; ++i )
  	{
  		sum_distance[i]=distans_b.at<float>(i,1)+distans_b.at<float>(i,2);
  		float max_value1=sum_distance[i];
  		float min_value1=sum_distance[i];
 		if(min_value1 < min_value ) min_value = min_value1;
 		if(max_value1 > max_value ) max_value = max_value1;
 	}
 
 	std::vector<float>score( detected_descriptors.rows );
 	float max_dist=FLT_MIN;
 	float min_dist=FLT_MAX;
 	for (int i=0; i<detected_descriptors.rows; ++i )
 	{
		score[i]=exp(-6*(sum_distance[i]-min_value)/(float)(max_value-min_value));	
		float max_value2=score[i];
		float min_value2=score[i];
		if(min_value2 < min_dist ) min_dist = min_value2;
		if(max_value2 > max_dist ) max_dist = max_value2;
	}

	int top_size=0;
	for (int i=0; i<detected_descriptors.rows; ++i )
	{
		if (score[i]<0.8*max_dist) top_size++;		
	}

	topdescriptors=cv::Mat(top_size,detected_descriptors.cols,detected_descriptors.type());
 
 	float* data_pointer = detected_descriptors.ptr<float>(0);
	float* top_pointer  = topdescriptors.ptr<float>(0);
	int k=0;
	for (int i=0; i<detected_descriptors.rows; ++i )
	{
		data_pointer = detected_descriptors.ptr<float>(i);		
		if (score[i]<0.8*max_dist)
		{			
			topkeypoinits.push_back(detected_keypoinits.at(i));

			top_pointer=topdescriptors.ptr<float>(k);
			for (int j=0; j<detected_descriptors.cols; ++j )
			{
				top_pointer[j]=data_pointer[j];
			}
			k++;
		} 
	}
}

/*
this function is used to generate the retina features
cv::Mat& image: the input image
cv::Mat& descriptors: the final feature
std::vector<cv::KeyPoint>& Key_Points: vector to storage the detected key points
*/
void RIFFDescriptor::Descriptor_Generation( cv::Mat& image, cv::Mat& descriptors, std::vector<cv::KeyPoint>& Key_Points )
{		
	Retinapattern_Bulid();
	Scale_Update(image,Key_Points);
	
	cv::GaussianBlur( image, image, cv::Size(), DESC_SIGMA, DESC_SIGMA );

	cv::Mat Integral_image;
	cv::integral(image, Integral_image);

	float Point_Value[NB_PATTERN_POINTS];
	int theta_index = 0;

	//Calculate the orientation of the keypoints
	for( size_t k = Key_Points.size(); k--; )
	{
		Key_Points[k].angle=Orentation_Calculate(image, Integral_image, Key_Points[k].pt.x, Key_Points[k].pt.y, k);
	}
 
	descriptors  = cv::Mat::zeros((int)Key_Points.size(), 72, CV_32F);

	for( size_t k = Key_Points.size(); k--; )
	{
		std::vector<float>tmp_value;
		tmp_value.assign(descriptors.cols,0.f);
		float* dst = descriptors.ptr<float>(k);
		theta_index = int(NB_ORIENTATION*Key_Points[k].angle*(1/360.0)+0.5);
		if( theta_index < 0 )
			theta_index += NB_ORIENTATION;

		if( theta_index >= NB_ORIENTATION )
			theta_index -= NB_ORIENTATION;			

		for( int i = NB_PATTERN_POINTS; i--; )
		{
			Point_Value[i] = Mean_Compute(image, Integral_image, Key_Points[k].pt.x, Key_Points[k].pt.y, Keypoint_Scale_Index[k], theta_index, i);			
		}

		tmp_value[0]  = Point_Value[0] -Point_Value[12];  tmp_value[1]  = Point_Value[1] -Point_Value[13];  tmp_value[2]  = Point_Value[2] -Point_Value[14];
		tmp_value[3]  = Point_Value[3] -Point_Value[15];  tmp_value[4]  = Point_Value[4] -Point_Value[16];  tmp_value[5]  = Point_Value[5] -Point_Value[17];
		tmp_value[6]  = Point_Value[0] -Point_Value[24];  tmp_value[7]  = Point_Value[1] -Point_Value[25];  tmp_value[8]  = Point_Value[2] -Point_Value[26];
		tmp_value[9]  = Point_Value[3] -Point_Value[27];  tmp_value[10] = Point_Value[4] -Point_Value[28];  tmp_value[11] = Point_Value[5] -Point_Value[29];
		tmp_value[12] = Point_Value[0] -Point_Value[36];  tmp_value[13] = Point_Value[1] -Point_Value[37];  tmp_value[14] = Point_Value[2] -Point_Value[38];
		tmp_value[15] = Point_Value[3] -Point_Value[39];  tmp_value[16] = Point_Value[4] -Point_Value[40];  tmp_value[17] = Point_Value[5] -Point_Value[41];
		tmp_value[18] = Point_Value[6] -Point_Value[18];  tmp_value[19] = Point_Value[7] -Point_Value[19];  tmp_value[20] = Point_Value[8] -Point_Value[20];
		tmp_value[21] = Point_Value[9] -Point_Value[21];  tmp_value[22] = Point_Value[10]-Point_Value[22];  tmp_value[23] = Point_Value[11]-Point_Value[23];
		tmp_value[24] = Point_Value[6] -Point_Value[30];  tmp_value[25] = Point_Value[7] -Point_Value[31];  tmp_value[26] = Point_Value[8] -Point_Value[32];
		tmp_value[27] = Point_Value[9] -Point_Value[33];  tmp_value[28] = Point_Value[10]-Point_Value[34];  tmp_value[29] = Point_Value[11]-Point_Value[35];
		tmp_value[30] = Point_Value[6] -Point_Value[42];  tmp_value[31] = Point_Value[7] -Point_Value[42];  tmp_value[32] = Point_Value[8] -Point_Value[42];
		tmp_value[33] = Point_Value[9] -Point_Value[42];  tmp_value[34] = Point_Value[10]-Point_Value[42];  tmp_value[35] = Point_Value[11]-Point_Value[42];
		tmp_value[36] = Point_Value[12]-Point_Value[24];  tmp_value[37] = Point_Value[13]-Point_Value[25];  tmp_value[38] = Point_Value[14]-Point_Value[26];
		tmp_value[39] = Point_Value[15]-Point_Value[27];  tmp_value[40] = Point_Value[16]-Point_Value[28];  tmp_value[41] = Point_Value[17]-Point_Value[29];
		tmp_value[42] = Point_Value[12]-Point_Value[36];  tmp_value[43] = Point_Value[13]-Point_Value[37];  tmp_value[44] = Point_Value[14]-Point_Value[38];
		tmp_value[45] = Point_Value[15]-Point_Value[39];  tmp_value[46] = Point_Value[16]-Point_Value[40];  tmp_value[47] = Point_Value[17]-Point_Value[41];
		tmp_value[48] = Point_Value[18]-Point_Value[30];  tmp_value[49] = Point_Value[19]-Point_Value[31];  tmp_value[50] = Point_Value[20]-Point_Value[32];
		tmp_value[51] = Point_Value[21]-Point_Value[33];  tmp_value[52] = Point_Value[22]-Point_Value[34];  tmp_value[53] = Point_Value[23]-Point_Value[35];
		tmp_value[54] = Point_Value[18]-Point_Value[42];  tmp_value[55] = Point_Value[19]-Point_Value[42];  tmp_value[56] = Point_Value[20]-Point_Value[42];
		tmp_value[57] = Point_Value[21]-Point_Value[42];  tmp_value[58] = Point_Value[22]-Point_Value[42];  tmp_value[59] = Point_Value[23]-Point_Value[42];
		tmp_value[60] = Point_Value[24]-Point_Value[36];  tmp_value[61] = Point_Value[25]-Point_Value[37];  tmp_value[62] = Point_Value[26]-Point_Value[38];
		tmp_value[63] = Point_Value[27]-Point_Value[39];  tmp_value[64] = Point_Value[28]-Point_Value[40];  tmp_value[65] = Point_Value[29]-Point_Value[41];
		tmp_value[66] = Point_Value[30]-Point_Value[42];  tmp_value[67] = Point_Value[31]-Point_Value[42];  tmp_value[68] = Point_Value[32]-Point_Value[42];
		tmp_value[69] = Point_Value[33]-Point_Value[42];  tmp_value[70] = Point_Value[34]-Point_Value[42];  tmp_value[71] = Point_Value[35]-Point_Value[42];
        
		//descriptor generation with normalization
		float square_mag = 0;
		for ( int d=0; d < descriptors.cols; d++ )
		{
			dst[d]=tmp_value[d];
			square_mag += tmp_value[d]*tmp_value[d];
		}
		dst = descriptors.ptr<float>(k);
		float scale = 1./(sqrt(square_mag) + FLT_EPSILON);
		for( int d= 0; d < descriptors.cols; d++ )
		{
			dst[d] *= scale;
		}
	}
}

//this function is used to stack the two compared images
cv::Mat Stack_Imgs( cv::Mat &img1, cv::Mat &img2 )
{
	cv::Mat stacked = Mat::zeros( MAX(img1.rows,img2.rows),img1.cols+img2.cols,img1.type() );

	cv::Mat roiImgResult_left=stacked(cv::Rect(0,0,img1.cols,img1.rows));
	cv::Mat roiImg1 = img1(cv::Rect(0,0,img1.cols,img1.rows));
	roiImg1.copyTo(roiImgResult_left);

	cv::Mat roiImgResult_right=stacked(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
	cv::Mat roiImg2= img2(cv::Rect(0, 0, img2.cols, img2.rows));
	roiImg2.copyTo(roiImgResult_right);
	

	return stacked;
}

/*
this function realize the matching between salient points in compared images
cv::Mat& image_a, cv::Mat& image_b: the compared two images
std::vector<cv::KeyPoint>& keypoints_a, std::vector<cv::KeyPoint> keypoints_b: two sets of detected key points from two compared images
cv::Mat& descriptor_a, cv::Mat& descriptor_b: two sets of calculated descriptors from two compared images
*/
void RIFFDescriptor::Descriptor_Match( cv::Mat& image_a, cv::Mat& image_b, cv::Mat& image_c, cv::Mat& image_d,
									   std::vector<cv::KeyPoint>& keypoints_a, std::vector<cv::KeyPoint> keypoints_b,
									   cv::Mat& descriptor_a, cv::Mat& descriptor_b )
{	
	cv::Mat results;
	cv::Mat distans;

	cv::flann::Index flannIndex(descriptor_b, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);
	results = cv::Mat( descriptor_a.rows, NEAREST_NUMBER, CV_32SC1);
	distans = cv::Mat( descriptor_a.rows, NEAREST_NUMBER, CV_32FC1);
	flannIndex.knnSearch(descriptor_a, results, distans, NEAREST_NUMBER, cv::flann::SearchParams( ) );

	// cross-check
	std::vector<DMatch> Desmatchs(descriptor_a.rows);
	for(int i=0;i<descriptor_a.rows;i++)
	{
		Desmatchs[i].queryIdx=i;
		Desmatchs[i].trainIdx=results.at<int>(i,0);
		Desmatchs[i].distance=(float)distans.at<float>(i,0)/distans.at<float>(i,1);
	}
	vector<DMatch> best_match(descriptor_b.rows);	
	for( int i = 0; i < descriptor_a.rows; i++ )
	{
		DMatch m = Desmatchs[i];
		int i1 = m.trainIdx, iK = m.queryIdx;
		CV_Assert( 0 <= i1 && i1 < descriptor_b.rows && 0 <= iK && iK < descriptor_a.rows );
		if( best_match[i1].trainIdx < 0 || best_match[i1].distance > m.distance )
			best_match[i1] = m;
	}

	// Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
	float nndrRatio = 0.72f;//0.7
	std::vector<cv::Point2f> mpts_1;
	std::vector<cv::Point2f> mpts_2; 
	std::vector<uchar> outlier_mask; 
	for(int i=0; i<descriptor_b.rows; i++)
	{
		int iK = best_match[i].queryIdx;
		if( iK >= 0 && best_match[i].distance<=nndrRatio)
		{
			mpts_1.push_back(keypoints_a.at(best_match[i].queryIdx).pt);
			mpts_2.push_back(keypoints_b.at(best_match[i].trainIdx).pt);			
		}
	}
	// FIND HOMOGRAPHY
	unsigned int nbMatches = 0;
	int bestmatchesnumber=0;
	cv::Mat H;
	if(mpts_1.size() >= nbMatches)
	{
		H = findHomography( mpts_1, mpts_2, cv::RANSAC, 10.0, outlier_mask );
	}
	//draw the matches line with RANSAC
	cv::Point2f pt1, pt2;
	cv::Mat stacked = Stack_Imgs( image_c, image_d );
	rectangle(stacked,cv::Point(799,0),cv::Point(802,640),cv::Scalar(255,255,255),-1);
	for( size_t i = 0; i < mpts_1.size(); ++i )
	{
		if(outlier_mask.at(i))
		{
			pt1 = cv::Point2f( cvRound( mpts_1.at(i).x ), cvRound( mpts_1.at(i).y) );
			pt2 = cv::Point2f( cvRound( mpts_2.at(i).x ), cvRound( mpts_2.at(i).y) );
			pt2.x += image_a.cols;
			cv::circle( stacked, pt1,2, Scalar( 255, 0, 0 ), 3, 1 );
			cv::circle( stacked, pt2,2, Scalar( 255, 0, 0 ), 3, 1 );
			cv::line( stacked, pt1, pt2, CV_RGB( 0, 0, 255 ), 2, 8, 0 );
			bestmatchesnumber++;
		}	
	}
	cout<<bestmatchesnumber<<endl;
	//-- Localize the object  
	//-- Get the corners from the image_a ( the object to be "detected" )  
	std::vector<cv::Point2f> obj_corners;  
	obj_corners.push_back( Point2f(0.0,0.0) );  
	obj_corners.push_back( Point2f((float)image_a.cols,0.0) );  
	obj_corners.push_back( Point2f((float)image_a.cols,(float)image_a.rows) );  
	obj_corners.push_back( Point2f(0.0,(float)image_a.rows) );  

	if (!H.empty())  
	{  
		vector<Point2f> scene_corners;  
		cv::perspectiveTransform(obj_corners, scene_corners, H);  

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )  
		int npts = scene_corners.size();  
		for (int i=0; i<npts; i++)  
			line( stacked, scene_corners[i] + Point2f( image_a.cols, 0),   
			scene_corners[(i+1)%npts] + Point2f( image_a.cols, 0), Scalar(0,255,255), 2 );  
	}  
	cv::imshow("stacked_image",stacked);
	cv::imwrite("RIFF_result.jpg",stacked);
	cv::waitKey(0);
}
