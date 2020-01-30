#include "MatchGMS.h"

#ifdef USE_GPU
#include <opencv2/cudafeatures2d.hpp>
    using cv::cuda::GpuMat;
#else
#include <opencv2/features2d.hpp>
#endif


class ProcessImagePairs
{
public:
    ProcessImagePairs(
        std::string& imagePath,
        unsigned int imageIdx = 0,
        unsigned int imageOffset = 50,
        unsigned int numFeatures = 1000,
        unsigned int imageHeight = 480
    );

    void computeMatch();

private:
    enum DrawType
    {
        Line,
        Circle,
        LineCircle
    };

    void extractFeatures();
    void generateFeatures();
    bool loadImages();
    void matchFeatures();

    void drawFeatures(DrawType type);
    void resizeImage(cv::Mat &image, int height);
    void printStats();

    cv::Mat image[2];
    const std::string imagePath;
    unsigned int imageIdx;
    const unsigned int imageOffset;
    const unsigned int imageHeight;

    std::vector<cv::KeyPoint> keypoints[2];
    cv::Mat descriptors[2];
    std::vector<cv::DMatch> matchesORB;
    std::vector<cv::DMatch> matchesGMS;

    cv::Ptr<cv::ORB> orb;

#ifdef USE_GPU
    GpuMat descriptorsGPU[2];
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
#else
    cv::BFMatcher matcher;
#endif

    int numInliers;
    std::vector<bool> inlierIndices;
};