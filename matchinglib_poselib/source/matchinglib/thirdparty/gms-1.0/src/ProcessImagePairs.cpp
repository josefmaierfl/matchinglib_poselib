#include "ProcessImagePairs.h"

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


ProcessImagePairs::ProcessImagePairs(string& imagePath, unsigned int imageIdx, unsigned int imageOffset,
                                     unsigned int numFeatures, unsigned int imageHeight):
    imagePath(imagePath),
    imageIdx(imageIdx),
    imageOffset(imageOffset),
    imageHeight(imageHeight),
    numInliers(0)
{
    // Initialize ORB features
    orb = ORB::create(numFeatures);
    orb->setFastThreshold(0);

    // Initialize matcher
#ifdef USE_GPU
    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
#else
    matcher = BFMatcher(NORM_HAMMING);
#endif
}

void ProcessImagePairs::computeMatch()
{
    while (loadImages())
    {
        generateFeatures();
        matchFeatures();
        extractFeatures();

        drawFeatures(LineCircle);
        printStats();
    }
}

void ProcessImagePairs::extractFeatures()
{
    // Calculate GMS feature correspondence
    MatchGMS gms(keypoints[0], image[0].size(), keypoints[1], image[1].size(), matchesORB);
    numInliers = gms.getInlierMask(inlierIndices, true, false);  // default values: false, false

    // Extracted GMS matches
    matchesGMS.clear();
    for (size_t i = 0; i < inlierIndices.size(); ++i)
    {
        if (inlierIndices[i] == true)
            matchesGMS.push_back(matchesORB[i]);
    }
}

void ProcessImagePairs::generateFeatures()
{
    // Generate ORB features
    orb->detectAndCompute(image[0], Mat(), keypoints[0], descriptors[0]);
    orb->detectAndCompute(image[1], Mat(), keypoints[1], descriptors[1]);
}

bool ProcessImagePairs::loadImages()
{
    char file[255];

    snprintf(file, 255, imagePath.c_str(), imageIdx);
    image[0] = imread(string(file));
    if(image[0].empty())
        return false;

    if (image[0].channels() == 3)
        cvtColor(image[0], image[0], CV_BGR2GRAY);

    snprintf(file, 255, imagePath.c_str(), imageIdx + imageOffset);
    image[1] = imread(string(file));
    if(image[1].empty())
        return false;

    if (image[1].channels() == 3)
        cvtColor(image[1], image[1], CV_BGR2GRAY);

    imageIdx++;

    resizeImage(image[0], imageHeight);
    resizeImage(image[1], imageHeight);

    return true;
}

void ProcessImagePairs::matchFeatures()
{
#ifdef USE_GPU
    descriptorsGPU[0] = GpuMat(descriptors[0]);
    descriptorsGPU[1] = GpuMat(descriptors[1]);
    matcher->match(descriptorsGPU[0], descriptorsGPU[1], matchesORB);
#else
    matcher.match(descriptors[0], descriptors[1], matchesORB, noArray());
#endif
}

void ProcessImagePairs::drawFeatures(DrawType type)
{
    static RNG colorRNG(12345);

    const int height = max(image[0].rows, image[1].rows);
    const int width = image[0].cols + image[1].cols;

    Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
    cvtColor(image[0], image[0], CV_GRAY2BGR);
    cvtColor(image[1], image[1], CV_GRAY2BGR);
    image[0].copyTo(output(Rect(0, 0, image[0].cols, image[0].rows)));
    image[1].copyTo(output(Rect(image[0].cols, 0, image[1].cols, image[1].rows)));

    switch(type)
    {
        case Line:
            for (size_t i = 0; i < matchesGMS.size(); i++)
            {
                Point2f left = keypoints[0][matchesGMS[i].queryIdx].pt;
                Point2f right = (keypoints[1][matchesGMS[i].trainIdx].pt + Point2f((float) image[0].cols, 0.0f));
                line(output, left, right, Scalar(0, 255, 255));
            }
            break;

        case Circle:
            for(size_t i = 0; i < matchesGMS.size(); i++)
            {
                Point2f left = keypoints[0][matchesGMS[i].queryIdx].pt;
                Point2f right = (keypoints[1][matchesGMS[i].trainIdx].pt + Point2f((float) image[0].cols, 0.0f));

                Scalar color = Scalar(colorRNG.uniform(0, 255), colorRNG.uniform(0, 255), colorRNG.uniform(0, 255));
                circle(output, left, 5, color, 1);
                circle(output, right, 5, color, 1);
            }
            break;

        case LineCircle:
            for (size_t i = 0; i < matchesGMS.size(); i++)
            {
                Point2f left = keypoints[0][matchesGMS[i].queryIdx].pt;
                Point2f right = (keypoints[1][matchesGMS[i].trainIdx].pt + Point2f((float) image[0].cols, 0.0f));
                line(output, left, right, Scalar(0, 255, 0));
            }

            for (size_t i = 0; i < matchesGMS.size(); i++)
            {
                Point2f left = keypoints[0][matchesGMS[i].queryIdx].pt;
                Point2f right = (keypoints[1][matchesGMS[i].trainIdx].pt + Point2f((float) image[0].cols, 0.f));
                circle(output, left, 2, Scalar(255, 0, 0), 2);
                circle(output, right, 2, Scalar(255, 0, 0), 2);
            }
            break;
    }

    imshow("Extracted features", output);
    waitKey(1);
}

void ProcessImagePairs::resizeImage(Mat &image, int height)
{
    double ratio = image.rows * 1.0 / height;
    int width = static_cast<int>(image.cols * 1.0 / ratio);
    resize(image, image, Size(width, height));
}

void ProcessImagePairs::printStats()
{
    printf("IMG %5u      ORB %5u      GMS %5u      GMS/ORB %.3f\n",
           imageIdx, static_cast<unsigned int>(matchesORB.size()), numInliers, numInliers / (float) matchesORB.size());
}
