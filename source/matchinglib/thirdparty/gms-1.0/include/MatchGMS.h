#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>

#define THRESH_FACTOR 6 // default value: 6


class MatchGMS
{
public:
	// OpenCV keypoints, corresponding image size and nearest-neighbor descriptor matches
	MatchGMS(
        const std::vector<cv::KeyPoint>& keypoints1,
        cv::Size imageSize1,
        const std::vector<cv::KeyPoint>& keypoints2,
        cv::Size imageSize2,
        const std::vector<cv::DMatch>& descriptorMatches
    );

    // Get inlier indices
    int getInlierMask(
        std::vector<bool>& inlierMask,
        bool useScale = false,
        bool useRotation = false
    );

private:
    // Extract and normalize keypoint position
    void normalizeKeypoints(
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Size& imageSize,
        std::vector<cv::Point2f>& normalizedKeypoints
    );

    // Extract query index and train index of OpenCV matches
    void extractMatches(
        const std::vector<cv::DMatch>& descriptorMatches,
        std::vector<std::pair<int, int> >& matches
    );

    // Initialize grid neighbors
    void initializeNeighbors(
        cv::Mat& gridNeighbors,
        const cv::Size& gridSize
    );

    // Calculate inlier indices for a specific scale level and rotation type
    int run(
        std::vector<bool>& inlierIndices,
        int rotationType = 0
    );

    // Assign matches to cell pairs
    void assignMatchPairs(
        int gridType
    );

    // Verify cell pairs
    void verifyCellPairs(
        int RotationType
    );

    // Set scale level for right grid
    void setScale(
        int scaleLevel = 0
    );

    // Get N9 neighborhood
    std::vector<int> getN9(
        int gridCellIdx,
        const cv::Size& gridSize
    );

    // Get grid index left
    int getGridIndexLeft(
        const cv::Point2f& keypoint,
        int gridType
    );

    // Get grid index right
    int getGridIndexRight(
        const cv::Point2f& keypoint
    );


	std::vector<cv::Point2f> normalizedKeypoints1;   // Normalized keypoints 1
    std::vector<cv::Point2f> normalizedKeypoints2;   // Normalized keypoints 2
    std::vector<std::pair<int, int> > matches;       // Query index and train index of OpenCV matches
	size_t numMatches;                               // Number of matches

	cv::Size gridSizeLeft;      // Grid size left
    cv::Size gridSizeRight;     // Grid size right
    int gridCellsLeft;          // Number of grid cells left
    int gridCellsRight;         // Number of grid cells right
    cv::Mat gridNeighborsLeft;  // Grid neighbors left
    cv::Mat gridNeighborsRight; // Grid neighbors right

    cv::Mat motionStatistics;                       // x: left grid idx | y: right grid idx | value: how many matches from idx_left to idx_right
    std::vector<std::pair<int, int> > matchPairs;   // Every match has a cell-pair | first: grid_idx_left | second: grid_idx_right
    std::vector<int> cellPairs;                     // Index: grid_idx_left, Value: grid_idx_right
    std::vector<int> numPointsInCellLeft;           // Number of points in left grid cell left
};
