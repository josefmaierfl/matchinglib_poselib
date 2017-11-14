#include "MatchGMS.h"

using namespace std;
using namespace cv;


// 8 possible rotations
const int rotationPatterns[8][9] = {
    1,2,3,
    4,5,6,
    7,8,9,

    4,1,2,
    7,5,3,
    8,9,6,

    7,4,1,
    8,5,2,
    9,6,3,

    8,7,4,
    9,5,1,
    6,3,2,

    9,8,7,
    6,5,4,
    3,2,1,

    6,9,8,
    3,5,7,
    2,1,4,

    3,6,9,
    2,5,8,
    1,4,7,

    2,3,6,
    1,5,9,
    4,7,8
};

// 5 scale levels
const double scaleRatios[5] = {
    1.0,
    1.0 / 2,
    1.0 / sqrt(2.0),
    sqrt(2.0),
    2.0
};

MatchGMS::MatchGMS(const vector<KeyPoint>& keypoints1, const Size imageSize1,
                   const vector<KeyPoint>& keypoints2, const Size imageSize2,
                   const vector<DMatch>& descriptorMatches)
{
    // Initialize keypoints and matches
    normalizeKeypoints(keypoints1, imageSize1, normalizedKeypoints1);
    normalizeKeypoints(keypoints2, imageSize2, normalizedKeypoints2);
    numMatches = descriptorMatches.size();
    extractMatches(descriptorMatches, matches);

    // Initialize left grid
    gridSizeLeft = Size(20, 20);
    gridCellsLeft = gridSizeLeft.width * gridSizeLeft.height;

    // right grid will be initialized according to scale
    gridSizeRight = Size(0, 0);
    gridCellsRight = 0;

    // Initialize neighbors of left grid
    gridNeighborsLeft = Mat::zeros(gridCellsLeft, 9, CV_32SC1);
    initializeNeighbors(gridNeighborsLeft, gridSizeLeft);
}

int MatchGMS::getInlierMask(vector<bool>& inlierMask, const bool useScale, const bool useRotation)
{
    const int nScales = useScale ? 5 : 1;
    const int nRotationTypes = useRotation ? 8 : 1;

    int maxInlier = 0;
    vector<bool> inlierIndices;

    for (int scaleLevel = 0; scaleLevel < nScales; scaleLevel++)
    {
        setScale(scaleLevel);
        for (int rotationType = 0; rotationType < nRotationTypes; rotationType++)
        {

            int numInlier = run(inlierIndices, rotationType);

            if (numInlier > maxInlier)
            {
                inlierMask = inlierIndices;
                maxInlier = numInlier;
            }
        }
    }

    return maxInlier;
}

void MatchGMS::normalizeKeypoints(const vector<KeyPoint>& keypoints, const Size& imageSize,
                                  vector<Point2f>& normalizedKeypoints)
{
    const float widthInv  = 1.0f / imageSize.width;
    const float heightInv = 1.0f / imageSize.height;

    const size_t numKeypoints = keypoints.size();

    normalizedKeypoints.resize(numKeypoints);
    for (size_t i = 0; i < numKeypoints; i++)
    {
        normalizedKeypoints[i].x = keypoints[i].pt.x * widthInv;
        normalizedKeypoints[i].y = keypoints[i].pt.y * heightInv;
    }
}

void MatchGMS::extractMatches(const vector<DMatch>& descriptorMatches, vector<pair<int, int> >& matches)
{
    matches.resize(numMatches);
    for (size_t i = 0; i < numMatches; i++)
    {
        matches[i] = pair<int, int>(descriptorMatches[i].queryIdx, descriptorMatches[i].trainIdx);
    }
}

void MatchGMS::initializeNeighbors(Mat& gridNeighbors, const Size& gridSize)
{
    int gridCells = gridNeighbors.rows;

    for (int i = 0; i < gridCells; i++)
    {
        vector<int> N9 = getN9(i, gridSize);
        int *data = gridNeighbors.ptr<int>(i);
        memcpy(data, &N9[0], sizeof(int) * 9);
    }
}

int MatchGMS::run(std::vector<bool>& inlierIndices, int rotationType)
{
    // Initialize inlier indices, motion statistics and match pairs
    inlierIndices.assign(numMatches, false);
    motionStatistics = Mat::zeros(gridCellsLeft, gridCellsRight, CV_32SC1);
    matchPairs.assign(numMatches, pair<int, int>(0, 0));

    for (int gridType = 1; gridType <= 4; gridType++)
    {
        // Initialize
        motionStatistics.setTo(0);
        cellPairs.assign(gridCellsLeft, -1);
        numPointsInCellLeft.assign(gridCellsLeft, 0);

        assignMatchPairs(gridType);
        verifyCellPairs(rotationType);

        // Mark inliers
        for (size_t i = 0; i < numMatches; i++)
        {
            if (cellPairs[matchPairs[i].first] == matchPairs[i].second)
            {
                inlierIndices[i] = true;
            }
        }
    }
    int numInlier = static_cast<int>(sum(inlierIndices)[0]);
    return numInlier;
}

void MatchGMS::assignMatchPairs(int gridType)
{
    for (size_t i = 0; i < numMatches; i++)
    {
        Point2f &leftKeypoint = normalizedKeypoints1[matches[i].first];
        Point2f &rightKeypoint = normalizedKeypoints2[matches[i].second];

        int leftGridIdx = matchPairs[i].first = getGridIndexLeft(leftKeypoint, gridType);
        int rightGridIdx;
        if (gridType == 1)
        {
            rightGridIdx = matchPairs[i].second = getGridIndexRight(rightKeypoint);
        }
        else
        {
            rightGridIdx = matchPairs[i].second;
        }

        if (leftGridIdx < 0 || rightGridIdx < 0)
            continue;

        motionStatistics.at<int>(leftGridIdx, rightGridIdx)++;
        numPointsInCellLeft[leftGridIdx]++;
    }
}

void MatchGMS::verifyCellPairs(int RotationType)
{
    const int *currentRotationPattern = rotationPatterns[RotationType];

    for (int i = 0; i < gridCellsLeft; i++)
    {
        if (sum(motionStatistics.row(i))[0] == 0)
        {
            cellPairs[i] = -1;
            continue;
        }

        int maxNumber = 0;
        for (int j = 0; j < gridCellsRight; j++)
        {
            int *value = motionStatistics.ptr<int>(i);
            if (value[j] > maxNumber)
            {
                cellPairs[i] = j;
                maxNumber = value[j];
            }
        }

        int gridIdxRight = cellPairs[i];

        const int *leftN9 = gridNeighborsLeft.ptr<int>(i);
        const int *rightN9 = gridNeighborsRight.ptr<int>(gridIdxRight);

        int score = 0;
        double thresh = 0;
        int numPair = 0;

        for (size_t j = 0; j < 9; j++)
        {
            int ll = leftN9[j];
            int rr = rightN9[currentRotationPattern[j] - 1];
            if (ll == -1 || rr == -1)
                continue;

            score += motionStatistics.at<int>(ll, rr);
            thresh += numPointsInCellLeft[ll];
            numPair++;
        }

        thresh = THRESH_FACTOR * sqrt(thresh / numPair);

        if (score < thresh)
            cellPairs[i] = -2;
    }
}

void MatchGMS::setScale(int scaleLevel)
{
    // Set scale level of right grid
    gridSizeRight.width = static_cast<int>(gridSizeLeft.width * scaleRatios[scaleLevel]);
    gridSizeRight.height = static_cast<int>(gridSizeLeft.height * scaleRatios[scaleLevel]);
    gridCellsRight = gridSizeRight.width * gridSizeRight.height;

    // Initialize the neighbors of right grid
    gridNeighborsRight = Mat::zeros(gridCellsRight, 9, CV_32SC1);
    initializeNeighbors(gridNeighborsRight, gridSizeRight);
}

vector<int> MatchGMS::getN9(const int gridCellIdx, const Size& gridSize)
{
    vector<int> N9(9, -1);

    int idx_x = gridCellIdx % gridSize.width;
    int idx_y = gridCellIdx / gridSize.width;

    for (int yi = -1; yi <= 1; yi++)
    {
        for (int xi = -1; xi <= 1; xi++)
        {
            int idx_xx = idx_x + xi;
            int idx_yy = idx_y + yi;

            if (idx_xx < 0 || idx_xx >= gridSize.width || idx_yy < 0 || idx_yy >= gridSize.height)
                continue;

            N9[xi + 4 + yi * 3] = idx_xx + idx_yy * gridSize.width;
        }
    }
    return N9;
}

int MatchGMS::getGridIndexLeft(const Point2f& keypoint, int gridType)
{
    int x = 0;
    int y = 0;

    if (gridType == 1)
    {
        x = static_cast<int>(floor(keypoint.x * gridSizeLeft.width));
        y = static_cast<int>(floor(keypoint.y * gridSizeLeft.height));
    }
    else if (gridType == 2)
    {
        x = static_cast<int>(floor(keypoint.x * gridSizeLeft.width + 0.5f));
        y = static_cast<int>(floor(keypoint.y * gridSizeLeft.height));
    }
    else if (gridType == 3)
    {
        x = static_cast<int>(floor(keypoint.x * gridSizeLeft.width));
        y = static_cast<int>(floor(keypoint.y * gridSizeLeft.height + 0.5f));
    }
    else if (gridType == 4)
    {
        x = static_cast<int>(floor(keypoint.x * gridSizeLeft.width + 0.5f));
        y = static_cast<int>(floor(keypoint.y * gridSizeLeft.height + 0.5f));
    }

    if (x >= gridSizeLeft.width || y >= gridSizeLeft.height)
    {
        return -1;
    }

    return x + y * gridSizeLeft.width;
}

int MatchGMS::getGridIndexRight(const Point2f& keypoint)
{
    int x = static_cast<int>(floor(keypoint.x * gridSizeRight.width));
    int y = static_cast<int>(floor(keypoint.y * gridSizeRight.height));

    return x + y * gridSizeRight.width;
}
