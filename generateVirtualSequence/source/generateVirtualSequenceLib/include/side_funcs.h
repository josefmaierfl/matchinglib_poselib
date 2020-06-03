//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2019 AIT Austrian Institute of Technology GmbH
//
//Permission is hereby granted, free of charge, to any person obtaining
//a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation
//the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included
//in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)
//
// Created by maierj on 04.03.19.
//

#ifndef GENERATEVIRTUALSEQUENCE_SIDE_FUNCS_H
#define GENERATEVIRTUALSEQUENCE_SIDE_FUNCS_H

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"
#include <pcl/visualization/pcl_visualizer.h>

void gen_palette(int num_labels, std::vector<cv::Vec3b> &pallete);

void color_HSV2RGB(float H, float S, float V, unsigned char &R, unsigned char &G, unsigned char &B);

void buildColorMapHSV2RGB(const cv::Mat &in16, cv::Mat &rgb8, uint16_t nrLabels, cv::InputArray mask);

void startPCLViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

void setPCLViewerCamPars(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                         Eigen::Matrix4f cam_extrinsics,
                         const cv::Mat &K1);

Eigen::Affine3f initPCLViewerCoordinateSystems(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                                               cv::InputArray R_C2W = cv::noArray(),
                                               cv::InputArray t_C2W = cv::noArray());

void getNColors(cv::OutputArray colorMat, size_t nr_Colors, int colormap);

void
getCloudCentroids(std::vector<pcl::PointCloud<pcl::PointXYZ>> &pointclouds, std::vector<pcl::PointXYZ> &cloudCentroids);

void getCloudCentroids(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &pointclouds,
                       std::vector<pcl::PointXYZ> &cloudCentroids);

void getCloudCentroid(pcl::PointCloud<pcl::PointXYZ> &pointcloud, pcl::PointXYZ &cloudCentroid);

void getMeanCloudStandardDevs(std::vector<pcl::PointCloud<pcl::PointXYZ>> &pointclouds,
                              std::vector<float> &cloudExtensions,
                              std::vector<pcl::PointXYZ> &cloudCentroids);

void getMeanCloudStandardDev(pcl::PointCloud<pcl::PointXYZ> &pointcloud, float &cloudExtension,
                             pcl::PointXYZ &cloudCentroid);

Eigen::Affine3f addVisualizeCamCenter(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                                      const cv::Mat &R,
                                      const cv::Mat &t);

void getCloudDimensionStdDev(pcl::PointCloud<pcl::PointXYZ> &pointcloud, pcl::PointXYZ &cloudDim,
                             pcl::PointXYZ &cloudCentroid);

void getSecPartContourPos(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd);

void getSecPartContourNeg(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd);

void getFirstPartContourPos(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd);

void getFirstPartContourNeg(std::vector<cv::Point> &target, std::vector<cv::Point> &source, int idxStart, int idxEnd);

bool checkPointValidity(const cv::Mat &mask, const cv::Point_<int32_t> &pt);

bool getValidRegBorders(const cv::Mat &mask, cv::Rect &validRect);

int deletedepthCatsByIdx(std::vector<std::vector<std::vector<cv::Point_<int32_t>>>> &seedsFromLast,
                         std::vector<size_t> &delListCorrs,
                         const cv::Mat &ptMat);

int deletedepthCatsByNr(std::vector<cv::Point_<int32_t>> &seedsFromLast,
                        int32_t nrToDel,
                        const cv::Mat &ptMat,
                        std::vector<size_t> &delListCorrs);

int deletedepthCatsByNr(std::vector<cv::Point2d> &seedsFromLast,
                        int32_t nrToDel,
                        const cv::Mat &ptMat,
                        std::vector<size_t> &delListCorrs);

/*template<typename T, typename A, typename T1, typename A1>
void deleteVecEntriesbyIdx(std::vector<T, A> &editVec, std::vector<T1, A1> const& delVec);*/

//Deletes some entries within a vector using a vector with indices that point to the entries to delete.
//The deletion vector containing the indices must be sorted in ascending order.
template<typename T, typename A, typename T1, typename A1>
void deleteVecEntriesbyIdx(std::vector<T, A> &editVec, std::vector<T1, A1> const &delVec) {
    size_t nrToDel = delVec.size();
    CV_Assert(nrToDel <= editVec.size());
    size_t n_new = editVec.size() - nrToDel;
    std::vector<T, A> editVecNew(n_new);
    T1 old_idx = 0;
    int startRowNew = 0;
    for (size_t i = 0; i < nrToDel; i++) {
        if (old_idx == delVec[i]) {
            old_idx = delVec[i] + 1;
            continue;
        }
        const int nr_new_cpy_elements = (int) delVec[i] - (int) old_idx;
        const int endRowNew = startRowNew + nr_new_cpy_elements;
        std::copy(editVec.begin() + old_idx, editVec.begin() + delVec[i], editVecNew.begin() + startRowNew);

        startRowNew = endRowNew;
        old_idx = delVec[i] + 1;
    }
    if (old_idx < editVec.size()) {
        std::copy(editVec.begin() + old_idx, editVec.end(), editVecNew.begin() + startRowNew);
    }
    editVec = editVecNew;
}

/*template<typename T, typename A>
void deleteMatEntriesByIdx(cv::Mat &editMat, std::vector<T, A> const& delVec, bool rowOrder);*/

//Deletes some entries within a Mat using a vector with indices that point to the entries to delete.
//The deletion vector containing the indices must be sorted in ascending order.
//It can be specified if the entries in Mat are colum ordered (rowOrder=false) or row ordered (rowOrder=true).
template<typename T, typename A>
void deleteMatEntriesByIdx(cv::Mat &editMat, std::vector<T, A> const &delVec, bool rowOrder) {
    size_t nrToDel = delVec.size();
    size_t nrData;
    if (rowOrder)
        nrData = (size_t) editMat.rows;
    else
        nrData = (size_t) editMat.cols;

    CV_Assert(nrToDel <= nrData);

    int n_new = (int)nrData - (int)nrToDel;
    cv::Mat editMatNew;
    if (rowOrder)
        editMatNew = cv::Mat(n_new, editMat.cols, editMat.type());
    else
        editMatNew = cv::Mat(editMat.rows, n_new, editMat.type());

    T old_idx = 0;
    int startRowNew = 0;
    for (size_t i = 0; i < nrToDel; i++) {
        if (old_idx == delVec[i]) {
            old_idx = delVec[i] + 1;
            continue;
        }
        const int nr_new_cpy_elements = (int) delVec[i] - (int) old_idx;
        const int endRowNew = startRowNew + nr_new_cpy_elements;
        if (rowOrder)
            editMat.rowRange((int) old_idx, (int) delVec[i]).copyTo(editMatNew.rowRange(startRowNew, endRowNew));
        else
            editMat.colRange((int) old_idx, (int) delVec[i]).copyTo(editMatNew.colRange(startRowNew, endRowNew));

        startRowNew = endRowNew;
        old_idx = delVec[i] + 1;
    }
    if ((size_t) old_idx < nrData) {
        if (rowOrder)
            editMat.rowRange((int) old_idx, editMat.rows).copyTo(editMatNew.rowRange(startRowNew, n_new));
        else
            editMat.colRange((int) old_idx, editMat.cols).copyTo(editMatNew.colRange(startRowNew, n_new));
    }
    editMatNew.copyTo(editMat);
}

/*Rounds a rotation matrix to its nearest integer values and checks if it is still a rotation matrix and does not change more than 22.5deg from the original rotation matrix.
As an option, the error of the rounded rotation matrix can be compared to an angular difference of a second given rotation matrix R_fixed to R_old.
The rotation matrix with the smaller angular difference is selected.
This function is used to select a proper rotation matrix if the "look at" and "up vector" are nearly equal. I trys to find the nearest rotation matrix aligened to the
"look at" vector taking into account the rotation matrix calculated from the old/last "look at" vector
*/
bool roundR(const cv::Mat R_old, cv::Mat & R_round, cv::InputArray R_fixed = cv::noArray());

#endif //GENERATEVIRTUALSEQUENCE_SIDE_FUNCS_H
