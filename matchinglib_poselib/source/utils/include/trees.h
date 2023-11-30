//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2021 Josef Maier
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

#pragma once

#include <vector>
#include <memory>

#include <opencv2/highgui.hpp>
#include <nanoflann.hpp>

#include "utilslib/utilslib_api.h"

namespace utilslib
{
    struct UTILSLIB_API FeatureDataStorage
    {
        struct FeatureData
        {
            cv::KeyPoint kp;
            cv::Mat descr;

            FeatureData() = default;
            FeatureData(const cv::KeyPoint &kp_, const cv::Mat &descr_) : kp(kp_), descr(descr_.clone()) {}
            FeatureData(FeatureData &&source) : kp(std::move(source.kp)), descr(std::move(source.descr)) {}
        };

        void add(const cv::KeyPoint &kp, const cv::Mat &descr)
        {
            data.emplace_back(FeatureData(kp, descr));
        }

        std::vector<FeatureData> data;

        FeatureDataStorage() = default;
        FeatureDataStorage(FeatureDataStorage &&source) : data(std::move(source.data)) {}

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const
        {
            return data.size();
        }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline float kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            if (dim == 0)
                return data.at(idx).kp.pt.x;
            else
                return data.at(idx).kp.pt.y;
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
    };

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, FeatureDataStorage>, FeatureDataStorage, 2 /* dim */, size_t> FeatureFlannTree;

    class UTILSLIB_API FeatureKDTree
    {
    public:
        explicit FeatureKDTree(const float &accuracy = 0.5f);
        FeatureKDTree(FeatureKDTree &&source) : data(source.data), tree(source.tree), single_search_acc(source.single_search_acc) {}

        void add(const cv::KeyPoint &kp, const cv::Mat &descr);
        void buildIndex();
        bool getDescriptor(const cv::Point2f &pt, cv::Mat &descr) const;
        bool getDescriptor(const cv::KeyPoint &kp, cv::Mat &descr) const;
        bool getDescriptorRef(const cv::Point2f &pt, cv::Mat &descr) const;
        bool getDescriptorRef(const cv::KeyPoint &kp, cv::Mat &descr) const;
        bool getDescriptorDistance(const cv::Point2f &pt, const cv::Mat &descr2, double &distance) const;
        bool getDescriptorDistance(const cv::KeyPoint &kp, const cv::Mat &descr2, double &distance) const;
        bool getKeypoint(const cv::Point2f &pt, cv::KeyPoint &kp) const;
        bool getKeypointDescriptor(const cv::Point2f &pt, cv::KeyPoint &kp, cv::Mat &descr) const;
        bool getBestKeypointDescriptorMatch(const cv::Point2f &pt, const cv::Mat &descr, cv::KeyPoint &kp2, cv::Mat &descr2, double &descr_dist, const double &max_descr_dist, const float &searchRadius) const;
        bool getKeypointDescriptorMatches(const cv::Point2f &pt, const cv::Mat &descr, std::vector<cv::KeyPoint> &kp2, cv::Mat &descr2, std::vector<double> &descr_dist, const double &max_descr_dist, const float &searchRadius) const;
        bool getKeypointDescriptorMatches(const cv::Point2f &pt, std::vector<cv::KeyPoint> &kp2, cv::Mat &descr2, const float &searchRadius) const;
        bool getKeypointDescriptorMatches(const cv::Point2f &pt, std::vector<cv::KeyPoint> &kp2, const float &searchRadius) const;

    private:
        bool getIndex(const cv::Point2f &pt, size_t &idx) const;
        size_t radiusSearch(const cv::Point2f &pt, const float &radius, std::vector<std::pair<size_t, float>> &ret_matches, const bool &sorted = false) const;

        std::shared_ptr<FeatureDataStorage> data;
        std::shared_ptr<FeatureFlannTree> tree;
        float single_search_acc;
    };

    struct UTILSLIB_API Circle
    {
        cv::Point2d midpoint;
        double radius;

        Circle() : radius(-1.)
        {
            midpoint.x = -1.;
            midpoint.y = -1.;
        }
        Circle(const cv::Point2d &mid, const double &r) : midpoint(mid), radius(r) {}
        Circle(const cv::Point2f &mid, const float &r)
        {
            midpoint.x = static_cast<double>(mid.x);
            midpoint.y = static_cast<double>(mid.y);
            radius = static_cast<double>(r);
        }
    };

    class UTILSLIB_API LineSearchNode
    {
    public:
        explicit LineSearchNode(const Circle &innerCircle, const size_t &max_leaf_entries = 10);
        LineSearchNode(const cv::KeyPoint &kp, const cv::Mat &descr, const Circle &innerCircle, const size_t &max_leaf_entries);
        void add(const cv::KeyPoint &kp, const cv::Mat &descr);
        bool containsLine(const cv::Mat &line, const double &distance, const double &ab_length) const;
        bool getPointsNearLine(const cv::Mat &line, std::vector<std::pair<cv::KeyPoint, cv::Mat>> &found_entries, const double &distance, const double &ab_length = 0) const;

        // ~LineSearchNode(){
        //   entries.clear();
        //   for(auto &child: childs){
        //     if(child){
        //       child.reset();
        //     }
        //   }
        // }

    private:
        const Circle innerCircle_;
        const size_t max_leaf_entries_;
        bool hasChilds;
        std::vector<std::pair<cv::KeyPoint, cv::Mat>> entries;
        double outerCircle_radius;
        std::array<std::unique_ptr<LineSearchNode>, 4> childs;
        cv::Mat l_t, l_b, l_l, l_r;
        double x_min, x_max, y_min, y_max;

        void computeProperties();
        void addToChild(const cv::KeyPoint &kp, const cv::Mat &descr);
        void createNewNode(const cv::KeyPoint &kp, const cv::Mat &descr, const unsigned char &pos);
        bool lineCrossesRectangle(const cv::Mat &line) const;
        double getLineDistance(const cv::Mat &line, const cv::Point2d &pt, const double &ab_length) const;
        double getLineDistance(const cv::Mat &line, const cv::Point2f &pt, const double &ab_length) const;
        bool linesParallel(const cv::Mat &line1, const cv::Mat &line2) const;
        void getParallelLines(const cv::Mat &line_in, const double &ab_length, const double &distance, cv::Mat &line1_out, cv::Mat &line2_out) const;
    };

    class UTILSLIB_API LineSearchTree
    {
    public:
        explicit LineSearchTree(const size_t &max_leaf_entries = 10) : nr_leaf(max_leaf_entries) {}
        LineSearchTree(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors, const size_t &max_leaf_entries = 10) : nr_leaf(max_leaf_entries)
        {
            outer_circle = getOuterCircle(keypoints);
            buildTree(keypoints, descriptors);
        }
        // ~LineSearchTree(){
        //   tree.reset();
        // }

        Circle getOuterCircle(const std::vector<cv::KeyPoint> &keypoints);
        void defineOuterCircle(const std::vector<cv::KeyPoint> &keypoints);
        void buildTree(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);
        bool getDataNearLine(const cv::Mat &line, const double &distance, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const;
        bool getBestMatch(const cv::Mat &x1, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::Mat &descriptor_match, cv::DMatch &match) const;
        bool getBestMatch(const cv::Point2f &pt, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::Mat &descriptor_match, cv::DMatch &match) const;
        bool getBestMatchSampson(const cv::Mat &x1, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::OutputArray descriptor_match = cv::noArray(), cv::DMatch *match = nullptr) const;
        bool getBestMatchSampson(const cv::Mat &x1, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::Point2f &pt_match, cv::OutputArray descriptor_match = cv::noArray(), cv::DMatch *match = nullptr) const;
        bool getBestMatchSampson(const cv::Point2f &pt, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::OutputArray descriptor_match = cv::noArray(), cv::DMatch *match = nullptr) const;
        bool getBestMatchSampson(const cv::Point2f &pt, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::Point2f &pt_match, cv::OutputArray descriptor_match = cv::noArray(), cv::DMatch *match = nullptr) const;

    protected:
        Circle outer_circle;
        size_t nr_leaf = 10;
        std::shared_ptr<LineSearchNode> tree;
    };
}
