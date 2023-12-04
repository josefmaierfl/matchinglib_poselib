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

#include <matchinglib/trees.h>
#include <matchinglib/matching_structs.h>
#include "opencv2/imgproc.hpp"

namespace matchinglib
{
    FeatureKDTree::FeatureKDTree(const float &accuracy) : single_search_acc(accuracy)
    {
        data = std::make_shared<FeatureDataStorage>();
    }

    void FeatureKDTree::add(const cv::KeyPoint &kp, const cv::Mat &descr)
    {
        CV_Assert(descr.rows == 1);
        data->add(kp, descr);
    }

    void FeatureKDTree::buildIndex()
    {
        if (tree)
        {
            tree.reset(new FeatureFlannTree(2, *(data.get()), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
        }
        else
        {
            tree = std::make_shared<FeatureFlannTree>(2, *(data.get()), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        }
        tree->buildIndex();
    }

    bool FeatureKDTree::getDescriptor(const cv::Point2f &pt, cv::Mat &descr) const
    {
        size_t idx;
        if (!getIndex(pt, idx))
        {
            return false;
        }
        descr = data->data.at(idx).descr.clone();
        return true;
    }

    bool FeatureKDTree::getDescriptor(const cv::KeyPoint &kp, cv::Mat &descr) const
    {
        return getDescriptor(kp.pt, descr);
    }

    bool FeatureKDTree::getDescriptorRef(const cv::Point2f &pt, cv::Mat &descr) const
    {
        size_t idx;
        if (!getIndex(pt, idx))
        {
            return false;
        }
        descr = data->data.at(idx).descr;
        return true;
    }

    bool FeatureKDTree::getDescriptorRef(const cv::KeyPoint &kp, cv::Mat &descr) const
    {
        return getDescriptorRef(kp.pt, descr);
    }

    bool FeatureKDTree::getDescriptorDistance(const cv::Point2f &pt, const cv::Mat &descr2, double &distance) const
    {
        cv::Mat descr1;
        if (!getDescriptorRef(pt, descr1))
        {
            return false;
        }
        distance = getDescriptorDist(descr1, descr2);
        return true;
    }

    bool FeatureKDTree::getDescriptorDistance(const cv::KeyPoint &kp, const cv::Mat &descr2, double &distance) const
    {
        return getDescriptorDistance(kp.pt, descr2, distance);
    }

    bool FeatureKDTree::getKeypoint(const cv::Point2f &pt, cv::KeyPoint &kp) const
    {
        size_t idx;
        if (!getIndex(pt, idx))
        {
            return false;
        }
        kp = data->data.at(idx).kp;
        return true;
    }

    bool FeatureKDTree::getKeypointDescriptor(const cv::Point2f &pt, cv::KeyPoint &kp, cv::Mat &descr) const
    {
        size_t idx;
        if (!getIndex(pt, idx))
        {
            return false;
        }
        kp = data->data.at(idx).kp;
        data->data.at(idx).descr.copyTo(descr);
        return true;
    }

    bool FeatureKDTree::getBestKeypointDescriptorMatch(const cv::Point2f &pt, const cv::Mat &descr, cv::KeyPoint &kp2, cv::Mat &descr2, double &descr_dist, const double &max_descr_dist, const float &searchRadius) const
    {
        std::vector<std::pair<size_t, float>> ret_matches;
        size_t nMatches = radiusSearch(pt, searchRadius, ret_matches);
        if (!nMatches)
        {
            return false;
        }
        double match_dist_smallest = DBL_MAX;
        size_t best_idx = 0;
        bool found = false;
        for (auto &m : ret_matches)
        {
            const double match_dist = getDescriptorDist(data->data.at(m.first).descr, descr);
            if (match_dist <= max_descr_dist && match_dist < match_dist_smallest)
            {
                best_idx = m.first;
                match_dist_smallest = match_dist;
                found = true;
            }
        }
        if (!found)
        {
            return false;
        }
        kp2 = data->data.at(best_idx).kp;
        data->data.at(best_idx).descr.copyTo(descr2);
        descr_dist = match_dist_smallest;
        return true;
    }

    bool FeatureKDTree::getKeypointDescriptorMatches(const cv::Point2f &pt, const cv::Mat &descr, std::vector<cv::KeyPoint> &kp2, cv::Mat &descr2, std::vector<double> &descr_dist, const double &max_descr_dist, const float &searchRadius) const
    {
        std::vector<std::pair<size_t, float>> ret_matches;
        size_t nMatches = radiusSearch(pt, searchRadius, ret_matches);
        if (!nMatches)
        {
            return false;
        }
        descr2.release();
        kp2.clear();
        descr_dist.clear();
        for (auto &m : ret_matches)
        {
            const double match_dist = getDescriptorDist(data->data.at(m.first).descr, descr);
            if (match_dist <= max_descr_dist)
            {
                const size_t best_idx = m.first;
                kp2.emplace_back(data->data.at(best_idx).kp);
                descr2.push_back(data->data.at(best_idx).descr);
                descr_dist.emplace_back(match_dist);
            }
        }
        return !kp2.empty();
    }

    bool FeatureKDTree::getKeypointDescriptorMatches(const cv::Point2f &pt, std::vector<cv::KeyPoint> &kp2, cv::Mat &descr2, const float &searchRadius) const
    {
        std::vector<std::pair<size_t, float>> ret_matches;
        size_t nMatches = radiusSearch(pt, searchRadius, ret_matches);
        if (!nMatches)
        {
            return false;
        }
        descr2.release();
        kp2.clear();
        for (auto &m : ret_matches)
        {
            const size_t idx = m.first;
            kp2.emplace_back(data->data.at(idx).kp);
            descr2.push_back(data->data.at(idx).descr);
        }
        return !kp2.empty();
    }

    bool FeatureKDTree::getKeypointDescriptorMatches(const cv::Point2f &pt, std::vector<cv::KeyPoint> &kp2, const float &searchRadius) const
    {
        std::vector<std::pair<size_t, float>> ret_matches;
        size_t nMatches = radiusSearch(pt, searchRadius, ret_matches);
        if (!nMatches)
        {
            return false;
        }
        kp2.clear();
        for (auto &m : ret_matches)
        {
            const size_t idx = m.first;
            kp2.emplace_back(data->data.at(idx).kp);
        }
        return !kp2.empty();
    }

    bool FeatureKDTree::getIndex(const cv::Point2f &pt, size_t &idx) const
    {
        std::vector<size_t> ret_index(1);
        std::vector<float> out_dist_sqr(1);
        const float query_pt[2] = {pt.x, pt.y};
        size_t num_results = tree->knnSearch(&query_pt[0], 1, &ret_index[0], &out_dist_sqr[0]);
        if (!num_results || out_dist_sqr[0] > single_search_acc)
        {
            return false;
        }
        idx = ret_index[0];
        return true;
    }

    size_t FeatureKDTree::radiusSearch(const cv::Point2f &pt, const float &radius, std::vector<std::pair<size_t, float>> &ret_matches, const bool &sorted) const
    {
        const float query_pt[2] = {pt.x, pt.y};
        nanoflann::SearchParameters params;
        params.sorted = sorted;
        std::vector<nanoflann::ResultItem<size_t, float>> ret_matches2;
        size_t nr = tree->radiusSearch(&query_pt[0], radius, ret_matches2, params);
        ret_matches.clear();
        for (const auto &i: ret_matches2)
        {
            ret_matches.emplace_back(std::make_pair(i.first, i.second));
        }
        
        return nr;
    }
}