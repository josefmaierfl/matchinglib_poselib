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

#include <trees.h>
#include <matching_structs.h>
#include <utils_cv.h>
#include "opencv2/imgproc.hpp"

namespace utilslib
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
        nanoflann::SearchParams params;
        params.sorted = sorted;
        return tree->radiusSearch(&query_pt[0], radius, ret_matches, params);
    }

    LineSearchNode::LineSearchNode(const Circle &innerCircle, const size_t &max_leaf_entries) : innerCircle_(innerCircle), max_leaf_entries_(innerCircle.radius > 0.1 ? max_leaf_entries : 1000), hasChilds(false)
    {
        computeProperties();
    }

    LineSearchNode::LineSearchNode(const cv::KeyPoint &kp, const cv::Mat &descr, const Circle &innerCircle, const size_t &max_leaf_entries) : innerCircle_(innerCircle), max_leaf_entries_(innerCircle.radius > 0.1 ? max_leaf_entries : 1000), hasChilds(false)
    {
        computeProperties();
        add(kp, descr);
    }

    void LineSearchNode::computeProperties()
    {
        outerCircle_radius = M_SQRT2 * innerCircle_.radius;
        x_min = innerCircle_.midpoint.x - innerCircle_.radius;
        x_max = innerCircle_.midpoint.x + innerCircle_.radius;
        y_min = innerCircle_.midpoint.y - innerCircle_.radius;
        y_max = innerCircle_.midpoint.y + innerCircle_.radius;
        l_t = (cv::Mat_<double>(3, 1) << 0, (-1.0 / y_max), 1.0); // Top horizontal line
        l_b = (cv::Mat_<double>(3, 1) << 0, (-1.0 / y_min), 1.0); // Bottom horizontal line
        l_l = (cv::Mat_<double>(3, 1) << (-1.0 / x_min), 0, 1.0); // Left vertical line
        l_r = (cv::Mat_<double>(3, 1) << (-1.0 / x_max), 0, 1.0); // Right vertical line
    }

    void LineSearchNode::add(const cv::KeyPoint &kp, const cv::Mat &descr)
    {
        if (!hasChilds && entries.size() < max_leaf_entries_)
        {
            entries.emplace_back(std::make_pair(kp, descr.clone()));
        }
        else
        {
            addToChild(kp, descr);
            if (entries.size() >= max_leaf_entries_)
            {
                cv::KeyPoint *e1 = nullptr;
                for (auto &e : entries)
                {
                    bool keep = true;
                    if (e1)
                    {
                        const cv::Point2f d = e.first.pt - e1->pt;
                        if (nearZero(d.x) && nearZero(d.y)){
                            const float respd = e.first.response - e1->response;
                            const float sid = e.first.size - e1->size;
                            if (nearZero(respd) && nearZero(sid))
                            {
                                keep = false;
                            }
                        }
                    }
                    if (keep){
                        addToChild(e.first, e.second);
                    }
                    e1 = &e.first;
                }
                entries.clear();
                hasChilds = true;
            }
        }
    }

    bool LineSearchNode::containsLine(const cv::Mat &line, const double &distance, const double &ab_length) const
    {
        const double dist = getLineDistance(line, innerCircle_.midpoint, ab_length);
        if (dist > outerCircle_radius + distance)
        {
            return false;
        }
        if (dist <= innerCircle_.radius + distance)
        {
            return true;
        }
        if (dist <= outerCircle_radius && dist > innerCircle_.radius)
        {
            return lineCrossesRectangle(line);
        }
        const double d = dist - innerCircle_.radius;
        cv::Mat line1, line2;
        getParallelLines(line, ab_length, d, line1, line2);
        bool contains = lineCrossesRectangle(line1);
        contains |= lineCrossesRectangle(line2);
        return contains;
    }

    bool LineSearchNode::getPointsNearLine(const cv::Mat &line, std::vector<std::pair<cv::KeyPoint, cv::Mat>> &found_entries, const double &distance, const double &ab_length) const
    {
        double ab_length1;
        if (nearZero(ab_length))
        {
            ab_length1 = std::sqrt(line.at<double>(0) * line.at<double>(0) + line.at<double>(1) * line.at<double>(1));
        }
        else
        {
            ab_length1 = ab_length;
        }
        if (!containsLine(line, distance, ab_length1))
        {
            return false;
        }
        bool found = false;
        if (!hasChilds)
        {
            for (auto &e : entries)
            {
                if (getLineDistance(line, e.first.pt, ab_length1) < distance)
                {
                    found = true;
                    found_entries.emplace_back(std::make_pair(e.first, e.second.clone()));
                }
            }
            return found;
        }

        for (auto &child : childs)
        {
            if (child)
            {
                found |= child->getPointsNearLine(line, found_entries, distance, ab_length1);
            }
        }
        return found;
    }

    void LineSearchNode::addToChild(const cv::KeyPoint &kp, const cv::Mat &descr)
    {
        unsigned char pos = (kp.pt.y < innerCircle_.midpoint.y) ? static_cast<unsigned char>(2) : 0;
        pos |= (kp.pt.x > innerCircle_.midpoint.x) ? static_cast<unsigned char>(1) : 0;
        if (childs[pos])
        {
            childs[pos]->add(kp, descr);
        }
        else
        {
            createNewNode(kp, descr, pos);
        }
    }

    void LineSearchNode::createNewNode(const cv::KeyPoint &kp, const cv::Mat &descr, const unsigned char &pos)
    {
        cv::Point2d newMid;
        double newRadius = innerCircle_.radius / 2.0;
        switch (pos)
        {
        case 0:
            newMid.x = innerCircle_.midpoint.x - newRadius;
            newMid.y = innerCircle_.midpoint.y + newRadius;
            break;
        case 1:
            newMid.x = innerCircle_.midpoint.x + newRadius;
            newMid.y = innerCircle_.midpoint.y + newRadius;
            break;
        case 2:
            newMid.x = innerCircle_.midpoint.x - newRadius;
            newMid.y = innerCircle_.midpoint.y - newRadius;
            break;
        case 3:
            newMid.x = innerCircle_.midpoint.x + newRadius;
            newMid.y = innerCircle_.midpoint.y - newRadius;
            break;
        default:
            throw std::runtime_error("Unknown tree node");
        }
        childs[pos] = std::make_unique<LineSearchNode>(kp, descr, Circle(newMid, newRadius), max_leaf_entries_);
    }

    bool LineSearchNode::lineCrossesRectangle(const cv::Mat &line) const
    {
        cv::Mat p;
        double pos;
        if (!linesParallel(l_t, line))
        {
            p = l_t.cross(line);
            pos = p.at<double>(0) / p.at<double>(2);
            if (pos < x_min || pos > x_max)
            {
                p = l_b.cross(line);
                pos = p.at<double>(0) / p.at<double>(2);
                if (pos < x_min || pos > x_max)
                {
                    return false;
                }
            }
        }

        if (!linesParallel(l_l, line))
        {
            p = l_l.cross(line);
            pos = p.at<double>(1) / p.at<double>(2);
            if (pos < y_min || pos > y_max)
            {
                p = l_r.cross(line);
                pos = p.at<double>(1) / p.at<double>(2);
                if (pos < y_min || pos > y_max)
                {
                    return false;
                }
            }
        }

        return true;
    }

    double LineSearchNode::getLineDistance(const cv::Mat &line, const cv::Point2d &pt, const double &ab_length) const
    {
        const double a = std::abs(line.at<double>(0) * pt.x + line.at<double>(1) * pt.y + line.at<double>(2));
        return a / ab_length;
    }

    double LineSearchNode::getLineDistance(const cv::Mat &line, const cv::Point2f &pt, const double &ab_length) const
    {
        return getLineDistance(line, cv::Point2d(static_cast<double>(pt.x), static_cast<double>(pt.y)), ab_length);
    }

    bool LineSearchNode::linesParallel(const cv::Mat &line1, const cv::Mat &line2) const
    {
        const double r1 = line1.at<double>(0) / line1.at<double>(1);
        const double r2 = line2.at<double>(0) / line2.at<double>(1);
        if (nearZero(r1 - r2))
        {
            return true;
        }
        return false;
    }

    void LineSearchNode::getParallelLines(const cv::Mat &line_in, const double &ab_length, const double &distance, cv::Mat &line1_out, cv::Mat &line2_out) const
    {
        const double c_add = distance * ab_length;
        line_in.copyTo(line1_out);
        line1_out.at<double>(2) += c_add;
        line_in.copyTo(line2_out);
        line2_out.at<double>(2) -= c_add;
    }

    Circle LineSearchTree::getOuterCircle(const std::vector<cv::KeyPoint> &keypoints)
    {
        std::vector<cv::Point2f> positions;
        for (auto &kp : keypoints)
        {
            positions.push_back(kp.pt);
        }
        cv::Point2f mid;
        float radius;
        cv::minEnclosingCircle(positions, mid, radius);
        return Circle(mid, radius);
    }

    void LineSearchTree::defineOuterCircle(const std::vector<cv::KeyPoint> &keypoints)
    {
        outer_circle = getOuterCircle(keypoints);
    }

    void LineSearchTree::buildTree(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors)
    {
        if (outer_circle.radius <= 0)
        {
            throw std::runtime_error("Initial circle for line search tree must be defined!");
        }
        if (tree)
        {
            tree.reset(new LineSearchNode(outer_circle, nr_leaf));
        }
        else
        {
            tree = std::make_shared<LineSearchNode>(outer_circle, nr_leaf);
        }
        for (int i = 0; i < descriptors.rows; ++i)
        {
            tree->add(keypoints.at(i), descriptors.row(i));
        }
    }

    bool LineSearchTree::getDataNearLine(const cv::Mat &line, const double &distance, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const
    {
        std::vector<std::pair<cv::KeyPoint, cv::Mat>> found_entries;
        bool success = tree->getPointsNearLine(line, found_entries, distance);
        if (success)
        {
            for (auto &i : found_entries)
            {
                keypoints.push_back(i.first);
                descriptors.push_back(i.second);
            }
        }
        return success;
    }

    bool LineSearchTree::getBestMatch(const cv::Mat &x1, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::Mat &descriptor_match, cv::DMatch &match) const
    {
        CV_Assert(x1.rows == 3 && x1.cols == 1);
        cv::Mat l = F * x1;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        int best_i = -1;
        double min_descr_dist = DBL_MAX;
        if (!getDataNearLine(l, th, keypoints, descriptors))
        {
            return false;
        }
        for (int i = 0; i < descriptors.rows; ++i)
        {
            double descr_dist = getDescriptorDist(descriptor1, descriptors.row(i));
            if (descr_dist > max_descr_dist)
            {
                continue;
            }
            if (descr_dist < min_descr_dist)
            {
                min_descr_dist = descr_dist;
                best_i = i;
            }
        }
        if (best_i < 0)
        {
            return false;
        }
        descriptors.row(best_i).copyTo(descriptor_match);
        kp_match = keypoints[best_i];
        match.distance = static_cast<float>(min_descr_dist);
        return true;
    }

    bool LineSearchTree::getBestMatch(const cv::Point2f &pt, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::Mat &descriptor_match, cv::DMatch &match) const
    {
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << static_cast<double>(pt.x), static_cast<double>(pt.y), 1.0);
        return getBestMatch(x1, F, th, descriptor1, max_descr_dist, kp_match, descriptor_match, match);
    }

    bool LineSearchTree::getBestMatchSampson(const cv::Mat &x1, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::OutputArray descriptor_match, cv::DMatch *match) const
    {
        CV_Assert(x1.rows == 3 && x1.cols == 1);
        cv::Mat l = F * x1;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        if (!getDataNearLine(l, th, keypoints, descriptors))
        {
            return false;
        }

        std::vector<double> descr_dists;
        std::vector<int> indices;
        for (int i = 0; i < descriptors.rows; ++i)
        {
            double descr_dist = getDescriptorDist(descriptor1, descriptors.row(i));            
            if (descr_dist > max_descr_dist)
            {
                continue;
            }
            descr_dists.emplace_back(descr_dist);
            indices.emplace_back(i);
        }
        if (indices.empty())
        {
            return false;
        }
        if(indices.size() == 1){
            if (descriptor_match.needed())
            {
                if (descriptor_match.empty())
                {
                    descriptor_match.create(1, descriptors.cols, descriptors.type());
                }
                cv::Mat descriptor_match_ = descriptor_match.getMat();
                descriptors.row(indices[0]).copyTo(descriptor_match_);
            }
            kp_match = keypoints[indices[0]];
            if (match)
            {
                match->distance = static_cast<float>(descr_dists[0]);
            }
            return true;
        }

        std::vector<double> sampson;
        cv::Point2f pt1(static_cast<float>(x1.at<double>(0)), static_cast<float>(x1.at<double>(1)));
        for (const auto &i : indices)
        {
            sampson.emplace_back(calculateSampsonError(pt1, keypoints.at(i).pt, F));
        }
        const double sampson_max = std::max(*std::max_element(sampson.begin(), sampson.end()), 1e-3);
        const double dd_max = std::max(*std::max_element(descr_dists.begin(), descr_dists.end()), 1e-3);

        if(sampson_max > 10.0)
        {
            for (auto &s : sampson)
            {
                s = std::max(1.0, s);
            }
        }
        else if (sampson_max > 5.0)
        {
            for (auto &s : sampson)
            {
                s = std::max(0.5, s);
            }
        }
        else if (sampson_max > 2.0)
        {
            for (auto &s : sampson)
            {
                s = std::max(0.2, s);
            }
        }

        std::vector<double> weights;
        for (size_t i = 0; i < indices.size(); i++)
        {
            const double ddm = descr_dists.at(i) / dd_max;
            const double sm = sampson.at(i) / sampson_max;
            weights.emplace_back(ddm * sm);
        }
        const auto best_i = std::distance(weights.begin(), std::min_element(weights.begin(), weights.end()));
        const int best_ii = indices.at(best_i);

        if (descriptor_match.needed()){
            if (descriptor_match.empty()){
                descriptor_match.create(1, descriptors.cols, descriptors.type());
            }
            cv::Mat descriptor_match_ = descriptor_match.getMat();
            descriptors.row(best_ii).copyTo(descriptor_match_);
        }
        kp_match = keypoints.at(best_ii);
        if (match){
            match->distance = static_cast<float>(descr_dists.at(best_i));
        }
        return true;
    }

    bool LineSearchTree::getBestMatchSampson(const cv::Mat &x1, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::Point2f &pt_match, cv::OutputArray descriptor_match, cv::DMatch *match) const
    {
        cv::KeyPoint kp_match;
        bool res = getBestMatchSampson(x1, F, th, descriptor1, max_descr_dist, kp_match, descriptor_match, match);
        if(res){
            pt_match = kp_match.pt;
        }
        return res;
    }

    bool LineSearchTree::getBestMatchSampson(const cv::Point2f &pt, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::KeyPoint &kp_match, cv::OutputArray descriptor_match, cv::DMatch *match) const
    {
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << static_cast<double>(pt.x), static_cast<double>(pt.y), 1.0);
        return getBestMatchSampson(x1, F, th, descriptor1, max_descr_dist, kp_match, descriptor_match, match);
    }

    bool LineSearchTree::getBestMatchSampson(const cv::Point2f &pt, const cv::Mat &F, const double &th, const cv::Mat &descriptor1, const double &max_descr_dist, cv::Point2f &pt_match, cv::OutputArray descriptor_match, cv::DMatch *match) const
    {
        cv::KeyPoint kp_match;
        bool res = getBestMatchSampson(pt, F, th, descriptor1, max_descr_dist, kp_match, descriptor_match, match);
        if (res)
        {
            pt_match = kp_match.pt;
        }
        return res;
    }
}