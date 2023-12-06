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
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include "matchinglib/glob_includes.h"

#include "matchinglib/matchinglib_api.h"

namespace matchinglib
{
    double MATCHINGLIB_API getDescriptorDist(const cv::Mat &descr1, const cv::Mat &descr2);

    struct MATCHINGLIB_API MatchData
    {
        std::vector<cv::DMatch> matches;
        std::vector<cv::KeyPoint> kps1, kps2;
        cv::Mat descr1, descr2;
        cv::Mat inlier_mask;
        int used_cnt = 0;

        void copyTo(MatchData &data) const;
        MatchData clone() const;
        bool check() const;
    };

    typedef ptr_wrapper<std::unordered_map<int, cv::Mat>> ImgDataPtr;

    struct MatchDataCams
    {
        std::unordered_map<int, cv::Mat> masks;
        std::unordered_map<int, cv::Mat> images;
        std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> features;
        std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> matches;
        std::vector<std::pair<int, int>> cam_pair_indices;
        int nr_cameras = -1;
        double imgScale = 0.5;

        MatchDataCams() = default;
        MatchDataCams(const std::unordered_map<int, std::pair<cv::Mat, cv::Mat>> &imageMap, 
                      const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &features_, 
                      const std::unordered_map<std::pair<int, int>, MatchData, pair_hash, pair_EqualTo> &matches_, 
                      const std::vector<std::pair<int, int>> &cam_pair_indices_, 
                      const int &nr_cameras_, 
                      const double &imgScale_);
        ImgDataPtr getImg_ptr() { return &images; }
        ImgDataPtr getMask_ptr() { return &masks; }
        std::shared_ptr<MatchDataCams> copyToSharedPtr() const;
    };

    class MATCHINGLIB_API MatchSearchBase
    {
    public:
        explicit MatchSearchBase(int minPointDistanceXY) : accuracy(minPointDistanceXY) {}

    protected:
        int accuracy;

        struct KeyHasher
        {
            int acc = 2;

            KeyHasher(const int accuracy = 2)
            {
                acc = accuracy;
            }

            std::size_t operator()(const cv::Point2f &pt) const;
        };

        struct EqualTo
        {
            int acc = 2;

            EqualTo(const int accuracy = 2)
            {
                acc = accuracy;
            }

            bool operator()(const cv::Point2f &pt1, const cv::Point2f &pt2) const;
        };
    };

    class MATCHINGLIB_API MatchInfoBase
    {
    public:
        std::vector<cv::DMatch> matches;
        std::vector<cv::KeyPoint> kps1, kps2;

        MatchInfoBase(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match);
        bool equal(const cv::Point2f &pt1, const cv::Point2f &pt2, const double &acc = 2.0) const;
        size_t contains(const cv::KeyPoint &kp2, const double &acc = 2.0);
    };

    class MATCHINGLIB_API MatchInfo : public MatchInfoBase
    {
    public:
        cv::Mat descr1_, descr2_;

        MatchInfo(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2);
        bool replace(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2);
    };

    class MATCHINGLIB_API ReMatchInfo : public MatchInfoBase
    {
    public:
        std::vector<double> epipolar_distance;
        cv::Mat descr1_, descr2_;
        const double img2_max_dist;

        ReMatchInfo(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2, const double &F_dist, const double &acc = 2.0);
        bool replace(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2, const double &F_dist);
    };

    template <typename T>
    class MATCHINGLIB_API MatchSearchTemplate : private MatchSearchBase
    {
    public:
        explicit MatchSearchTemplate(int minPointDistanceXY = 2) : MatchSearchBase(minPointDistanceXY)
        {
            kpMap = std::unordered_map<cv::Point2f, T, KeyHasher, EqualTo>(101, KeyHasher(accuracy), EqualTo(accuracy));
        }

        template <typename... Args>
        bool addMatch(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, Args... more_pars)
        {
            if (kpMap.find(kp1.pt) == kpMap.end())
            {
                kpMap.emplace(kp1.pt, T(kp1, kp2, match, more_pars...));
                return true;
            }
            T &m_ref = kpMap.at(kp1.pt);
            return m_ref.replace(kp1, kp2, match, more_pars...);
        }

        void composeAll(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2)
        {
            size_t idx = 0;
            for (auto &ms : kpMap)
            {
                size_t nr_maches = ms.second.matches.size();
                for (size_t i = 0; i < nr_maches; ++i)
                {
                    cv::DMatch m = ms.second.matches[i];
                    m.queryIdx = m.trainIdx = idx;
                    matches.emplace_back(std::move(m));
                    kps1.push_back(ms.second.kps1[i]);
                    kps2.push_back(ms.second.kps2[i]);
                    idx++;
                }
            }
        }

        size_t size()
        {
            return kpMap.size();
        }

        bool empty()
        {
            return kpMap.empty();
        }

    protected:
        std::unordered_map<cv::Point2f, T, KeyHasher, EqualTo> kpMap;
    };

    class MATCHINGLIB_API MatchSearch : public MatchSearchTemplate<MatchInfo>
    {
    public:
        explicit MatchSearch(int minPointDistanceXY = 2) : MatchSearchTemplate(minPointDistanceXY) {}

        void composeAll(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2, cv::Mat &descr1, cv::Mat &descr2)
        {
            size_t idx = 0;
            for (auto &ms : kpMap)
            {
                size_t nr_maches = ms.second.matches.size();
                for (size_t i = 0; i < nr_maches; ++i)
                {
                    cv::DMatch m = ms.second.matches[i];
                    m.queryIdx = m.trainIdx = idx;
                    matches.emplace_back(std::move(m));
                    kps1.push_back(ms.second.kps1[i]);
                    kps2.push_back(ms.second.kps2[i]);
                    idx++;
                }
                descr1.push_back(ms.second.descr1_);
                descr2.push_back(ms.second.descr2_);
            }
        }
    };

    class MATCHINGLIB_API ReMatchSearch : public MatchSearchTemplate<ReMatchInfo>
    {
    public:
        explicit ReMatchSearch(int minPointDistanceXY = 2) : MatchSearchTemplate(minPointDistanceXY) {}

        void composeAll(const size_t &start_idx, std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &kps1, std::vector<cv::KeyPoint> &kps2, cv::Mat &descr1, cv::Mat &descr2, std::vector<double> &sampson_errors)
        {
            size_t idx = start_idx;
            for (auto &ms : kpMap)
            {
                size_t nr_maches = ms.second.matches.size();
                for (size_t i = 0; i < nr_maches; ++i)
                {
                    cv::DMatch m = ms.second.matches[i];
                    m.queryIdx = m.trainIdx = idx;
                    matches.emplace_back(std::move(m));
                    kps1.push_back(ms.second.kps1[i]);
                    kps2.push_back(ms.second.kps2[i]);
                    sampson_errors.push_back(ms.second.epipolar_distance[i]);
                    idx++;
                }
                descr1.push_back(ms.second.descr1_);
                descr2.push_back(ms.second.descr2_);
            }
        }
    };

    template <typename T>
    class MATCHINGLIB_API KeypointSearchBase : protected MatchSearchBase
    {
    public:
        explicit KeypointSearchBase(int minPointDistanceXY = 1) : MatchSearchBase(minPointDistanceXY)
        {
            kpMap = std::unordered_map<cv::Point2f, T, KeyHasher, EqualTo>(101, KeyHasher(accuracy), EqualTo(accuracy));
        }

        bool add(const cv::Point2f &pt, const T &data)
        {
            typename std::unordered_map<cv::Point2f, T, KeyHasher, EqualTo>::const_iterator it;
            it = kpMap.find(pt);
            if (it == kpMap.end())
            {
                kpMap.emplace(pt, data);
                return true;
            }
            return false;
        }

        bool contains(const cv::KeyPoint &kp) const
        {
            return kpMap.find(kp.pt) != kpMap.end();
        }

        bool contains(const cv::Point2f &pt) const
        {
            return kpMap.find(pt) != kpMap.end();
        }

        size_t size() const
        {
            return kpMap.size();
        }

        bool empty() const
        {
            return kpMap.empty();
        }

        void clear()
        {
            kpMap.clear();
        }

        bool erase(const cv::Point2f &pt)
        {
            if (!contains(pt))
            {
                return false;
            }
            kpMap.erase(pt);
            return true;
        }

    protected:
        std::unordered_map<cv::Point2f, T, KeyHasher, EqualTo> kpMap;
    };

    class MATCHINGLIB_API KeypointSearch : public KeypointSearchBase<std::pair<cv::KeyPoint, cv::Mat>>
    {
    public:
        explicit KeypointSearch(int minPointDistanceXY = 1) : KeypointSearchBase(minPointDistanceXY) {}

        void add(const cv::KeyPoint &kp, const cv::Mat &descr)
        {
            CV_Assert(descr.rows == 1);
            std::unordered_map<cv::Point2f, std::pair<cv::KeyPoint, cv::Mat>, KeyHasher, EqualTo>::iterator it;
            it = kpMap.find(kp.pt);
            if (it == kpMap.end())
            {
                kpMap.emplace(kp.pt, std::make_pair(kp, descr.clone()));
            }
            else if (it->second.first.response < kp.response)
            {
                it->second = std::make_pair(kp, descr.clone());
            }
        }

        bool getDescriptor(const cv::Point2f &pt, cv::Mat &descr) const
        {
            std::unordered_map<cv::Point2f, std::pair<cv::KeyPoint, cv::Mat>, KeyHasher, EqualTo>::const_iterator it;
            it = kpMap.find(pt);
            if (it != kpMap.end())
            {
                it->second.second.copyTo(descr);
                return true;
            }
            return false;
        }

        bool getDescriptor(const cv::KeyPoint &kp, cv::Mat &descr) const
        {
            return getDescriptor(kp.pt, descr);
        }

        bool getDescriptorRef(const cv::Point2f &pt, cv::Mat &descr) const
        {
            std::unordered_map<cv::Point2f, std::pair<cv::KeyPoint, cv::Mat>, KeyHasher, EqualTo>::const_iterator it;
            it = kpMap.find(pt);
            if (it != kpMap.end())
            {
                descr = it->second.second;
                return true;
            }
            return false;
        }

        bool getDescriptorRef(const cv::KeyPoint &kp, cv::Mat &descr) const
        {
            return getDescriptorRef(kp.pt, descr);
        }

        bool getDescriptorDistance(const cv::Point2f &pt, const cv::Mat &descr2, double &distance) const
        {
            cv::Mat descr1;
            if (!getDescriptorRef(pt, descr1))
            {
                return false;
            }
            distance = getDescriptorDist(descr1, descr2);
            return true;
        }

        bool getDescriptorDistance(const cv::KeyPoint &kp, const cv::Mat &descr2, double &distance) const
        {
            return getDescriptorDistance(kp.pt, descr2, distance);
        }

        bool getKeypoint(const cv::Point2f &pt, cv::KeyPoint &kp) const
        {
            std::unordered_map<cv::Point2f, std::pair<cv::KeyPoint, cv::Mat>, KeyHasher, EqualTo>::const_iterator it;
            it = kpMap.find(pt);
            if (it != kpMap.end())
            {
                kp = it->second.first;
                return true;
            }
            return false;
        }

        bool getKeypointDescriptor(const cv::Point2f &pt, cv::KeyPoint &kp, cv::Mat &descr) const
        {
            std::unordered_map<cv::Point2f, std::pair<cv::KeyPoint, cv::Mat>, KeyHasher, EqualTo>::const_iterator it;
            it = kpMap.find(pt);
            if (it != kpMap.end())
            {
                kp = it->second.first;
                it->second.second.copyTo(descr);
                return true;
            }
            return false;
        }

        void composeAll(std::vector<cv::KeyPoint> &kps, cv::Mat &descr)
        {
            kps.clear();
            descr.release();
            for (auto &ms : kpMap)
            {
                descr.push_back(ms.second.second);
                kps.emplace_back(ms.second.first);
            }
        }
    };

    class MATCHINGLIB_API KeypointSearchSimple : public KeypointSearchBase<cv::KeyPoint>
    {
    public:
        explicit KeypointSearchSimple(int minPointDistanceXY = 1) : KeypointSearchBase(minPointDistanceXY) {}

        void add(const cv::KeyPoint &kp);
        void composeAll(std::vector<cv::KeyPoint> &kps) const;
        std::vector<cv::KeyPoint> composeAll() const;
    };

    class MATCHINGLIB_API PointSearchSimpleQ : public KeypointSearchBase<int>
    {
    public:
        explicit PointSearchSimpleQ(int minPointDistanceXY = 1) : KeypointSearchBase(minPointDistanceXY), first_it(true) {}

        bool getIdx(const cv::Point2f &pt, int &idx) const
        {
            if(!contains(pt)){
                return false;
            }
            idx = kpMap.at(pt);
            return true;
        }

        int getIdx(const cv::Point2f &pt) const
        {
            if (!contains(pt))
            {
                return -1;
            }
            return kpMap.at(pt);
        }

        void updateIdx(const cv::Point2f &pt, const int &idx)
        {
            if (!contains(pt))
            {
                std::cerr << "Point not in map. Discarding update." << std::endl;
                return;
            }
            kpMap.at(pt) = idx;
        }

        void resetIterator()
        {
            kp_it = kpMap.begin();
        }

        bool next(int &idx)
        {
            if (first_it)
            {
                kp_it = kpMap.begin();
                first_it = false;
            }
            if (kp_it == kpMap.end())
            {
                return false;
            }
            idx = kp_it->second;
            kp_it++;
            return true;
        }

        int next()
        {
            if (first_it)
            {
                kp_it = kpMap.begin();
                first_it = false;
            }
            if (kp_it == kpMap.end())
            {
                return -1;
            }
            const int idx = kp_it->second;
            kp_it++;
            return idx;
        }

        void writeBinary(std::ofstream &resultsToFile) const
        {
            size_t nr_values = kpMap.size();
            resultsToFile.write((char *)&nr_values, sizeof(size_t));
            for (const auto &m : kpMap)
            {
                resultsToFile.write((char *)&m.first, sizeof(m.first));
                resultsToFile.write((char *)&m.second, sizeof(int));
            }
        }

        void readBinary(std::ifstream &resultsFromFile)
        {
            size_t nr_values;
            resultsFromFile.read((char *)&nr_values, sizeof(size_t));
            kpMap.clear();
            for (size_t i = 0; i < nr_values; i++)
            {
                cv::Point2f pt;
                resultsFromFile.read((char *)&pt, sizeof(cv::Point2f));
                int idx;
                resultsFromFile.read((char *)&idx, sizeof(int));
                kpMap.emplace(pt, std::move(idx));
            }
        }

    private:
        std::unordered_map<cv::Point2f, int, KeyHasher, EqualTo>::const_iterator kp_it;
        bool first_it = true;
    };

    class MATCHINGLIB_API PointSearch : private MatchSearchBase
    {
    public:
        explicit PointSearch(int minPointDistanceXY = 1);

        void add(const cv::Point2f &pt);
        void add(const cv::KeyPoint &kp);
        bool contains(const cv::KeyPoint &kp) const;
        bool contains(const cv::Point2f &pt) const;
        void clear();
        void copyTo(PointSearch &obj) const;
        void *getSetPtr();
        bool erase(const cv::Point2f &pt);

    private:
        std::unordered_set<cv::Point2f, KeyHasher, EqualTo> ptSet;
    };
}