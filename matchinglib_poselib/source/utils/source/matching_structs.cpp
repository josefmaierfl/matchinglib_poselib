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

#include <matching_structs.h>
#include <FileHelper.h>

#include <map>

#include <opencv2/imgproc.hpp>

namespace utilslib
{
    double getDescriptorDist(const cv::Mat &descr1, const cv::Mat &descr2)
    {
        if (descr1.type() == CV_8U)
        {
            return norm(descr1, descr2, cv::NORM_HAMMING);
        }
        return norm(descr1, descr2, cv::NORM_L2);
    }

    void MatchData::copyTo(MatchData &data) const
    {
        data.matches = matches;
        data.kps1 = kps1;
        data.kps2 = kps2;
        descr1.copyTo(data.descr1);
        descr2.copyTo(data.descr2);
        inlier_mask.copyTo(data.inlier_mask);
        data.used_cnt = used_cnt;
        data.img_idxs = img_idxs;
    }

    MatchData MatchData::clone() const
    {
        MatchData data;
        copyTo(data);
        return data;
    }

    bool MatchData::check() const
    {
        if (matches.empty() || kps1.empty() || kps2.empty() || descr1.empty() || descr2.empty())
        {
            std::cerr << "Invalid struct MatchData" << std::endl;
            return false;
        }
        if (kps1.size() != static_cast<size_t>(descr1.rows) || kps2.size() != static_cast<size_t>(descr2.rows)){
            std::cerr << "Number of keypoints and descriptors do not match struct MatchData" << std::endl;
            return false;
        }
        if(matches.size() > kps1.size()){
            std::cerr << "Number of keypoints is lower compered to number of matches in struct MatchData" << std::endl;
            return false;
        }
        return true;
    }

    std::size_t MatchSearchBase::KeyHasher::operator()(const cv::Point2f &pt) const
    {
        std::size_t seed = 0;
        int ptx = static_cast<int>(round(pt.x));
        int pty = static_cast<int>(round(pt.y));
        ptx -= ptx % acc;
        pty -= pty % acc;
        hash_combine(seed, ptx);
        hash_combine(seed, pty);
        return seed;
    }

    bool MatchSearchBase::EqualTo::operator()(const cv::Point2f &pt1, const cv::Point2f &pt2) const
    {
        int ptx1 = static_cast<int>(round(pt1.x));
        int pty1 = static_cast<int>(round(pt1.y));
        int ptx2 = static_cast<int>(round(pt2.x));
        int pty2 = static_cast<int>(round(pt2.y));
        ptx1 -= ptx1 % acc;
        pty1 -= pty1 % acc;
        ptx2 -= ptx2 % acc;
        pty2 -= pty2 % acc;
        return nearZero(static_cast<double>(ptx1) - static_cast<double>(ptx2)) &&
               nearZero(static_cast<double>(pty1) - static_cast<double>(pty2));
    }

    MatchInfoBase::MatchInfoBase(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match)
    {
        matches.push_back(match);
        kps1.push_back(kp1);
        kps2.push_back(kp2);
    }

    bool MatchInfoBase::equal(const cv::Point2f &pt1, const cv::Point2f &pt2, const double &acc) const
    {
        return nearZero(static_cast<double>(pt1.x) - static_cast<double>(pt2.x), acc) &&
               nearZero(static_cast<double>(pt1.y) - static_cast<double>(pt2.y), acc);
    }

    size_t MatchInfoBase::contains(const cv::KeyPoint &kp2, const double &acc)
    {
        int i = 0;
        for (auto &k : kps2)
        {
            if (equal(k.pt, kp2.pt, acc))
            {
                return i;
            }
            i++;
        }
        return -1;
    }

    MatchInfo::MatchInfo(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2, const int &img_idx) : MatchInfoBase(kp1, kp2, match)
    {
        CV_Assert(descr1.rows == 1 && descr2.rows == 1);
        descr1_.push_back(descr1);
        descr2_.push_back(descr2);
        img_idxs.push_back(img_idx);
    }

    bool MatchInfo::replace(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2, const int &img_idx)
    {
        int pos = contains(kp2);
        if (pos < 0)
        {
            matches.push_back(match);
            kps1.push_back(kp1);
            kps2.push_back(kp2);
            descr1_.push_back(descr1);
            descr2_.push_back(descr2);
            img_idxs.push_back(img_idx);
            return true;
        }
        else
        {
            cv::KeyPoint &kp1_old = kps1[pos];
            cv::KeyPoint &kp2_old = kps2[pos];
            cv::DMatch &m_old = matches[pos];
            if (nearZero(m_old.distance - match.distance))
            {
                return false;
            }
            float meanResp_o = (kp1_old.response + kp2_old.response) / 2.f;
            float meanResp_n = (kp1.response + kp2.response) / 2.f;
            if (match.distance < m_old.distance)
            {
                if ((1.15f * match.distance < m_old.distance) || (meanResp_n > 1.03f * meanResp_o))
                {
                    kp1_old = kp1;
                    kp2_old = kp2;
                    m_old = match;
                    descr1.copyTo(descr1_.row(pos));
                    descr2.copyTo(descr2_.row(pos));
                    img_idxs.at(pos) = img_idx;
                    return true;
                }
            }
            else if ((0.98f * match.distance < m_old.distance) && (0.8f * meanResp_n > meanResp_o))
            {
                kp1_old = kp1;
                kp2_old = kp2;
                m_old = match;
                descr1.copyTo(descr1_.row(pos));
                descr2.copyTo(descr2_.row(pos));
                img_idxs.at(pos) = img_idx;
                return true;
            }
        }
        return false;
    }

    ReMatchInfo::ReMatchInfo(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2, const double &F_dist, const int &img_idx, const double &acc) : MatchInfoBase(kp1, kp2, match), img2_max_dist(acc)
    {
        CV_Assert(descr1.rows == 1 && descr2.rows == 1);
        descr1_.push_back(descr1);
        descr2_.push_back(descr2);
        epipolar_distance.push_back(F_dist);
        img_idxs.push_back(img_idx);
    }

    bool ReMatchInfo::replace(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::DMatch &match, const cv::Mat &descr1, const cv::Mat &descr2, const double &F_dist, const int &img_idx)
    {
        int pos = contains(kp2, img2_max_dist);
        if (pos < 0)
        {
            matches.push_back(match);
            kps1.push_back(kp1);
            kps2.push_back(kp2);
            descr1_.push_back(descr1);
            descr2_.push_back(descr2);
            epipolar_distance.push_back(F_dist);
            img_idxs.push_back(img_idx);
            return true;
        }
        else
        {
            cv::KeyPoint &kp1_old = kps1[pos];
            cv::KeyPoint &kp2_old = kps2[pos];
            cv::DMatch &m_old = matches[pos];
            double &d_old = epipolar_distance[pos];
            const double diff = F_dist - d_old;
            if (nearZero(m_old.distance - match.distance) && nearZero(diff, 0.01))
            {
                return false;
            }
            if (diff < -0.05 && (match.distance < 1.1f * m_old.distance))
            {
                if (diff < -0.15)
                {
                    kp1_old = kp1;
                    kp2_old = kp2;
                    m_old = match;
                    d_old = F_dist;
                    descr1.copyTo(descr1_.row(pos));
                    descr2.copyTo(descr2_.row(pos));
                    img_idxs.at(pos) = img_idx;
                    return true;
                }
                else
                {
                    float meanResp_o = (kp1_old.response + kp2_old.response) / 2.f;
                    float meanResp_n = (kp1.response + kp2.response) / 2.f;
                    if (match.distance < m_old.distance)
                    {
                        if ((1.15f * match.distance < m_old.distance) || (meanResp_n > 1.03f * meanResp_o))
                        {
                            kp1_old = kp1;
                            kp2_old = kp2;
                            m_old = match;
                            d_old = F_dist;
                            descr1.copyTo(descr1_.row(pos));
                            descr2.copyTo(descr2_.row(pos));
                            img_idxs.at(pos) = img_idx;
                            return true;
                        }
                    }
                    else if ((0.98f * match.distance < m_old.distance) && (0.8f * meanResp_n > meanResp_o))
                    {
                        kp1_old = kp1;
                        kp2_old = kp2;
                        m_old = match;
                        d_old = F_dist;
                        descr1.copyTo(descr1_.row(pos));
                        descr2.copyTo(descr2_.row(pos));
                        img_idxs.at(pos) = img_idx;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    void KeypointSearchSimple::add(const cv::KeyPoint &kp)
    {
        std::unordered_map<cv::Point2f, cv::KeyPoint, KeyHasher, EqualTo>::iterator it;
        it = kpMap.find(kp.pt);
        if (it == kpMap.end())
        {
            kpMap.emplace(kp.pt, kp);
        }
        else if (it->second.response < kp.response)
        {
            it->second = kp;
        }
    }

    void KeypointSearchSimple::composeAll(std::vector<cv::KeyPoint> &kps) const
    {
        for(const auto &kp: kpMap){
            kps.emplace_back(kp.second);
        }
    }

    std::vector<cv::KeyPoint> KeypointSearchSimple::composeAll() const
    {
        std::vector<cv::KeyPoint> kps_tmp;
        for (const auto &kp : kpMap)
        {
            kps_tmp.emplace_back(kp.second);
        }
        return kps_tmp;
    }

    PointSearch::PointSearch(int minPointDistanceXY) : MatchSearchBase(minPointDistanceXY)
    {
        ptSet = std::unordered_set<cv::Point2f, KeyHasher, EqualTo>(101, KeyHasher(accuracy), EqualTo(accuracy));
    }

    void PointSearch::add(const cv::Point2f &pt)
    {
        if (ptSet.find(pt) == ptSet.end())
        {
            ptSet.emplace(pt);
        }
    }

    void PointSearch::add(const cv::KeyPoint &kp)
    {
        add(kp.pt);
    }

    bool PointSearch::contains(const cv::KeyPoint &kp) const
    {
        return ptSet.find(kp.pt) != ptSet.end();
    }

    bool PointSearch::contains(const cv::Point2f &pt) const
    {
        return ptSet.find(pt) != ptSet.end();
    }

    void PointSearch::clear()
    {
        ptSet.clear();
    }

    void PointSearch::copyTo(PointSearch &obj) const
    {
        std::unordered_set<cv::Point2f, KeyHasher, EqualTo> *ptSet2 = static_cast<std::unordered_set<cv::Point2f, KeyHasher, EqualTo> *>(obj.getSetPtr());
        *ptSet2 = ptSet;
    }

    void *PointSearch::getSetPtr()
    {
        return static_cast<void *>(&ptSet);
    }

    bool PointSearch::erase(const cv::Point2f &pt)
    {
        return ptSet.erase(pt) == 1;
    }
}