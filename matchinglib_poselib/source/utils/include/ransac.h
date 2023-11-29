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
#include <map>
#include <random>

#include <opencv2/highgui.hpp>

#include "utilslib/utilslib_api.h"

namespace utilslib
{
    template <typename T>
    void UTILSLIB_API generateUniformRandomSample(const size_t &dataSize, const size_t &sampleSize, std::vector<T> &sample, std::mt19937 &mt)
    {
        size_t count = 0;
        T index = 0;
        typename std::vector<T>::iterator pos;
        if (sample.size() != sampleSize)
        {
            sample.resize(sampleSize);
        }
        pos = sample.begin();
        do
        {
            index = mt() % dataSize;
            if (std::find(sample.begin(), pos, index) == pos)
            {
                sample[count] = index;
                ++count;
                ++pos;
            }
        } while (count < sampleSize);
    }

    size_t UTILSLIB_API updateStandardStoppingRANSAC(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize, const size_t &max_iterations, const double &p_conf_th = 0.98);
    
    class UTILSLIB_API PlaneRANSAC
    {
    public:
        explicit PlaneRANSAC(std::mt19937 &mt_, const cv::Mat Qs_, const double &th_, const size_t max_iterations_ = 100, const double p_conf_th_ = 0.98) : mt(mt_), Qs(Qs_), th(th_), length(static_cast<size_t>(Qs_.rows)), max_iterations(max_iterations_), p_conf_th(p_conf_th_)
        {
            sample.resize(3);
        }
        cv::Mat_<double> compute(std::vector<int> *inliers = nullptr);

    private:
        std::mt19937 &mt;
        const cv::Mat Qs;
        double th;
        const size_t length;
        const size_t max_iterations;
        size_t max_support = 0;
        const double p_conf_th;
        std::vector<int> sample;
        std::multimap<size_t, std::pair<cv::Mat, std::vector<int>>, std::greater<size_t>> hypotheses;

        size_t updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize = 3);
        void getSampleIdxs(const size_t &sampleSize = 3);
        size_t getSupport(const cv::Mat &plane, std::vector<int> &supportSet);
        cv::Mat_<double> planeLeastSquares(const std::vector<int> &supportSet, std::vector<int> &supportSetNew);
        bool checkNotOnLine();
    };

    class UTILSLIB_API PlaneKnownNormalRANSAC
    {
    public:
        explicit PlaneKnownNormalRANSAC(std::mt19937 &mt_, const cv::Mat planeNormal_, const cv::Mat Qs_, const double &th_, const size_t max_iterations_ = 100, const double p_conf_th_ = 0.98) : mt(mt_), planeNormal(planeNormal_), Qs(Qs_), th(th_), length(static_cast<size_t>(Qs_.rows)), max_iterations(max_iterations_), p_conf_th(p_conf_th_)
        {
            sample.resize(1);
        }
        cv::Mat_<double> compute(std::vector<int> *inliers = nullptr);

    private:
        std::mt19937 &mt;
        const cv::Mat planeNormal;
        const cv::Mat Qs;
        double th;
        const size_t length;
        const size_t max_iterations;
        size_t max_support = 0;
        const double p_conf_th;
        std::vector<int> sample;
        std::multimap<size_t, std::pair<cv::Mat, std::vector<int>>, std::greater<size_t>> hypotheses;

        size_t updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize = 1);
        void getSampleIdxs(const size_t &sampleSize = 1);
        size_t getSupport(const cv::Mat &plane, std::vector<int> &supportSet);
        cv::Mat_<double> planeMean(const std::vector<int> &supportSet, std::vector<int> &supportSetNew);
    };

    class UTILSLIB_API CircleRANSAC
    {
    public:
        explicit CircleRANSAC(std::mt19937 &mt_, const std::vector<cv::Point2d> &ps_, const double &th_, const size_t &max_iterations_ = 100, const double &p_conf_th_ = 0.98) : mt(mt_), ps(ps_), th(th_), length(ps_.size()), max_iterations(max_iterations_), p_conf_th(p_conf_th_)
        {
            sample.resize(3);
        }
        cv::Point3d compute(std::vector<size_t> *inliers = nullptr); //Point3d: x=mid.x, y=mid.y, z=radius; z<0 for failed estimation

    private:
        std::mt19937 &mt;
        const std::vector<cv::Point2d> &ps;
        double th;
        const size_t length;
        const size_t max_iterations;
        size_t max_support = 0;
        const double p_conf_th;
        std::vector<size_t> sample;
        std::multimap<size_t, std::pair<cv::Point3d, std::vector<size_t>>, std::greater<size_t>> hypotheses; //Point3d: x=mid.x, y=mid.y, z=radius

        size_t updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize = 3);
        void getSampleIdxs(const size_t &sampleSize = 3);
        size_t getSupport(const cv::Point3d &circle, std::vector<size_t> &supportSet);
        cv::Point3d circleOptimization(const std::vector<size_t> &supportSet, const cv::Point3d &circle, std::vector<size_t> &supportSetNew);
        bool checkNotOnLine();
    };

    class UTILSLIB_API LineRANSAC
    {
    public:
        explicit LineRANSAC(std::mt19937 &mt_, const cv::Mat &planeNormal_, const cv::Mat &pts_, const double &th_, const size_t max_iterations_ = 1000, const double p_conf_th_ = 0.98) : mt(mt_), planeNormal(planeNormal_), pts(pts_), th(th_), length(static_cast<size_t>(pts_.rows)), max_iterations(max_iterations_), p_conf_th(p_conf_th_), sample(0) {}
        cv::Mat_<double> compute(std::vector<int> *inliers = nullptr);

    private:
        std::mt19937 &mt;
        const cv::Mat planeNormal;
        const cv::Mat pts;
        double th;
        const size_t length;
        const size_t max_iterations;
        size_t max_support = 0;
        const double p_conf_th;
        int sample;
        std::multimap<size_t, std::pair<cv::Mat, std::vector<int>>, std::greater<size_t>> hypotheses;

        size_t updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize = 1);
        void getSampleIdx();
        size_t getSupport(const cv::Mat &pt, std::vector<int> &supportSet);
        cv::Mat_<double> ptMean(const std::vector<int> &supportSet, std::vector<int> &supportSetNew);
    };

    class UTILSLIB_API Rigid3DTransformRANSAC
    {
    public:
        explicit Rigid3DTransformRANSAC(std::mt19937 &mt_, const cv::Mat_<double> &points1_, const cv::Mat_<double> &points2_, const double &th_, const size_t max_iterations_ = 100, const double p_conf_th_ = 0.98) : mt(mt_), points1(points1_), points2(points2_), th(th_), length(static_cast<size_t>(points1_.rows)), max_iterations(max_iterations_), p_conf_th(p_conf_th_)
        {
            CV_Assert(points1_.rows == points2_.rows);
            sample.resize(3);
        }
        std::pair<cv::Mat_<double>, double> compute(std::vector<int> *inliers = nullptr);

    private:
        std::mt19937 &mt;
        const cv::Mat_<double> points1;
        const cv::Mat_<double> points2;
        double th;
        const size_t length;
        const size_t max_iterations;
        size_t max_support = 0;
        const double p_conf_th;
        std::vector<int> sample;
        std::multimap<size_t, std::tuple<cv::Mat_<double>, double, std::vector<int>>, std::greater<size_t>> hypotheses;

        size_t updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize = 3);
        void getSampleIdxs(const size_t &sampleSize = 3);
        size_t getSupport(const cv::Mat_<double> &P, const double &scale, std::vector<int> &supportSet);
        std::pair<cv::Mat_<double>, double> transformLeastSquares(const std::vector<int> &supportSet, std::vector<int> &supportSetNew);
    };

    class UTILSLIB_API VectorRANSAC
    {
    public:
        explicit VectorRANSAC(std::mt19937 &mt_, const std::vector<cv::Mat> &vecs_, const double &ang_th_, const bool normalize_, const size_t max_iterations_ = 100, const double p_conf_th_ = 0.99) : mt(mt_), vecs(vecs_), th(1.0 - std::cos(M_PI * ang_th_ / 180.0)), normalize(normalize_), length(vecs_.size()), max_iterations(max_iterations_), p_conf_th(p_conf_th_)
        {
            sample.resize(1);
            if (normalize_){
                for(auto &v : vecs){
                    v /= cv::norm(v);
                }
            }
        }
        cv::Mat compute(std::vector<size_t> *inliers = nullptr);

    private:
        std::mt19937 &mt;
        std::vector<cv::Mat> vecs;        
        double th;
        const bool normalize;
        const size_t length;
        const size_t max_iterations;
        size_t max_support = 0;
        const double p_conf_th;
        std::vector<size_t> sample;
        std::multimap<size_t, std::pair<cv::Mat, std::vector<size_t>>, std::greater<size_t>> hypotheses;

        size_t updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize = 1);
        void getSampleIdxs(const size_t &sampleSize = 1);
        size_t getSupport(const cv::Mat &vec, std::vector<size_t> &supportSet) const;
        cv::Mat vectorCenterMass(const std::vector<size_t> &supportSet, std::vector<size_t> &supportSetNew) const;
        cv::Mat getVectorMean(const cv::Mat &v1, const cv::Mat &v2) const;
        double getAngularDistance(const cv::Mat &v1, const cv::Mat &v2) const;
    };
}