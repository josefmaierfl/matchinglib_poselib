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

#include <ransac.h>

#include <utils_cv.h>
#include <utils_common.h>

using namespace std;

namespace utilslib
{
    size_t updateStandardStoppingRANSAC(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize, const size_t &max_iterations, const double &p_conf_th)
    {
        double n_inliers = 1.0;
        double n_pts = 1.0;

        for (unsigned int i = 0; i < sampleSize; ++i)
        {
            n_inliers *= numInliers - i;
            n_pts *= totPoints - i;
        }
        double prob_good_model = n_inliers / n_pts;

        if (prob_good_model < std::numeric_limits<double>::epsilon())
        {
            return max_iterations;
        }
        else if (1.0 - prob_good_model < std::numeric_limits<double>::epsilon())
        {
            return 1;
        }
        else
        {
            double nusample_s = log(1.0 - p_conf_th) / log(1.0 - prob_good_model);
            return static_cast<size_t>(ceil(nusample_s));
        }
    }

    cv::Mat_<double> PlaneRANSAC::compute(std::vector<int> *inliers)
    {
        size_t iters = max_iterations;
        size_t cnt = 0, cnt2 = 0;
        while (cnt < iters && cnt < max_iterations)
        {
            //Get sample
            getSampleIdxs();

            //Check if all points are on a line
            if (!checkNotOnLine())
            {
                cnt++;
                continue;
            }

            //Estimate plane
            cv::Mat plane = calculatePlane(Qs.row(sample.at(0)), Qs.row(sample.at(1)), Qs.row(sample.at(2)));

            //Get support set
            std::vector<int> supportSet;
            const size_t support = getSupport(plane, supportSet);
            hypotheses.emplace(support, make_pair(plane, move(supportSet)));

            if (support > max_support)
            {
                //Update stopping
                iters = updateStandardStopping(support, length);
                max_support = support;
            }

            if (length > 5 && cnt >= max_iterations / 2UL && hypotheses.begin()->first < 4 && cnt2 < 3)
            {
                //Alter threshold and start over
                th *= 1.25;
                cnt2++;
                cnt = 0;
                iters = max_iterations;
                hypotheses.clear();
                continue;
            }
            cnt++;
        }

        //Get best model
        if (hypotheses.empty())
        {
            return cv::Mat();
        }
        std::multimap<size_t, std::pair<cv::Mat, std::vector<int>>, std::greater<size_t>>::const_iterator it = hypotheses.begin();
        if (it->first == 3)
        {
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        std::vector<int> supportSet2;
        cv::Mat plane = planeLeastSquares(it->second.second, supportSet2);
        if (plane.empty())
        {
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        if (inliers)
        {
            *inliers = supportSet2;
        }
        return plane;
    }

    size_t PlaneRANSAC::updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize)
    {
        return updateStandardStoppingRANSAC(numInliers, totPoints, sampleSize, max_iterations, p_conf_th);
    }

    void PlaneRANSAC::getSampleIdxs(const size_t &sampleSize)
    {
        generateUniformRandomSample(length, sampleSize, sample, mt);
    }

    size_t PlaneRANSAC::getSupport(const cv::Mat &plane, std::vector<int> &supportSet)
    {
        for (size_t i = 0; i < length; i++)
        {
            const double d = distanceToPlane(Qs.row(i), plane);
            if (d < th)
            {
                supportSet.emplace_back(i);
            }
        }
        return supportSet.size();
    }

    cv::Mat_<double> PlaneRANSAC::planeLeastSquares(const std::vector<int> &supportSet, std::vector<int> &supportSetNew)
    {
        cv::Mat A = cv::Mat::zeros(3, 3, CV_64FC1);
        cv::Mat b = cv::Mat::zeros(3, 1, CV_64FC1);
        const size_t support = supportSet.size();
        A.at<double>(2, 2) = static_cast<double>(support);
        for (size_t i = 0; i < support; i++)
        {
            const int &iq = supportSet[i];
            const double &x = Qs.at<double>(iq, 0);
            const double &y = Qs.at<double>(iq, 1);
            const double &z = Qs.at<double>(iq, 2);
            const double xy = x * y;

            A.at<double>(0, 0) += x * x;
            A.at<double>(0, 1) += xy;
            A.at<double>(0, 2) += x;
            A.at<double>(1, 0) += xy;
            A.at<double>(1, 1) += y * y;
            A.at<double>(1, 2) += y;
            A.at<double>(2, 0) += x;
            A.at<double>(2, 1) += y;

            b.at<double>(0) += x * z;
            b.at<double>(1) += y * z;
            b.at<double>(0) += z;
        }

        cv::Mat plane3;
        if (!cv::solve(A, b, plane3))
        {
            return cv::Mat();
        }
        cv::Mat plane = (cv::Mat_<double>(3, 1) << plane3.at<double>(0), plane3.at<double>(1), -1.0);
        const double nn = cv::norm(plane);
        plane /= nn;
        plane.resize(4);
        plane.at<double>(3) = plane3.at<double>(2) / nn;

        std::vector<int> supportSet2, supportSet3;
        size_t support2 = getSupport(plane, supportSet2);
        if (support2 < support)
        {
            return cv::Mat();
        }
        else if (support2 > support)
        {
            cv::Mat plane2 = planeLeastSquares(supportSet2, supportSet3);
            if (plane2.empty())
            {
                supportSetNew = supportSet2;
                return plane;
            }
            supportSetNew = supportSet3;
            return plane2;
        }
        supportSetNew = supportSet2;
        return plane;
    }

    bool PlaneRANSAC::checkNotOnLine()
    {
        cv::Mat diff1 = Qs.row(sample.at(0)) - Qs.row(sample.at(1));
        cv::Mat diff2 = Qs.row(sample.at(0)) - Qs.row(sample.at(2));
        cv::divide(diff1, diff2, diff1);
        if (nearZero(diff1.at<double>(0) - diff1.at<double>(1)) && nearZero(diff1.at<double>(0) - diff1.at<double>(2)))
        {
            return false;
        }
        return true;
    }

    cv::Mat_<double> PlaneKnownNormalRANSAC::compute(std::vector<int> *inliers)
    {
        size_t iters = max_iterations;
        size_t cnt = 0, cnt2 = 0;
        while (cnt < iters && cnt < max_iterations)
        {
            // Get sample
            getSampleIdxs();

            // Estimate plane
            cv::Mat plane = calculatePlaneKnownNormal(Qs.row(sample.at(0)), planeNormal);

            // Get support set
            std::vector<int> supportSet;
            const size_t support = getSupport(plane, supportSet);
            hypotheses.emplace(support, make_pair(plane, move(supportSet)));

            if (support > max_support)
            {
                // Update stopping
                iters = updateStandardStopping(support, length);
                max_support = support;
            }

            if (length > 5 && cnt >= max_iterations / 2UL && hypotheses.begin()->first < 4 && cnt2 < 3)
            {
                // Alter threshold and start over
                th *= 1.25;
                cnt2++;
                cnt = 0;
                iters = max_iterations;
                hypotheses.clear();
                continue;
            }
            cnt++;
        }

        // Get best model
        if (hypotheses.empty())
        {
            return cv::Mat();
        }
        std::multimap<size_t, std::pair<cv::Mat, std::vector<int>>, std::greater<size_t>>::const_iterator it = hypotheses.begin();
        if (it->first == 1)
        {
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        std::vector<int> supportSet2;
        cv::Mat plane = planeMean(it->second.second, supportSet2);
        if (plane.empty())
        {
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        if (inliers)
        {
            *inliers = supportSet2;
        }
        return plane;
    }

    size_t PlaneKnownNormalRANSAC::updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize)
    {
        return updateStandardStoppingRANSAC(numInliers, totPoints, sampleSize, max_iterations, p_conf_th);
    }

    void PlaneKnownNormalRANSAC::getSampleIdxs(const size_t &sampleSize)
    {
        generateUniformRandomSample(length, sampleSize, sample, mt);
    }

    size_t PlaneKnownNormalRANSAC::getSupport(const cv::Mat &plane, std::vector<int> &supportSet)
    {
        for (size_t i = 0; i < length; i++)
        {
            const double d = distanceToPlane(Qs.row(i), plane);
            if (d < th)
            {
                supportSet.emplace_back(i);
            }
        }
        return supportSet.size();
    }

    cv::Mat_<double> PlaneKnownNormalRANSAC::planeMean(const std::vector<int> &supportSet, std::vector<int> &supportSetNew)
    {
        const size_t support = supportSet.size();
        double sum_d = 0;
        cv::Mat planeNormal_t = planeNormal.t();
        for (size_t i = 0; i < support; i++)
        {
            const int &iq = supportSet[i];
            sum_d += -1.0 * Qs.row(iq).dot(planeNormal_t);
        }
        sum_d /= static_cast<double>(support);

        cv::Mat plane;
        planeNormal.copyTo(plane);
        plane.resize(4);
        plane.at<double>(3) = sum_d;

        std::vector<int> supportSet2, supportSet3;
        size_t support2 = getSupport(plane, supportSet2);
        if (support2 < support)
        {
            return cv::Mat();
        }
        else if (support2 > support)
        {
            cv::Mat plane2 = planeMean(supportSet2, supportSet3);
            if (plane2.empty())
            {
                supportSetNew = supportSet2;
                return plane;
            }
            supportSetNew = supportSet3;
            return plane2;
        }
        supportSetNew = supportSet2;
        return plane;
    }

    cv::Point3d CircleRANSAC::compute(std::vector<size_t> *inliers)
    {
        size_t iters = max_iterations;
        size_t cnt = 0, cnt2 = 0;
        while (cnt < iters && cnt < max_iterations)
        {
            //Get sample
            getSampleIdxs();

            //Check if all points are on a line
            if (!checkNotOnLine())
            {
                cnt++;
                continue;
            }

            //Estimate circle
            cv::Point2d mid;
            double radius;
            const double err = fitCircle(ps, mid, radius);
            cv::Point3d model(mid.x, mid.y, radius);
            if (!isfinite(mid.x) || !isfinite(mid.y) || !isfinite(radius))
            {
                continue;
            }
            size_t support = 0;
            if (radius < 0)
            {
                continue;
            }
            else if (err > th)
            {
                hypotheses.emplace(support, make_pair(model, std::vector<size_t>()));
            }
            else
            {
                //Get support set
                std::vector<size_t> supportSet;
                support = getSupport(model, supportSet);
                hypotheses.emplace(support, make_pair(model, move(supportSet)));
            }

            if (support > max_support)
            {
                //Update stopping
                iters = updateStandardStopping(support, length);
                max_support = support;
            }

            if (length > 5 && cnt >= max_iterations / 2UL && hypotheses.begin()->first < 4 && cnt2 < 3)
            {
                //Alter threshold and start over
                th *= 1.25;
                cnt2++;
                cnt = 0;
                iters = max_iterations;
                hypotheses.clear();
                continue;
            }
            cnt++;
        }

        //Get best model
        if (hypotheses.empty())
        {
            return cv::Point3d(0, 0, -1.0);
        }
        std::multimap<size_t, std::pair<cv::Point3d, std::vector<size_t>>, std::greater<size_t>>::const_iterator it = hypotheses.begin();
        std::vector<size_t> supportSet2;
        cv::Point3d model = circleOptimization(it->second.second, it->second.first, supportSet2);
        if (model.z < 0 || supportSet2.empty())
        {
            if (it->second.second.empty())
            {
                return cv::Point3d(0, 0, -1.0);
            }
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        if (inliers)
        {
            *inliers = supportSet2;
        }
        return model;
    }

    size_t CircleRANSAC::updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize)
    {
        return updateStandardStoppingRANSAC(numInliers, totPoints, sampleSize, max_iterations, p_conf_th);
    }

    void CircleRANSAC::getSampleIdxs(const size_t &sampleSize)
    {
        generateUniformRandomSample(length, sampleSize, sample, mt);
    }

    size_t CircleRANSAC::getSupport(const cv::Point3d &circle, std::vector<size_t> &supportSet)
    {
        for (size_t i = 0; i < length; i++)
        {
            const double dx = ps[i].x - circle.x;
            const double dy = ps[i].y - circle.y;
            const double d = abs(sqrt(dx * dx + dy * dy) - circle.z);
            if (d < th)
            {
                supportSet.emplace_back(i);
            }
        }
        return supportSet.size();
    }

    cv::Point3d CircleRANSAC::circleOptimization(const std::vector<size_t> &supportSet, const cv::Point3d &circle, std::vector<size_t> &supportSetNew)
    {
        std::vector<cv::Point2d> points;
        const size_t support = supportSet.size();
        if (support < 3)
        {
            points = ps;
        }
        else
        {
            for (const auto &i : supportSet)
            {
                points.push_back(ps.at(i));
            }
        }
        const cv::Point2d mid_init(circle.x, circle.y);
        const double &radius_init = circle.z;
        cv::Point2d mid;
        double radius;
        if (geometric_circle_fit(points, mid_init, radius_init, mid, radius) != 0)
        {
            return cv::Point3d(0, 0, -1.);
        }
        if (!isfinite(mid.x) || !isfinite(mid.y) || !isfinite(radius))
        {
            return cv::Point3d(0, 0, -1.);
        }
        if (radius < 0)
        {
            return cv::Point3d(0, 0, -1.);
        }
        const cv::Point3d circle_opti(mid.x, mid.y, radius);

        std::vector<size_t> supportSet2, supportSet3;
        size_t support2 = getSupport(circle_opti, supportSet2);
        if (support2 < support)
        {
            return cv::Point3d(0, 0, -1.);
        }
        else if (support2 > support && support2 > 2)
        {
            cv::Point3d circle_opti2 = circleOptimization(supportSet2, circle_opti, supportSet3);
            if (circle_opti2.z < 0)
            {
                supportSetNew = supportSet2;
                return circle_opti;
            }
            supportSetNew = supportSet3;
            return circle_opti2;
        }
        supportSetNew = supportSet2;
        return circle_opti;
    }

    bool CircleRANSAC::checkNotOnLine()
    {
        cv::Point2d diff1 = ps.at(sample.at(1)) - ps.at(sample.at(0));
        cv::Point2d diff2 = ps.at(sample.at(2)) - ps.at(sample.at(0));
        double cross = diff1.x * diff2.y - diff1.y * diff2.x;
        if (nearZero(cross))
        {
            return false;
        }
        return true;
    }

    cv::Mat_<double> LineRANSAC::compute(std::vector<int> *inliers)
    {
        size_t iters = max_iterations;
        size_t cnt = 0, cnt2 = 0;
        while (cnt < iters && cnt < max_iterations)
        {
            //Get sample
            getSampleIdx();

            //Estimate line (just take a point on the line and the provided plane normal vector)
            cv::Mat pt0 = pts.row(sample).t();

            //Get support set
            std::vector<int> supportSet;
            const size_t support = getSupport(pt0, supportSet);
            hypotheses.emplace(support, make_pair(pt0, move(supportSet)));

            if (support > max_support)
            {
                //Update stopping
                iters = updateStandardStopping(support, length);
                max_support = support;
            }

            if (length > 5 && cnt >= max_iterations / 2UL && hypotheses.begin()->first < 10 && cnt2 < 3)
            {
                //Alter threshold and start over
                th *= 1.25;
                cnt2++;
                cnt = 0;
                iters = max_iterations;
                hypotheses.clear();
                continue;
            }
            cnt++;
        }

        //Get best model
        if (hypotheses.empty())
        {
            return cv::Mat();
        }
        std::multimap<size_t, std::pair<cv::Mat, std::vector<int>>, std::greater<size_t>>::const_iterator it = hypotheses.begin();
        if (it->first < 5)
        {
            return cv::Mat();
        }
        std::vector<int> supportSet2;
        cv::Mat pt0 = ptMean(it->second.second, supportSet2);
        if (pt0.empty())
        {
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        if (inliers)
        {
            *inliers = supportSet2;
        }
        return pt0;
    }

    size_t LineRANSAC::updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize)
    {
        return updateStandardStoppingRANSAC(numInliers, totPoints, sampleSize, max_iterations, p_conf_th);
    }

    void LineRANSAC::getSampleIdx()
    {
        sample = mt() % length;
    }

    size_t LineRANSAC::getSupport(const cv::Mat &pt, std::vector<int> &supportSet)
    {
        for (size_t i = 0; i < length; i++)
        {
            cv::Mat v = pts.row(i).t() - pt;
            v = v.cross(planeNormal);
            const cv::Vec3d vv(v);
            const double d = sqrt(vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2]);
            if (d < th)
            {
                supportSet.emplace_back(i);
            }
        }
        return supportSet.size();
    }

    cv::Mat_<double> LineRANSAC::ptMean(const std::vector<int> &supportSet, std::vector<int> &supportSetNew)
    {
        cv::Mat pt0 = cv::Mat::zeros(1, 3, CV_64FC1);
        const size_t support = supportSet.size();
        for (const auto &i : supportSet)
        {
            pt0 += pts.row(i);
        }
        pt0 /= static_cast<double>(support);
        pt0 = pt0.t();

        std::vector<int> supportSet2, supportSet3;
        size_t support2 = getSupport(pt0, supportSet2);
        if (support2 < support)
        {
            return cv::Mat();
        }
        else if (support2 > support)
        {
            cv::Mat pt02 = ptMean(supportSet2, supportSet3);
            if (pt02.empty())
            {
                supportSetNew = supportSet2;
                return pt0;
            }
            supportSetNew = supportSet3;
            return pt02;
        }
        supportSetNew = supportSet2;
        return pt0;
    }

    std::pair<cv::Mat_<double>, double> Rigid3DTransformRANSAC::compute(std::vector<int> *inliers)
    {
        size_t iters = max_iterations;
        size_t cnt = 0, cnt2 = 0;
        while (cnt < iters && cnt < max_iterations)
        {
            //Get sample
            getSampleIdxs();

            //Estimate rigid 3D transformation (R, t, scale)
            cv::Mat_<double> pts1, pts2;
            for (const auto &i : sample)
            {
                // cv::Vec3d p1 = points1(i);
                pts1.push_back(points1.row(i));
                pts2.push_back(points2.row(i));
            }
            double scale = 1.0;
            cv::Mat_<double> P = estimateRigid3DTansformation(pts1, pts2, &scale);
            // printCvMat(P, "P");

            //Get support set
            std::vector<int> supportSet;
            const size_t support = getSupport(P, scale, supportSet);
            hypotheses.emplace(support, make_tuple(P, scale, move(supportSet)));

            if (support > max_support)
            {
                //Update stopping
                iters = updateStandardStopping(support, length);
                max_support = support;
            }

            if (length > 5 && cnt >= max_iterations / 2UL && hypotheses.begin()->first < 4 && cnt2 < 3)
            {
                //Alter threshold and start over
                th *= 1.25;
                cnt2++;
                cnt = 0;
                iters = max_iterations;
                hypotheses.clear();
                continue;
            }
            cnt++;
        }

        //Get best model
        if (hypotheses.empty())
        {
            return make_pair(cv::Mat(), 0.0);
        }
        std::multimap<size_t, std::tuple<cv::Mat_<double>, double, std::vector<int>>, std::greater<size_t>>::const_iterator it = hypotheses.begin();
        if (it->first < 3)
        {
            return make_pair(cv::Mat(), 0.0);
        }
        if (it->first == 3)
        {
            if (inliers)
            {
                *inliers = std::get<2>(it->second);
            }
            return make_pair(std::get<0>(it->second), std::get<1>(it->second));
        }
        std::vector<int> supportSet2;
        std::pair<cv::Mat_<double>, double> Ps = transformLeastSquares(std::get<2>(it->second), supportSet2);
        if (Ps.first.empty())
        {
            if (inliers)
            {
                *inliers = std::get<2>(it->second);
            }
            return make_pair(std::get<0>(it->second), std::get<1>(it->second));
        }
        // printCvMat(Ps.first, "P_final");
        if (inliers)
        {
            *inliers = supportSet2;
        }
        return Ps;
    }

    size_t Rigid3DTransformRANSAC::updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize)
    {
        return updateStandardStoppingRANSAC(numInliers, totPoints, sampleSize, max_iterations, p_conf_th);
    }

    void Rigid3DTransformRANSAC::getSampleIdxs(const size_t &sampleSize)
    {
        generateUniformRandomSample(length, sampleSize, sample, mt);
    }

    size_t Rigid3DTransformRANSAC::getSupport(const cv::Mat_<double> &P, const double &scale, std::vector<int> &supportSet)
    {
        for (size_t i = 0; i < length; i++)
        {
            cv::Mat_<double> p1 = points1.row(i).t();
            p1 *= scale;
            cv::Mat_<double> p2 = points2.row(i).t();
            const double d = getRigidTransformPt3DError(P, p1, p2);
            if (d < th)
            {
                supportSet.emplace_back(i);
            }
        }
        return supportSet.size();
    }

    std::pair<cv::Mat_<double>, double> Rigid3DTransformRANSAC::transformLeastSquares(const std::vector<int> &supportSet, std::vector<int> &supportSetNew)
    {
        const size_t support = supportSet.size();
        cv::Mat_<double> pts1, pts2;
        for (const auto &i : supportSet)
        {
            pts1.push_back(points1.row(i));
            pts2.push_back(points2.row(i));
        }
        double scale = 1.0;
        cv::Mat_<double> P = estimateRigid3DTansformation(pts1, pts2, &scale);

        std::vector<int> supportSet2, supportSet3;
        size_t support2 = getSupport(P, scale, supportSet2);
        if (support2 < support)
        {
            return make_pair(cv::Mat(), 0.0);
        }
        else if (support2 > support)
        {
            std::pair<cv::Mat_<double>, double> P2 = transformLeastSquares(supportSet2, supportSet3);
            if (P2.first.empty())
            {
                supportSetNew = supportSet2;
                return make_pair(P, scale);
            }
            supportSetNew = supportSet3;
            return P2;
        }
        supportSetNew = supportSet2;
        return make_pair(P, scale);
    }

    cv::Mat VectorRANSAC::compute(std::vector<size_t> *inliers)
    {
        size_t iters = max_iterations;
        size_t cnt = 0, cnt2 = 0;
        while (cnt < iters && cnt < max_iterations)
        {
            //Get sample
            getSampleIdxs();

            //Estimate vector
            // cv::Mat vec = getVectorMean(vecs.at(sample.at(0)), vecs.at(sample.at(1)));
            cv::Mat vec = vecs.at(sample.at(0));

            //Get support set
            std::vector<size_t> supportSet;
            const size_t support = getSupport(vec, supportSet);
            hypotheses.emplace(support, make_pair(vec, move(supportSet)));
            if (support > max_support){
                //Update stopping
                iters = updateStandardStopping(support, length);
                max_support = support;
            }

            if (length > 5 && cnt >= max_iterations / 2UL && hypotheses.begin()->first < 4 && cnt2 < 3)
            {
                //Alter threshold and start over
                th *= 1.25;
                cnt2++;
                cnt = 0;
                iters = max_iterations;
                hypotheses.clear();
                continue;
            }
            cnt++;
        }

        //Get best model
        if (hypotheses.empty())
        {
            return cv::Mat();
        }
        std::multimap<size_t, std::pair<cv::Mat, std::vector<size_t>>, std::greater<size_t>>::const_iterator it = hypotheses.begin();
        if (it->first == 1)
        {
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        std::vector<size_t> supportSet2;
        cv::Mat vec = vectorCenterMass(it->second.second, supportSet2);
        if (vec.empty())
        {
            if (inliers)
            {
                *inliers = it->second.second;
            }
            return it->second.first;
        }
        if (inliers)
        {
            *inliers = supportSet2;
        }
        return vec;
    }

    cv::Mat VectorRANSAC::getVectorMean(const cv::Mat &v1, const cv::Mat &v2) const
    {
        cv::Mat vm = v1 + v2;
        if (normalize){
            vm /= cv::norm(vm);
        }else{
            vm /= 2.0;
        }
        return vm;
    }

    size_t VectorRANSAC::updateStandardStopping(const size_t &numInliers, const size_t &totPoints, const size_t &sampleSize)
    {
        return updateStandardStoppingRANSAC(numInliers, totPoints, sampleSize, max_iterations, p_conf_th);
    }

    void VectorRANSAC::getSampleIdxs(const size_t &sampleSize)
    {
        generateUniformRandomSample(length, sampleSize, sample, mt);
    }

    size_t VectorRANSAC::getSupport(const cv::Mat &vec, std::vector<size_t> &supportSet) const
    {
        for (size_t i = 0; i < length; i++)
        {
            const double d = getAngularDistance(vecs.at(i), vec);
            if (d < th)
            {
                supportSet.emplace_back(i);
            }
        }
        return supportSet.size();
    }

    double VectorRANSAC::getAngularDistance(const cv::Mat &v1, const cv::Mat &v2) const {
        return 1.0 - (v1.dot(v2) / (cv::norm(v1) * cv::norm(v2)));
    }

    cv::Mat VectorRANSAC::vectorCenterMass(const std::vector<size_t> &supportSet, std::vector<size_t> &supportSetNew) const
    {
        cv::Mat vec = cv::Mat::zeros(vecs.at(0).size(), vecs.at(0).type());
        for (const auto &i : supportSet)
        {
            vec += vecs.at(i);
        }
        if (normalize)
        {
            vec /= cv::norm(vec);
        }
        else
        {
            vec /= static_cast<double>(supportSet.size());
        }

        const size_t support = supportSet.size();

        std::vector<size_t> supportSet2, supportSet3;
        size_t support2 = getSupport(vec, supportSet2);
        if (support2 < support)
        {
            return cv::Mat();
        }
        else if (support2 > support)
        {
            cv::Mat vec2 = vectorCenterMass(supportSet2, supportSet3);
            if (vec2.empty())
            {
                supportSetNew = supportSet2;
                return vec;
            }
            supportSetNew = supportSet3;
            return vec2;
        }
        supportSetNew = supportSet2;
        return vec;
    }
}