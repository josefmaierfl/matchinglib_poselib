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

#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <opencv2/highgui.hpp>

#include "utilslib/utilslib_api.h"

namespace utilslib
{
    struct UTILSLIB_API CameraParameters
    {
        double fx, fy, cx, cy, skew;
        std::vector<double> distortionParams;
    };    

    struct UTILSLIB_API Statistics
    {
        Statistics(const std::vector<double> &vals);

        Statistics() : mean(0.), median(0.), sd(0.), min(0.), max(0.), nr_elements(0) {}

        void scale(const double &s);

        double mean = 0.;
        double median = 0.;
        double sd = 0.;
        double min = 0.;
        double max = 0.;
        size_t nr_elements = 0;
    };

    std::string UTILSLIB_API str_replace(const std::string &in, const std::vector<char> &from, const std::string &to);

    size_t UTILSLIB_API binomialCoeff(const size_t &n, const size_t &k);

    void UTILSLIB_API printCvMat(const cv::Mat &m, const std::string &name);

    template <class T>
    T UTILSLIB_API getMedian(std::vector<T> &measurements)
    {
        const size_t length = measurements.size();
        std::sort(measurements.begin(), measurements.end());
        T median;
        if (length % 2)
        {
            median = measurements[(length - 1) / 2];
        }
        else
        {
            median = (measurements[length / 2] + measurements[length / 2 - 1]) / static_cast<T>(2.0);
        }
        return median;
    }

    template <class T>
    T UTILSLIB_API getMedianConst(const std::vector<T> &measurements)
    {
        std::vector<T> tmp = measurements;
        return getMedian(tmp);
    }

    template <class T>
    std::pair<int, int> UTILSLIB_API getMedianIdx(const std::vector<T> &measurements)
    {
        const size_t length = measurements.size();
        std::vector<size_t> idx(length);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&measurements](const size_t &i1, const size_t &i2)
                         { return measurements[i1] < measurements[i2]; });
        if (length % 2)
        {
            return std::make_pair(static_cast<int>(idx[(length - 1) / 2]), -1);
        }
        return std::make_pair(static_cast<int>(idx[length / 2]), static_cast<int>(idx[length / 2 - 1]));
    }

    template <class T>
    size_t UTILSLIB_API getQuantilePos(const std::vector<T> &measurements, const double &pos)
    {
        const size_t length = measurements.size();
        if (length == 0)
        {
            return 0;
        }
        double pos_take = std::round(static_cast<double>(length) * pos) - 1.0;
        if (pos_take < 0.)
        {
            pos_take = 0.;
        }
        else if (pos_take > static_cast<double>(length) - 1.0)
        {
            pos_take = static_cast<double>(length) - 1.0;
        }
        return static_cast<size_t>(pos_take);
    }

    template <class T>
    T UTILSLIB_API getQuantile(const std::vector<T> &measurements, const double &pos)
    {
        const size_t length = measurements.size();
        if(length == 0){
            return 0;
        }
        const size_t pos_take = getQuantilePos(measurements, pos);
        std::vector<T> tmp = measurements;
        std::sort(tmp.begin(), tmp.end());
        return tmp.at(pos_take);
    }

    template <class T>
    T UTILSLIB_API getMean(const std::vector<T> &measurements)
    {
        T mean = 0;
        T n_d = static_cast<T>(measurements.size());
        for (const auto &val : measurements)
        {
            mean += val;
        }
        mean /= n_d;

        return mean;
    }

    template <class T>
    T UTILSLIB_API getMeanToQuantile(const std::vector<T> &measurements, const double &pos)
    {
        const size_t length = measurements.size();
        if (length == 0)
        {
            return 0;
        }
        const size_t pos_take = getQuantilePos(measurements, pos);
        std::vector<T> tmp = measurements;
        std::sort(tmp.begin(), tmp.end());
        T mean = 0;
        for (size_t i = 0; i <= pos_take; i++)
        {
            mean += tmp.at(i);
        }
        mean /= static_cast<T>(pos_take + 1);

        return mean;
    }

    template <class T>
    void UTILSLIB_API getMeanStandardDeviation(const std::vector<T> &measurements, T &mean, T &sd)
    {
        mean = 0;
        T err2sum = 0;
        T n_d = static_cast<T>(measurements.size());
        for (const auto &val : measurements)
        {
            mean += val;
            err2sum += val * val;
        }
        mean /= n_d;

        T hlp = err2sum - n_d * mean * mean;
        sd = std::sqrt(hlp / (n_d - static_cast<T>(1.0)));
    }

    template <class T>
    std::pair<T, T> UTILSLIB_API getMeanStandardDeviation(const std::vector<T> &measurements)
    {
        T mean, sd;
        getMeanStandardDeviation(measurements, mean, sd);
        return std::make_pair(mean, sd);
    }

    template <class T>
    inline bool UTILSLIB_API nearZero(const T d, const double EPSILON = 1e-4)
    {
        return (static_cast<double>(d) < EPSILON) && (static_cast<double>(d) > -EPSILON);
    }

    template <class T>
    inline void UTILSLIB_API hash_combine(std::size_t &seed, const T &v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    struct UTILSLIB_API pair_hash
    {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &x) const
        {
            std::size_t seed = 0;
            hash_combine(seed, x.first);
            hash_combine(seed, x.second);
            return seed;
            return std::hash<T1>()(x.first) ^ std::hash<T2>()(x.second);
        }
    };

    struct UTILSLIB_API pair_EqualTo
    {
        template <class T1, class T2>
        bool operator()(const std::pair<T1, T2> &x1, const std::pair<T1, T2> &x2) const
        {
            const bool equal1 = (x1.first == x2.first);
            const bool equal2 = (x1.second == x2.second);
            return equal1 && equal2;
        }
    };

    namespace hash_tuple
    {
        // Recursive template code derived from Matthieu M.
        // Source: https://stackoverflow.com/questions/7110301/generic-hash-for-tuples-in-unordered-map-unordered-set
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct UTILSLIB_API HashValueImpl
        {
            static void apply(size_t &seed, Tuple const &tuple)
            {
                HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
                hash_combine(seed, std::get<Index>(tuple));
            }
        };

        template <class Tuple>
        struct UTILSLIB_API HashValueImpl<Tuple, 0>
        {
            static void apply(size_t &seed, Tuple const &tuple)
            {
                hash_combine(seed, std::get<0>(tuple));
            }
        };

        template <typename>
        struct UTILSLIB_API hashT;

        template <typename... TT>
        struct UTILSLIB_API hashT<std::tuple<TT...>>
        {
            size_t operator()(std::tuple<TT...> const &tt) const
            {
                size_t seed = 0;
                HashValueImpl<std::tuple<TT...>>::apply(seed, tt);
                return seed;
            }
        };

        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct UTILSLIB_API TupleReadValueImpl
        {
            static void isEqual(bool &is_equal, Tuple const &tuple1, Tuple const &tuple2)
            {
                TupleReadValueImpl<Tuple, Index - 1>::isEqual(is_equal, tuple1, tuple2);
                is_equal &= (std::get<Index>(tuple1) == std::get<Index>(tuple2));
            }
        };

        template <class Tuple>
        struct UTILSLIB_API TupleReadValueImpl<Tuple, 0>
        {
            static void isEqual(bool &is_equal, Tuple const &tuple1, Tuple const &tuple2)
            {
                is_equal &= (std::get<0>(tuple1) == std::get<0>(tuple2));
            }
        };

        template <typename>
        struct UTILSLIB_API equalTo;

        template <typename... TT>
        struct UTILSLIB_API equalTo<std::tuple<TT...>>
        {
            size_t operator()(std::tuple<TT...> const &tt1, std::tuple<TT...> const &tt2) const
            {
                bool is_equal = true;
                TupleReadValueImpl<std::tuple<TT...>>::isEqual(is_equal, tt1, tt2);
                return is_equal;
            }
        };
    }

    //Pointer wrapper for pybind
    //From: https://stackoverflow.com/questions/48982143/returning-and-passing-around-raw-pod-pointers-arrays-with-python-c-and-pyb
    template <class T>
    class UTILSLIB_API ptr_wrapper
    {
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T *ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper &other) : ptr(other.ptr) {}
        T &operator*() const { return *ptr; }
        T *operator->() const { return ptr; }
        T *get() const { return ptr; }
        // void destroy() { delete ptr; }
        // T &operator[](std::size_t idx) const { return ptr[idx]; }

    private:
        T *ptr;
    };

    template <class T>
    struct UTILSLIB_API ThreadSafeValueAccess
    {
        T &value;
        std::mutex m;

        ThreadSafeValueAccess(T &value_) : value(value_) {}

        T read()
        {
            const std::lock_guard<std::mutex> lock(m);
            return value;
        }

        void write(const T &value_)
        {
            const std::lock_guard<std::mutex> lock(m);
            value = value_;
        }

        ThreadSafeValueAccess &operator=(const T &value_)
        {
            const std::lock_guard<std::mutex> lock(m);
            value = value_;
            return *this;
        }

        T operator+(const T &value_)
        {
            const std::lock_guard<std::mutex> lock(m);
            return value + value_;
        }

        ThreadSafeValueAccess &operator+=(const T &value_)
        {
            const std::lock_guard<std::mutex> lock(m);
            value += value_;
            return *this;
        }

        T operator/(const T &value_)
        {
            const std::lock_guard<std::mutex> lock(m);
            return value / value_;
        }

        ThreadSafeValueAccess &operator/=(const T &value_)
        {
            const std::lock_guard<std::mutex> lock(m);
            value /= value_;
            return *this;
        }
    };

    class UTILSLIB_API Semaphore
    {
    public:
        Semaphore(const int &count_ = 0)
            : count(count_), cnt_waiting(0) {}

        void release()
        {
            const std::lock_guard<std::mutex> lock(mtx);
            count++;
            cv.notify_one();
        }

        void release_notifyAll()
        {
            const std::lock_guard<std::mutex> lock(mtx);
            count = cnt_waiting;
            cv.notify_all();
        }

        bool release_ifNrWaitingOrWait(const int &count_, const size_t &maxSecs = 0)
        {
            std::unique_lock<std::mutex> lock(mtx);
            if (cnt_waiting >= count_){
                count = cnt_waiting;
                cv.notify_all();
                return true;
            }
            cnt_waiting++;
            if(maxSecs){
                using namespace std::chrono_literals;
                cv.wait_for(lock, maxSecs * 1s, [this](){ return count > 0; });
                if(count > 0){
                    count--;
                }else{
                    cnt_waiting--;
                    count = cnt_waiting;
                    cv.notify_all();
                    return true;
                }
            }else{
                cv.wait(lock, [this](){ return count > 0; });
                count--;
            }
            cnt_waiting--;
            return false;
        }

        void acquire()
        {
            std::unique_lock<std::mutex> lock(mtx);
            cnt_waiting++;
            cv.wait(lock, [this](){ return count > 0; });
            cnt_waiting--;
            count--;
        }

        bool try_acquire()
        {
            const std::lock_guard<std::mutex> lock(mtx);
            if (count > 0)
            {
                count--;
                return true;
            }
            return false;
        }

        void setCount(const int &count_){
            std::lock_guard<std::mutex> lock(mtx);
            count = count_;
        }

        void addCount(const int &count_)
        {
            std::lock_guard<std::mutex> lock(mtx);
            count += count_;
        }

    private:
        std::mutex mtx;
        std::condition_variable cv;
        int count = 0, cnt_waiting = 0;
    };

    template <class T>
    void UTILSLIB_API getThreadBatchSize(const T &nrTasks, T &threadCount, T &batchSize)
    {
        T batchDiff = threadCount * batchSize - nrTasks;
        while (batchDiff >= batchSize && threadCount > 1)
        {
            threadCount--;
            batchSize = std::ceil(nrTasks / static_cast<float>(threadCount));
            batchDiff = threadCount * batchSize - nrTasks;
        }
    }

    inline void UTILSLIB_API convertPrecision(double &inOut, const double &preci = FLT_EPSILON)
    {
        if (nearZero(preci - 1.0) || preci > 1.0)
        {
            return;
        }
        else if (nearZero(preci / FLT_EPSILON - 1.0))
        {
            const float tmp = static_cast<float>(inOut);
            inOut = static_cast<double>(tmp);
        }
        else if (preci > DBL_EPSILON)
        {
            inOut = std::round(inOut / preci) * preci;
        }
    }

    inline double UTILSLIB_API convertPrecisionRet(const double &in, const double &preci = FLT_EPSILON)
    {
        if (nearZero(preci - 1.0) || preci > 1.0)
        {
            return in;
        }
        else if (nearZero(preci / FLT_EPSILON - 1.0))
        {
            const float tmp = static_cast<float>(in);
            return static_cast<double>(tmp);
        }
        else if (preci > DBL_EPSILON)
        {
            return std::round(in / preci) * preci;
        }
        return in;
    }

    inline void UTILSLIB_API convertPrecisionMat(cv::InputOutputArray mat, const double &preci = FLT_EPSILON)
    {
        CV_Assert(mat.type() == CV_64FC1);
        cv::Mat mat_ = mat.getMat();
        for (int y = 0; y < mat_.rows; y++)
        {
            for (int x = 0; x < mat_.cols; x++)
            {
                convertPrecision(mat_.at<double>(y, x), preci);
            }
        }
    }

    inline void UTILSLIB_API convertPrecisionVec(std::vector<double> &vec, const double &preci = FLT_EPSILON)
    {
        for (auto &v : vec)
        {
            convertPrecision(v, preci);
        }
    }
}
