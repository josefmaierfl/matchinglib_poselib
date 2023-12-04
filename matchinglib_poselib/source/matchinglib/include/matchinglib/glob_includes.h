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

#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
//#include <cmath>
#include <math.h>       /* isnan, sqrt */

#include "matchinglib/matchinglib_api.h"

namespace matchinglib
{
    template <class T>
    inline bool nearZero(const T d, const double EPSILON = 1e-4)
    {
        return (static_cast<double>(d) < EPSILON) && (static_cast<double>(d) > -EPSILON);
    }

    template <class T>
    void getThreadBatchSize(const T &nrTasks, T &threadCount, T &batchSize)
    {
        T batchDiff = threadCount * batchSize - nrTasks;
        while (batchDiff >= batchSize && threadCount > 1)
        {
            threadCount--;
            batchSize = std::ceil(nrTasks / static_cast<float>(threadCount));
            batchDiff = threadCount * batchSize - nrTasks;
        }
    }

    template <class T>
    inline void MATCHINGLIB_API hash_combine(std::size_t &seed, const T &v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    struct MATCHINGLIB_API pair_hash
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

    struct MATCHINGLIB_API pair_EqualTo
    {
        template <class T1, class T2>
        bool operator()(const std::pair<T1, T2> &x1, const std::pair<T1, T2> &x2) const
        {
            const bool equal1 = (x1.first == x2.first);
            const bool equal2 = (x1.second == x2.second);
            return equal1 && equal2;
        }
    };

    //Pointer wrapper for pybind
    //From: https://stackoverflow.com/questions/48982143/returning-and-passing-around-raw-pod-pointers-arrays-with-python-c-and-pyb
    template <class T>
    class MATCHINGLIB_API ptr_wrapper
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
}