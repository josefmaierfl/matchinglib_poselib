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

#include "utils_common.h"
#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>

#include <opencv2/highgui.hpp>
#include "utilslib/utilslib_api.h"

namespace utilslib
{
    // From: https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
    class UTILSLIB_API RandomGenerator
    {
    public:
        static RandomGenerator &getInstance(std::seed_seq seed = {0})
        {
            static RandomGenerator instance(seed); // Guaranteed to be destroyed.
                               // Instantiated on first use.
            return instance;
        }

    private:
        RandomGenerator(std::seed_seq &seed) : mt(seed) {
            // std::cout << "RandomGenerator()" << std::endl;
        } // Constructor? (the {} brackets) are needed here.

        std::mt19937 mt;
        // C++ 11
        // =======
        // We can use the better technique of deleting the methods
        // we don't want.
    public:
        RandomGenerator() = delete;
        RandomGenerator(RandomGenerator const &) = delete;
        void operator=(RandomGenerator const &) = delete;
        // Note: Scott Meyers mentions in his Effective Modern
        //       C++ book, that deleted functions should generally
        //       be public as it results in better error messages
        //       due to the compilers behavior to check accessibility
        //       before deleted status

        std::mt19937 &getTwisterEngineRef(){
            // std::cout << "Returning ref: " << mt() << std::endl;
            return mt;
        }

        template <typename T>
        T getRandomNumber(const T &maxNr = T(0)){
            if(maxNr){
                return mt() % maxNr;
            }
            return mt();
        }
    };

    std::vector<size_t> UTILSLIB_API getVecOfMapSizes(const std::unordered_map<std::pair<int, int>, std::pair<std::vector<cv::KeyPoint>, cv::Mat>, pair_hash, pair_EqualTo> &keypoints_descriptors);
    std::seed_seq UTILSLIB_API getSeedSeqFromVec(const std::vector<size_t> &vec);
    std::seed_seq UTILSLIB_API getSeedSeqFromVecOfMapSizes(const std::unordered_map<std::pair<int, int>, std::pair<std::vector<cv::KeyPoint>, cv::Mat>, pair_hash, pair_EqualTo> &keypoints_descriptors);
}
