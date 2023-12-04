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

#include "matchinglib/random_numbers.h"

namespace utilslib
{
    std::vector<size_t> getVecOfMapSizes(const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors)
    {
        std::vector<size_t> sizes;
        sizes.reserve(keypoints_descriptors.size());
        for (const auto &vec : keypoints_descriptors)
        {
            sizes.emplace_back(vec.second.first.size());
        }
        return sizes;
    }

    std::seed_seq getSeedSeqFromVec(const std::vector<size_t> &vec)
    {
        return std::seed_seq(vec.begin(), vec.end());
    }

    std::seed_seq getSeedSeqFromVecOfMapSizes(const std::unordered_map<int, std::pair<std::vector<cv::KeyPoint>, cv::Mat>> &keypoints_descriptors)
    {
        return getSeedSeqFromVec(getVecOfMapSizes(keypoints_descriptors));
    }
}