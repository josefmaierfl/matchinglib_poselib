//
// Created by maierj on 11.01.19.
//

#ifndef GENERATEVIRTUALSEQUENCE_NANOFLANN_UTILS_H
#define GENERATEVIRTUALSEQUENCE_NANOFLANN_UTILS_H

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"

struct SeedCloud
{
    std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> *seedsNear;
    std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> *seedsMid;
    std::vector<std::vector<std::vector<cv::Point3_<int32_t>>>> *seedsFar;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        size_t cloudSize = 0;
        for (const auto& i : *seedsNear) {
            for (const auto& j : i) {
                cloudSize += j.size();
            }
        }
        for (const auto& i : *seedsMid) {
            for (const auto& j : i) {
                cloudSize += j.size();
            }
        }
        for (const auto& i : *seedsFar) {
            for (const auto& j : i) {
                cloudSize += j.size();
            }
        }

        return cloudSize;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline int32_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        size_t idx_tmp = idx + 1;
        for (const auto& i : *seedsNear) {
            for (const auto& j : i) {
                if(j.size() >= idx_tmp)
                {
                    if (dim == 0)
                    {
                        return j[idx_tmp - 1].x;
                    } else
                    {
                        return j[idx_tmp - 1].y;
                    }
                } else{
                    idx_tmp -= j.size();
                }
            }
        }
        for (const auto& i : *seedsMid) {
            for (const auto& j : i) {
                if(j.size() >= idx_tmp)
                {
                    if (dim == 0)
                    {
                        return j[idx_tmp - 1].x;
                    } else
                    {
                        return j[idx_tmp - 1].y;
                    }
                } else{
                    idx_tmp -= j.size();
                }
            }
        }
        for (const auto& i : *seedsFar) {
            for (const auto& j : i) {
                if(j.size() >= idx_tmp)
                {
                    if (dim == 0)
                    {
                        return j[idx_tmp - 1].x;
                    } else
                    {
                        return j[idx_tmp - 1].y;
                    }
                } else{
                    idx_tmp -= j.size();
                }
            }
        }
        cerr << "Coordinate with Nanoflann index " << idx << "not found!" << endl;
        return 0;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};


#endif //GENERATEVIRTUALSEQUENCE_NANOFLANN_UTILS_H
