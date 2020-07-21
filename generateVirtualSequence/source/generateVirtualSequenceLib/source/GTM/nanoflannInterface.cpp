//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2020 Josef Maier
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

#include "GTM/nanoflannInterface.h"
#include "nanoflann.hpp"
#include <memory>
#include <stdexcept>

using namespace nanoflann;

/* --------------------------- Defines --------------------------- */


/* ---------------------- Classes & Structs ---------------------- */

// Data structure for the tree
struct CoordinateInterface
{
    std::list<KeyPoint> *featuresPtr;
    std::unordered_map<size_t, std::list<KeyPoint>::iterator> *poolIdxIt;

    CoordinateInterface():
    featuresPtr(nullptr),
    poolIdxIt(nullptr){}
};

// And this is the "dataset to kd-tree" adaptor class:
template <typename Derived>
struct PointCloudAdaptor
{
    //typedef typename Derived::coord_t coord_t;

    const Derived &obj; //!< A const ref to the data set origin

                        /// The constructor that sets the data set source
    explicit PointCloudAdaptor(const Derived &obj_) : obj(obj_) { }

    /// CRTP helper method
    inline const Derived& derived() const { return obj; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return derived().size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim == 0) return derived()[idx].pt.x;
        else return derived()[idx].pt.y;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

}; // end of PointCloudAdaptor

class keyPointTree
{
private:
    const std::vector<cv::KeyPoint> *coordInteraface;

    typedef PointCloudAdaptor<std::vector<cv::KeyPoint>> PC2KD;
    std::unique_ptr<PC2KD> pc2kd; // The adaptor
                                  // construct a kd-tree index:
    typedef KDTreeSingleIndexDynamicAdaptor<
        L2_Simple_Adaptor<float, PC2KD >,
        PC2KD,
        2 /* dim */
    > coordinateKDTree;
    std::unique_ptr<coordinateKDTree> index;
public:

    keyPointTree(const std::vector<cv::KeyPoint> *featuresPtr_)
    {
        coordInteraface = featuresPtr_;
        pc2kd.reset(new PC2KD(*coordInteraface));
    }

    int buildInitialTree()
    {
        try
        {
            index.reset(new coordinateKDTree(2 /*dim*/, *pc2kd, KDTreeSingleIndexAdaptorParams(20 /* max leaf */)));
        }
        catch (std::exception const& ex)
        {
            std::cerr << "std::exception: " << ex.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown Exception!" << std::endl;
            throw;
        }
        return 0;
    }

    int resetTree(const std::vector<cv::KeyPoint> *featuresPtr_)
    {
        coordInteraface = featuresPtr_;
        pc2kd.reset(new PC2KD(*coordInteraface));
        try
        {
            index.reset(new coordinateKDTree(2 /*dim*/, *pc2kd, KDTreeSingleIndexAdaptorParams(20 /* max leaf */)));
        }
        catch (std::exception const& ex)
        {
            std::cerr << "std::exception: " << ex.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown Exception!" << std::endl;
            throw;
        }
        return 0;
    }

    void killTree()
    {
        index.release();
        pc2kd.release();
    }

    size_t knnSearch(const cv::Point2f &queryPt, size_t knn, std::vector<std::pair<size_t, float>> & result)
    {
        result.clear();
        size_t *indices = new size_t[knn];
        float *distances = new float[knn];
        float queryPt_[2];
        size_t nr_results = 0;
        queryPt_[0] = queryPt.x;
        queryPt_[1] = queryPt.y;
        KNNResultSet<float> resultSet(knn);
        resultSet.init(indices, distances);
        try
        {
            index->findNeighbors(resultSet, &queryPt_[0], nanoflann::SearchParams(10));
        }
        catch (std::exception const& ex)
        {
            std::cerr << "std::exception: " << ex.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown Exception!" << std::endl;
            throw;
        }
        nr_results = resultSet.size();
        result.reserve(nr_results);
        for (size_t i = 0; i < nr_results; i++)
        {
            result.emplace_back(std::make_pair(indices[i], distances[i]));
        }
        delete[] indices;
        delete[] distances;
        return nr_results;
    }

    size_t radiusSearch(const cv::Point2f &queryPt, float radius, std::vector<std::pair<size_t, float>> & result)
    {
        // Unsorted radius search:
        result.clear();
        RadiusResultSet<float, size_t> resultSet(radius, result);
        float queryPt_[2];
        queryPt_[0] = queryPt.x;
        queryPt_[1] = queryPt.y;
        try
        {
            index->findNeighbors(resultSet, queryPt_, nanoflann::SearchParams());
        }
        catch (std::exception const& ex)
        {
            std::cerr << "std::exception: " << ex.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown Exception!" << std::endl;
            throw;
        }
        //Sort
        if (!result.empty()) {
            std::sort(result.begin(), result.end(),
                      [](std::pair<size_t, float> const &first, std::pair<size_t, float> const &second) {
                          return first.second < second.second;
                      });
        }
        return result.size();
    }
};

/* --------------------- Function prototypes --------------------- */


/* -------------------------- Functions -------------------------- */

keyPointTreeInterface::keyPointTreeInterface(const std::vector<cv::KeyPoint> *featuresPtr_)
{
    treePtr = new keyPointTree(featuresPtr_);
}

int keyPointTreeInterface::buildInitialTree()
{
    return ((keyPointTree*)treePtr)->buildInitialTree();
}

int keyPointTreeInterface::resetTree(const std::vector<cv::KeyPoint> *featuresPtr_)
{
    if (!treePtr)
    {
        treePtr = new keyPointTree(featuresPtr_);
    }
    else
    {
        return ((keyPointTree*)treePtr)->resetTree(featuresPtr_);
    }

    return 0;
}

void keyPointTreeInterface::killTree()
{
    ((keyPointTree*)treePtr)->killTree();
    if (treePtr)
    {
        delete (keyPointTree*)treePtr;
        treePtr = nullptr;
    }
}

size_t keyPointTreeInterface::knnSearch(const cv::Point2f &queryPt, size_t knn, std::vector<std::pair<size_t, float>> & result)
{
    return ((keyPointTree*)treePtr)->knnSearch(queryPt, knn, result);
}

size_t keyPointTreeInterface::radiusSearch(const cv::Point2f &queryPt, float radius, std::vector<std::pair<size_t, float>> & result)
{
    return ((keyPointTree*)treePtr)->radiusSearch(queryPt, radius, result);
}

keyPointTreeInterface::~keyPointTreeInterface()
{
    if(treePtr) {
        killTree();
    }
}
