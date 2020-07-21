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
/**********************************************************************************************************
FILE: nanoflannInterface.cpp

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions to search keypoint positions within a tree built using nanoflann
**********************************************************************************************************/

#include "poselib/nanoflannInterface.h"
#include "nanoflann.hpp"
#include <memory>
#include <stdexcept>

using namespace nanoflann;

namespace poselib
{

    /* --------------------------- Defines --------------------------- */


    /* ---------------------- Classes & Structs ---------------------- */

    class InvalidPoolIteratorException : public std::runtime_error
    {
    public:
        explicit InvalidPoolIteratorException(const std::string &mess) : std::runtime_error(mess) {}
    };

    // Data structure for the tree
    struct CoordinateInterface
    {
        std::list<CoordinateProps> *correspondencePool;
        std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt;

        CoordinateInterface():
        correspondencePool(nullptr),
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
        inline size_t kdtree_get_point_count() const { return derived().correspondencePool->size(); }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline float kdtree_get_pt(const size_t idx, int dim) const
        {
            if ((*(derived().poolIdxIt))[idx] == derived().correspondencePool->end())
                throw InvalidPoolIteratorException("Invalid pool iterator");
            if (dim == 0) return (*(derived().poolIdxIt))[idx]->pt1.x;
            else return (*(derived().poolIdxIt))[idx]->pt1.y;
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
        CoordinateInterface coordInteraface = CoordinateInterface();

        typedef PointCloudAdaptor<CoordinateInterface > PC2KD;
        std::unique_ptr<PC2KD> pc2kd; // The adaptor
                                      // construct a kd-tree index:
        typedef KDTreeSingleIndexDynamicAdaptor<
            L2_Simple_Adaptor<float, PC2KD >,
            PC2KD,
            2 /* dim */
        > coordinateKDTree;
        std::unique_ptr<coordinateKDTree> index;
    public:

        keyPointTree(std::list<CoordinateProps> *correspondencePool_,
            std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt_)
        {
            coordInteraface.correspondencePool = correspondencePool_;
            coordInteraface.poolIdxIt = poolIdxIt_;
            pc2kd.reset(new PC2KD(coordInteraface));
        }

        int buildInitialTree()
        {
            try
            {
                index.reset(new coordinateKDTree(2 /*dim*/, *pc2kd, KDTreeSingleIndexAdaptorParams(20 /* max leaf */)));
            }
            catch (InvalidPoolIteratorException const& ex)
            {
                std::cout << "Exception: " << ex.what() << std::endl;
                std::cout << "Unable to build KD-tree!" << std::endl;
                return -1;
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

        int resetTree(std::list<CoordinateProps> *correspondencePool_,
            std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt_)
        {
            coordInteraface.correspondencePool = correspondencePool_;
            coordInteraface.poolIdxIt = poolIdxIt_;
            pc2kd.reset(new PC2KD(coordInteraface));
            try
            {
                index.reset(new coordinateKDTree(2 /*dim*/, *pc2kd, KDTreeSingleIndexAdaptorParams(20 /* max leaf */)));
            }
            catch (InvalidPoolIteratorException const& ex)
            {
                std::cout << "Exception: " << ex.what() << std::endl;
                std::cout << "Unable to build KD-tree!" << std::endl;
                return -1;
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

        int addElements(size_t firstIdx, size_t length)
        {
            //The indices within this range must be continous and stored in the correspondence pool and the index map(poolIdxIt_) must be valid
            try
            {
                index->addPoints(firstIdx, firstIdx + length - 1);
            }
            catch (InvalidPoolIteratorException const& ex)
            {
                std::cout << "Exception: " << ex.what() << std::endl;
                std::cout << "Unable to add elements to the KD-tree!" << std::endl;
                return -1;
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

        void removeElements(size_t idx)
        {
            index->removePoint(idx);
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
            catch (InvalidPoolIteratorException const& ex)
            {
                std::cout << "Exception: " << ex.what() << std::endl;
                std::cout << "Unable to perform a knn search!!" << std::endl;
                delete[] indices;
                delete[] distances;
                return 0;
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
            RadiusResultSet<float, size_t> resultSet(radius * radius, result);
            float queryPt_[2];
            queryPt_[0] = queryPt.x;
            queryPt_[1] = queryPt.y;
            try
            {
                index->findNeighbors(resultSet, queryPt_, nanoflann::SearchParams());
            }
            catch (InvalidPoolIteratorException const& ex)
            {
                std::cout << "Exception: " << ex.what() << std::endl;
                std::cout << "Unable to perform a radius search!" << std::endl;
                return 0;
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

    keyPointTreeInterface::keyPointTreeInterface(std::list<CoordinateProps> *correspondencePool_,
        std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt_)
    {
        treePtr = new keyPointTree(correspondencePool_, poolIdxIt_);
    }

    int keyPointTreeInterface::buildInitialTree()
    {
        return ((keyPointTree*)treePtr)->buildInitialTree();
        //static_cast<keyPointTree*>(treePtr)->buildInitialTree();
    }

    int keyPointTreeInterface::resetTree(std::list<CoordinateProps> *correspondencePool_,
        std::unordered_map<size_t, std::list<CoordinateProps>::iterator> *poolIdxIt_)
    {
        if (!treePtr)
        {
            treePtr = new keyPointTree(correspondencePool_, poolIdxIt_);
        }
        else
        {
            return ((keyPointTree*)treePtr)->resetTree(correspondencePool_, poolIdxIt_);
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

    int keyPointTreeInterface::addElements(size_t firstIdx, size_t length)
    {
        return ((keyPointTree*)treePtr)->addElements(firstIdx, length);
    }

    void keyPointTreeInterface::removeElements(size_t idx)
    {
        ((keyPointTree*)treePtr)->removeElements(idx);
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
        killTree();
    }

}
