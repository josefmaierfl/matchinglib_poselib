/**********************************************************************************************************
FILE: nanoflannInterface.h

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: October 2017

LOCATION: TechGate Vienna, Donau-City-Stra?e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions to search keypoint positions within a tree built using nanoflann
**********************************************************************************************************/

#pragma once

#include "poselib/glob_includes.h"
#include "nanoflann.hpp"
#include "poselib/stereo_pose_refinement.h"
#include <list>
#include <memory>
#include <unordered_map>

using namespace nanoflann;

namespace poselib
{

	/* --------------------------- Defines --------------------------- */


	/* --------------------- Function prototypes --------------------- */


	/* ---------------------- Classes & Structs ---------------------- */
	
	// Data structure for the tree
	struct CoordinateInterface
	{
		std::list<CoordinateProps> *correspondencePool;
		std::unordered_map<unsigned int, std::list<CoordinateProps>::iterator> *poolIdxIt;
	};

	// And this is the "dataset to kd-tree" adaptor class:
	template <typename Derived>
	struct PointCloudAdaptor
	{
		//typedef typename Derived::coord_t coord_t;

		const Derived &obj; //!< A const ref to the data set origin

							/// The constructor that sets the data set source
		PointCloudAdaptor(const Derived &obj_) : obj(obj_) { }

		/// CRTP helper method
		inline const Derived& derived() const { return obj; }

		// Must return the number of data points
		inline size_t kdtree_get_point_count() const { return derived().correspondencePool->size(); }

		// Returns the dim'th component of the idx'th point in the class:
		// Since this is inlined and the "dim" argument is typically an immediate value, the
		//  "if/else's" are actually solved at compile time.
		inline float kdtree_get_pt(const size_t idx, int dim) const
		{
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
		CoordinateInterface coordInteraface;

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
			std::unordered_map<unsigned int, std::list<CoordinateProps>::iterator> *poolIdxIt_)
		{
			coordInteraface.correspondencePool = correspondencePool_;
			coordInteraface.poolIdxIt = poolIdxIt_;
			pc2kd.reset( new PC2KD(coordInteraface));
		}

		void buildInitialTree()
		{
			index.reset(new coordinateKDTree(2 /*dim*/, *pc2kd, KDTreeSingleIndexAdaptorParams(20 /* max leaf */)));
		}

		void resetTree(std::list<CoordinateProps> *correspondencePool_,
			std::unordered_map<unsigned int, std::list<CoordinateProps>::iterator> *poolIdxIt_)
		{
			coordInteraface.correspondencePool = correspondencePool_;
			coordInteraface.poolIdxIt = poolIdxIt_;
			pc2kd.reset(new PC2KD(coordInteraface));
			index.reset(new coordinateKDTree(2 /*dim*/, *pc2kd, KDTreeSingleIndexAdaptorParams(20 /* max leaf */)));
		}

		void addElements(size_t firstIdx, size_t length)
		{
			//The indices within this range must be continous and stored in the correspondence pool and the index map(poolIdxIt_) must be valid
			index->addPoints(firstIdx, firstIdx + length - 1);
		}

		void removeElements(size_t idx)
		{
			index->removePoint(idx);
		}

		size_t knnSearch(cv::Point2f queryPt, size_t knn, std::vector<std::pair<size_t, float>> & result)
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
			index->findNeighbors(resultSet, &queryPt_[0], nanoflann::SearchParams(10));
			nr_results = resultSet.size();
			result.reserve(nr_results);
			for (size_t i = 0; i < nr_results; i++)
			{
				result.push_back(std::make_pair(indices[i], distances[i]));
			}
			delete[] indices;
			delete[] distances;
			return nr_results;
		}

		size_t radiusSearch(cv::Point2f queryPt, float radius, std::vector<std::pair<size_t, float>> & result)
		{
			// Unsorted radius search:
			result.clear();
			RadiusResultSet<float, size_t> resultSet(radius, result);
			float queryPt_[2];
			queryPt_[0] = queryPt.x;
			queryPt_[1] = queryPt.y;
			index->findNeighbors(resultSet, queryPt_, nanoflann::SearchParams());
			//Sort
			std::sort(result.begin(), result.end(), [](std::pair<size_t, float> const & first, std::pair<size_t, float> const & second)
			{
				return first.second < second.second;
			});
			return result.size();
		}
	};


	/* -------------------------- Functions -------------------------- */

}