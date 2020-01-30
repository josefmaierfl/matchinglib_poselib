/******************************************************************************
* FILENAME:     PointCloudOperation
* PURPOSE:      %{Cpp:License:ClassName}
* AUTHOR:       jungr - Roland Jung
* MAIL:         Roland.Jung@ait.ac.at
* VERSION:      v1.0.0
*
*  Copyright (C) 2016 Austrian Institute of Technologies GmbH - AIT
*  All rights reserved. See the LICENSE file for details.
******************************************************************************/

#ifndef POINTCLOUDOPERATION_HPP
#define POINTCLOUDOPERATION_HPP
#include <nanoflann.hpp>

// USAGE:
//  - std::vector<Point2D<float>> vec2d;
//  - SmartPointCloud<float, Cloud2D<float>, Point2D<float>, 2> smartCloud2d(vec2d);


namespace PointCloud
{
  using namespace nanoflann;

  int const ID_UNDEF = -1;

  template <typename T>
  struct Point3D
  {
    T  x,y,z;
    int id; // to determine the original position in the input vector
    Point3D() : x(), y(), z(), id(ID_UNDEF) {}
    Point3D(T const a, T const b , T const c, int const i=ID_UNDEF) : x(a), y(b), z(c), id(i) {}
  };

  template <typename T>
  struct Point2D
  {
    T  x,y;
    int id; // to determine the original position in the input vector
    Point2D() : x(), y(), id(ID_UNDEF) {}
    Point2D(T const a, T const b , int const i=ID_UNDEF) : x(a), y(b), id(i) {}
  };

  // 2D - CLOUD
  // This is an exampleof a custom data set class
  template <typename T>
  struct Cloud2D_Adaptor
  {

    typedef Point2D<T> Point;
    std::vector< Point > pts;

    Cloud2D_Adaptor(std::vector<Point2D<T> > vecPts) : pts(vecPts) {}
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
      return pts.size();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t size) const
    {
      const T d0=p1[0]-pts[idx_p2].x;
      const T d1=p1[1]-pts[idx_p2].y;
      return d0*d0+d1*d1;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, int dim) const
    {
      if (dim==0)
      {
        return pts[idx].x;
      }
      else
      {
        return pts[idx].y;
      }
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &bb) const
    {
      return false;
    }

  };

  // This is an exampleof a custom data set class
  template <typename T>
  struct Cloud3D_Adaptor
  {

    typedef Point3D<T> Point;
    std::vector< Point > pts;

    Cloud3D_Adaptor(std::vector<Point3D<T> > vecPts) : pts(vecPts) {}
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
      return pts.size();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t size) const
    {
      const T d0=p1[0]-pts[idx_p2].x;
      const T d1=p1[1]-pts[idx_p2].y;
      const T d2=p1[2]-pts[idx_p2].z;
      return d0*d0+d1*d1+d2*d2;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, int dim) const
    {
      if (dim==0)
      {
        return pts[idx].x;
      }
      else if (dim==1)
      {
        return pts[idx].y;
      }
      else
      {
        return pts[idx].z;
      }
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &bb) const
    {
      return false;
    }

  };


  template<typename T=float, class TCloud_Adapter=Cloud3D_Adaptor<T>, class TPoint=Point3D<T>, int Tdim = 3 >
  class SmartPointCloud
  {
    public:
      // construct a kd-tree index:
      typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<T, TCloud_Adapter> ,TCloud_Adapter, Tdim> my_kd_tree_t;

      // creates a point cloud on which the operations can be applied.
      // HINT: if the input was the lenght 10, then 20 elements are in the pc -> 10 of them are default initialized and trash!
      // that's why the Point3D has an ID which can be used to determine the original index in the input vector!
      SmartPointCloud(std::vector<TPoint >&  pts) : mPC(pts), mpIndex(0)
      {
        mpIndex = new my_kd_tree_t(Tdim , mPC, KDTreeSingleIndexAdaptorParams(10));
        mpIndex->buildIndex();
      }
      ~SmartPointCloud()
      {
        if(mpIndex)
        {
          delete mpIndex;
        }
      }
      unsigned findInRadius(TPoint const& pt, float const search_radius, std::vector<std::pair<size_t,T> >& ret_matches)
      {
        nanoflann::SearchParams params;
        return mpIndex->radiusSearch((T*)&pt,search_radius, ret_matches, params);
      }

      void findKnn(TPoint const& pt, size_t const num_results, std::vector<size_t>& ret_index, std::vector<T>& out_dist_sqr)
      {
        nanoflann::KNNResultSet<T> resultSet(num_results);
        size_t ret;
        T out_dist;
        resultSet.init(&ret, &out_dist);
        mpIndex->findNeighbors(resultSet, &pt.x, nanoflann::SearchParams(10));
        ret_index.push_back(ret);
        out_dist_sqr.push_back(out_dist);
      }

      void findKnn(TPoint const& pt, size_t const num_results, size_t* arr_ret_index,  T* arr_out_dist_sqr)
      {
        nanoflann::KNNResultSet<T> resultSet(num_results);
        resultSet.init(arr_ret_index, arr_out_dist_sqr);
        mpIndex->findNeighbors(resultSet, &pt.x, nanoflann::SearchParams(10));
      }

      std::vector<TPoint >& getPoints()
      {
        return mPC.pts;
      }

    private:

      TCloud_Adapter mPC;
      // mIndex;
      my_kd_tree_t* mpIndex;

  };







} // namespace PointCloud




#endif // POINTCLOUDOPERATION_HPP
