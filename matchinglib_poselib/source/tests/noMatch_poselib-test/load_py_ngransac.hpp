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

#ifndef MATCHING_LOAD_PY_NGRANSAC_HPP
#define MATCHING_LOAD_PY_NGRANSAC_HPP
#include <iostream>
#include <vector>
#include <memory>

#include "boost/shared_ptr.hpp"
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/dict.hpp>
#include "boost/python/stl_iterator.hpp"

#include "opencv2/core/core.hpp"

namespace bp = boost::python;
using namespace std;

template<typename T>
inline
std::vector< T > py_list_2_vec( const bp::object& iterable )
{
    return std::vector< T >( bp::stl_input_iterator< T >( iterable ),
                             bp::stl_input_iterator< T >( ) );
}


template <class T>
inline
bp::list std_vector_to_py_list(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    bp::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

template<typename T>
cv::Mat vecToMat(std::vector<T> &vec, int nr_rows, int nr_cols){
    cv::Mat data = cv::Mat_<T>(nr_rows, nr_cols);
    for(int i=0; i < nr_cols; i++){
        for(int j=0; j < nr_rows; j++){
            int idx = i * nr_rows + j;
            data.at<T>(i, j) = vec[idx];
        }
    }
    return data;
}

//std::vector<bp::tuple> vec_keypoints_to_vec_py_tuple(std::vector<cv::Point2f> &pts){
//    std::vector<bp::tuple> tuples;
//    for(auto &i: pts){
//        tuples.emplace_back(bp::make_tuple(i.x, i.y));
//    }
//    return tuples;
//}
//
//std::vector<double> mat_to_vector(cv::Mat &data){
//    CV_Assert(data.type() == CV_64FC1);
//    return std::vector<double>(data.begin<double>(), data.end<double>());
//}

struct Py_input{
        std::string model_file_name;
        double threshold;
        bp::list pts1, pts2;
        bp::list K1, K2;
    };

struct Py_output{
    unsigned int nr_inliers;
    cv::Mat mask;
    cv::Mat model;
};

class ComputeInstance;

class ComputeServer{
public:
    void transferModel(ComputeInstance&, bp::list model, bp::list inlier_mask, unsigned int nr_inliers);
};

class ComputeInstance{
public:
    ComputeInstance(ComputeServer&);
    virtual ~ComputeInstance() = default;

    virtual void compute() = 0;
    void transferModel(bp::list model, bp::list inlier_mask, unsigned int nr_inliers);

private:
    ComputeServer& _server;
};


#endif //MATCHING_LOAD_PY_NGRANSAC_HPP
