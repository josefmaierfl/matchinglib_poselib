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
//
// Created by maierj on 7/14/20.
//

#ifndef GENERATEVIRTUALSEQUENCE_READPOINTCLOUDS_H
#define GENERATEVIRTUALSEQUENCE_READPOINTCLOUDS_H

#include "io_data.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/io/pcd_io.h"

struct PointClouds{
    /* Point cloud in the world coordinate system holding all generated 3D points */
    pcl::PointCloud<pcl::PointXYZ>::Ptr staticWorld3DPts;
    /* Every vector element holds the point cloud of a moving object.
     * It also holds the transformed point clouds of already transformed moving objects from older frames */
    std::vector<pcl::PointCloud<pcl::PointXYZ>> movObj3DPtsWorldAllFrames;
};

bool readPointClouds(const std::string &path, const size_t &nrMovObjAllFrames, PointClouds &ptCl) {
    const std::string basename = "pclCloud";//Base name for storing PCL point clouds. A specialization is added at the end of this string

    string filename = concatPath(path, basename);
    string staticWorld3DPtsFileName = filename + "_staticWorld3DPts.pcd";

    ptCl.staticWorld3DPts.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile(staticWorld3DPtsFileName, *ptCl.staticWorld3DPts) == -1) {
        cerr << "Could not read PCL point cloud " << staticWorld3DPtsFileName << endl;
        return false;
    }

    if (nrMovObjAllFrames > 0) {
        ptCl.movObj3DPtsWorldAllFrames.clear();
        ptCl.movObj3DPtsWorldAllFrames.resize(nrMovObjAllFrames);
        for (size_t i = 0; i < nrMovObjAllFrames; ++i) {
            string fname = filename + "_movObj3DPts_" + std::to_string(i) + ".pcd";
            if (pcl::io::loadPCDFile(fname, ptCl.movObj3DPtsWorldAllFrames[i]) == -1) {
                cerr << "Could not read PCL point cloud " << fname << endl;
                return false;
            }
        }
    }

    return true;
}

#endif //GENERATEVIRTUALSEQUENCE_READPOINTCLOUDS_H
