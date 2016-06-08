/*  Copyright (c) 2013, Bo Li, prclibo@gmail.com
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the copyright holder nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/****************************************************************************
 *							IMPORTANT NOTICE								*
 ****************************************************************************
 *																			*
 * Most of this code was completele rewritten by Josef Maier				*
 * (josef.maier.fl@ait.ac.at) as the original code was not working			*
 * correctly. Moreover, additional functionalities (like ARRSAC or			*
 * refinement procedures) were added. Most interfaces also differ			*
 * completely to the original ones.											*
 *																			*
 * For further information please contact Josef Maier, AIT Austrian			*
 * Institute of Technology.													*
 *																			*
 ****************************************************************************/


#ifndef FIVE_POINT_HPP
#define FIVE_POINT_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>

#define ARRSAC 3

using namespace cv; 

bool findEssentialMat(OutputArray Essential, InputArray points1, InputArray points2, /*double focal = 1.0, Point2d pp = Point2d(0, 0), */
					int method = CV_RANSAC, 
					double prob = 0.999, double threshold = 1, OutputArray mask = noArray(), bool lesqu = false, 
					void (*refineEssential)(cv::InputArray points1, cv::InputArray points2, cv::InputArray E_init, 
											cv::Mat & E_refined, double th, unsigned int iters, bool makeClosestE, 
											double *sumSqrErr_init, double *sumSqrErr, 
											cv::OutputArray errors, cv::InputOutputArray mask, int model) = NULL); 

void decomposeEssentialMat( const Mat & E, Mat & R1, Mat & R2, Mat & t ); 

int recoverPose( const Mat & E, InputArray points1, InputArray points2, Mat & R, Mat & t, Mat & Q3D,
					/*double focal = 1.0, Point2d pp = Point2d(0, 0), */
					InputOutputArray mask = noArray(), const double dist = 50.0, cv::InputArray t_only = cv::noArray()); 

inline bool isZero(double d)
{
    //Decide if determinants, etc. are too close to 0 to bother with
    const double EPSILON = 1e-3;
    return (d<EPSILON) && (d>-EPSILON);
}

#endif
