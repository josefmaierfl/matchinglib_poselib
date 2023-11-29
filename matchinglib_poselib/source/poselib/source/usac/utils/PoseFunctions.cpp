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

#include "usac/utils/PoseFunctions.h"
#include <opengv/triangulation/methods.hpp>

namespace PoseTools
{
	void getPerturbedRotation(
		opengv::rotation_t &rotation,
		std::mt19937 &mt,
		const double &amplitude)
	{
		opengv::cayley_t cayley = opengv::math::rot2cayley(rotation);
		for (size_t i = 0; i < 3; i++)
		{
			cayley[i] =
				cayley[i] + (static_cast<double>(mt()) / static_cast<double>(mt.max()) - 0.5) * 2.0 * amplitude;
		}
		rotation = opengv::math::cayley2rot(cayley);
	}

	unsigned int getRotError(const std::vector<unsigned int>& test, unsigned int numPoints, std::vector<double>& errs,
		const std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter>& adapter,
		opengv::rotation_t model, double threshold)
	{
		double temp_err;
		unsigned int num_inliers = 0;
		errs.clear(); errs.resize(numPoints);

		for (unsigned int i = 0; i < numPoints; i++)
		{
			opengv::bearingVector_t f1 = adapter->getBearingVector1(test[i]);
			opengv::bearingVector_t f2 = adapter->getBearingVector2(test[i]);

			//unrotate bearing-vector f2
			opengv::bearingVector_t f2_unrotated = model * f2;

			//bearing-vector based outlier criterium (select threshold accordingly):
			//1-(f1'*f2) = 1-cos(alpha) \in [0:2]
			temp_err = 1.0 - (f1.transpose() * f2_unrotated);
			errs[i] = temp_err;

			if (temp_err < threshold)
			{
				++num_inliers;
			}
		}
		return num_inliers;
	}

	unsigned int getTransError(const std::vector<unsigned int>& test, unsigned int numPoints, std::vector<double>& errs,
		const std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter>& adapter,
		opengv::translation_t model, double threshold)
	{
		double temp_err;
		unsigned int num_inliers = 0;
		errs.clear(); errs.resize(numPoints);

		//translation_t translation = model.col(3);
		opengv::rotation_t rotation = opengv::rotation_t::Identity();
		adapter->sett12(model);
		adapter->setR12(rotation);

		opengv::transformation_t inverseSolution;
		inverseSolution.block<3, 3>(0, 0) = rotation.transpose();
		inverseSolution.col(3) = -inverseSolution.block<3, 3>(0, 0)*model;

		Eigen::Matrix<double, 4, 1> p_hom;
		p_hom[3] = 1.0;

		for (unsigned int i = 0; i < numPoints; i++)
		{
			p_hom.block<3, 1>(0, 0) =
				opengv::triangulation::triangulate2(*adapter, test[i]);
			opengv::bearingVector_t reprojection1 = p_hom.block<3, 1>(0, 0);
			opengv::bearingVector_t reprojection2 = inverseSolution * p_hom;
			reprojection1 = reprojection1 / reprojection1.norm();
			reprojection2 = reprojection2 / reprojection2.norm();
			opengv::bearingVector_t f1 = adapter->getBearingVector1(test[i]);
			opengv::bearingVector_t f2 = adapter->getBearingVector2(test[i]);

			//bearing-vector based outlier criterium (select threshold accordingly):
			//1-(f1'*f2) = 1-cos(alpha) \in [0:2]
			double reprojError1 = 1.0 - (f1.transpose() * reprojection1);
			double reprojError2 = 1.0 - (f2.transpose() * reprojection2);
			temp_err = reprojError1 + reprojError2;
			errs[i] = temp_err;

			if (temp_err < threshold)
			{
				++num_inliers;
			}
		}
		return num_inliers;
	}

	unsigned int getNoMotError(const std::vector<unsigned int>& test, unsigned int numPoints, std::vector<double>& errs,
		const std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter>& adapter, double threshold)
	{
		double temp_err;
		unsigned int num_inliers = 0;
		errs.clear(); errs.resize(numPoints);

		for (unsigned int i = 0; i < numPoints; i++)
		{
			opengv::bearingVector_t f1 = adapter->getBearingVector1(test[i]);
			opengv::bearingVector_t f2 = adapter->getBearingVector2(test[i]);

			//bearing-vector based outlier criterium (select threshold accordingly):
			//1-(f1'*f2) = 1-cos(alpha) \in [0:2]
			temp_err = 1.0 - (f1.transpose() * f2);
			errs[i] = temp_err;

			if (temp_err < threshold)
			{
				++num_inliers;
			}
		}
		return num_inliers;
	}

	// ============================================================================================
	// getSampsonError: Calculate the Sampson error of a single data point
	// ============================================================================================
	double getSampsonError(const std::vector<double> & model, double *input_points_denorm_, unsigned int pt_index)
	{
		double rx, ry, rwc, ryc, rxc, r, temp_err;
		//double* model = models_denorm_[modelIndex];
		double* pt;

		// compute sampson error
		pt = input_points_denorm_ + 6 * pt_index;
		rxc = model[0] * (*(pt + 3)) + model[3] * (*(pt + 4)) + model[6];
		ryc = model[1] * (*(pt + 3)) + model[4] * (*(pt + 4)) + model[7];
		rwc = model[2] * (*(pt + 3)) + model[5] * (*(pt + 4)) + model[8];
		r = ((*(pt)) * rxc + (*(pt + 1)) * ryc + rwc);
		rx = model[0] * (*(pt)) + model[1] * (*(pt + 1)) + model[2];
		ry = model[3] * (*(pt)) + model[4] * (*(pt + 1)) + model[5];
		temp_err = r*r / (rxc*rxc + ryc*ryc + rx*rx + ry*ry);
		return temp_err;
	}
}