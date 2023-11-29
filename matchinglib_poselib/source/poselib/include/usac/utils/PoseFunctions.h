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

#pragma once

#include <opengv/math/cayley.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <memory>
#include <random>

namespace PoseTools
{
	unsigned int getRotError(const std::vector<unsigned int>& test, unsigned int numPoints, std::vector<double>& errs,
		const std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter>& adapter,
		opengv::rotation_t model, double threshold);

	unsigned int getTransError(const std::vector<unsigned int>& test, unsigned int numPoints, std::vector<double>& errs,
		const std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter>& adapter,
		opengv::translation_t model, double threshold);

	unsigned int getNoMotError(const std::vector<unsigned int>& test, unsigned int numPoints, std::vector<double>& errs,
		const std::shared_ptr<opengv::relative_pose::CentralRelativeAdapter>& adapter, double threshold);

	void getPerturbedRotation(
		opengv::rotation_t &rotation,
		std::mt19937 &mt,
		const double &amplitude);

	double getSampsonError(const std::vector<double> & model, double *input_points_denorm_, unsigned int pt_index);
}
