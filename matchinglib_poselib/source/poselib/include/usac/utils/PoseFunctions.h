#pragma once

#include <opengv/math/cayley.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <memory>

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
		opengv::rotation_t & rotation,
		double amplitude);

	double getSampsonError(const std::vector<double> & model, double *input_points_denorm_, unsigned int pt_index);
}
