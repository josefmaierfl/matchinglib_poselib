#pragma once

#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/Indices.hpp>

opengv::complexEssentials_t fivept_stewenius_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter);

opengv::complexEssentials_t fivept_stewenius_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter,
	const opengv::Indices & indices);

opengv::essentials_t fivept_nister_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter);

opengv::essentials_t fivept_nister_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter,
	const opengv::Indices & indices);

double computePseudoHuberWeight(const opengv::bearingVector_t  & f, const opengv::bearingVector_t  & fprime, const opengv::essential_t & E, const double & th);


