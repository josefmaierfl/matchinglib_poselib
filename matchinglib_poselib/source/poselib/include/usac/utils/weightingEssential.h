/**********************************************************************************************************
FILE: weightingEssential.h

PLATFORM: Windows 7, MS Visual Studio 2010

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: June 2017

LOCATION: TechGate Vienna, Donau-City-Stra?e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions for linear refinement of Essential matrices using weights
**********************************************************************************************************/

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

double computeTorrWeight(const opengv::bearingVector_t  & f, const opengv::bearingVector_t  & fprime, const opengv::essential_t & E);

opengv::essential_t eightpt_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter);

opengv::essential_t eightpt_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter,
	const opengv::Indices & indices);


