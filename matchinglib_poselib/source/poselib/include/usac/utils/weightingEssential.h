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
/**********************************************************************************************************
FILE: weightingEssential.h

PLATFORM: Windows 7, MS Visual Studio 2010

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: June 2017

LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

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


