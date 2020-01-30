/**********************************************************************************************************
FILE: weightingEssential.cpp

PLATFORM: Windows 7, MS Visual Studio 2010

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: June 2017

LOCATION: TechGate Vienna, Donau-City-Stra?e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functions for linear refinement of Essential matrices using weights
**********************************************************************************************************/


#include "usac/utils/weightingEssential.h"

//#include <opengv/relative_pose/methods.hpp>
//#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
//#include <opengv/triangulation/methods.hpp>
#include <opengv/relative_pose/modules/main.hpp>

#include "BA_driver.h"


void SampsonL1_Eigen(const opengv::bearingVector_t & x1, const opengv::bearingVector_t & x2, const opengv::essential_t & E, double & denom1, double & num);

opengv::complexEssentials_t fivept_stewenius_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter)
{
	opengv::Indices idx(adapter.getNumberCorrespondences());
	return fivept_stewenius_weight(adapter, idx);
}

opengv::complexEssentials_t fivept_stewenius_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter,
	const opengv::Indices & indices)
{
	size_t numberCorrespondences = indices.size();
	assert(numberCorrespondences > 4);

	Eigen::MatrixXd Q(numberCorrespondences, 9);
	double weightnorm = 0;
	for (size_t i = 0; i < numberCorrespondences; i++)
	{
		weightnorm += std::pow(adapter.getWeight(indices[i]), 2);
	}
	weightnorm = std::sqrt(weightnorm);
	for (size_t i = 0; i < numberCorrespondences; i++)
	{
		//bearingVector_t f = adapter.getBearingVector1(indices[i]);
		//bearingVector_t fprime = adapter.getBearingVector2(indices[i]);
		//Stewenius' algorithm is computing the inverse transformation, so we simply
		//invert the input here
		opengv::bearingVector_t f = adapter.getBearingVector2(indices[i]);
		opengv::bearingVector_t fprime = adapter.getBearingVector1(indices[i]);
		Eigen::Matrix<double, 1, 9> row;
		double weight = adapter.getWeight(indices[i]);
		/*double denom1, num;
		SampsonL1_Eigen(f, fprime, E, denom1, num);
		const double weight = poselib::costPseudoHuber(num*denom1, th);*/
		row << f[0] * fprime[0], f[1] * fprime[0], f[2] * fprime[0],
			f[0] * fprime[1], f[1] * fprime[1], f[2] * fprime[1],
			f[0] * fprime[2], f[1] * fprime[2], f[2] * fprime[2];
		row *= weight / weightnorm;
		Q.row(i) = row;
	}

	Eigen::JacobiSVD< Eigen::MatrixXd > SVD(Q, Eigen::ComputeFullV);
	Eigen::Matrix<double, 9, 4> EE = SVD.matrixV().block(0, 5, 9, 4);
	opengv::complexEssentials_t complexEssentials;
	opengv::relative_pose::modules::fivept_stewenius_main(EE, complexEssentials);
	return complexEssentials;
}

opengv::essentials_t fivept_nister_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter)
{
	opengv::Indices idx(adapter.getNumberCorrespondences());
	return fivept_nister_weight(adapter, idx);
}

opengv::essentials_t fivept_nister_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter,
	const opengv::Indices & indices)
{
	size_t numberCorrespondences = indices.size();
	assert(numberCorrespondences > 4);

	Eigen::MatrixXd Q(numberCorrespondences, 9);
	double weightnorm = 0;
	for (size_t i = 0; i < numberCorrespondences; i++)
	{
		weightnorm += std::pow(adapter.getWeight(indices[i]), 2);
	}
	weightnorm = std::sqrt(weightnorm);

	for (size_t i = 0; i < numberCorrespondences; i++)
	{
		//bearingVector_t f = adapter.getBearingVector1(indices[i]);
		//bearingVector_t fprime = adapter.getBearingVector2(indices[i]);
		//Nister's algorithm is computing the inverse transformation, so we simply
		//invert the input here
		opengv::bearingVector_t f = adapter.getBearingVector2(indices[i]);
		opengv::bearingVector_t fprime = adapter.getBearingVector1(indices[i]);
		Eigen::Matrix<double, 1, 9> row;
		double weight = adapter.getWeight(indices[i]);
		row << f[0] * fprime[0], f[1] * fprime[0], f[2] * fprime[0],
			f[0] * fprime[1], f[1] * fprime[1], f[2] * fprime[1],
			f[0] * fprime[2], f[1] * fprime[2], f[2] * fprime[2];
		row *= weight / weightnorm;
		Q.row(i) = row;
	}

	Eigen::JacobiSVD< Eigen::MatrixXd > SVD(Q, Eigen::ComputeFullV);
	Eigen::Matrix<double, 9, 4> EE = SVD.matrixV().block(0, 5, 9, 4);
	opengv::essentials_t essentials;
	opengv::relative_pose::modules::fivept_nister_main(EE, essentials);

	return essentials;
}

/* Calculates the Sampson L1-distance for a point correspondence and returns the invers of the
* denominator (in denom1) and the numerator of the Sampson L1-distance. To calculate the
* Sampson distance, simply multiply these two. For the Sampson error, multiply and square them.
*
* bearingVector_t x1				Input  -> Image projection of the lweft image
* bearingVector_t x2				Input  -> Image projection of the right image
* essential_t E						Input  -> Essential matrix
* double & denom1					Output -> invers of the denominator of the Sampson distance
* double & num						Output -> numerator of the Sampson distance
*
* Return value:					none
*/
void SampsonL1_Eigen(const opengv::bearingVector_t & x1, const opengv::bearingVector_t & x2, const opengv::essential_t & E, double & denom1, double & num)
{
	/*Mat X1, X2;
	if (x1.rows > x1.cols)
	{
	X1 = (Mat_<double>(3, 1) << x1.at<double>(0, 0), x1.at<double>(1, 0), 1.0);
	X2 = (Mat_<double>(3, 1) << x2.at<double>(0, 0), x2.at<double>(1, 0), 1.0);
	}
	else
	{
	X1 = (Mat_<double>(3, 1) << x1.at<double>(0, 0), x1.at<double>(0, 1), 1.0);
	X2 = (Mat_<double>(3, 1) << x2.at<double>(0, 0), x2.at<double>(0, 1), 1.0);
	}*/
	Eigen::Vector3d xpE = x2.transpose() * E;
	//Mat xpE = X2.t() * E;
	xpE = xpE.transpose();
	//xpE = xpE.t();
	num = xpE.dot(x1);
	//num = xpE.dot(X1);
	Eigen::Vector3d Ex1 = E * x1;
	//Mat Ex1 = E * X1;

	double a = Ex1(0) * Ex1(0);
	double b = Ex1(1) * Ex1(1);
	double c = xpE(0) * xpE(0);
	double d = xpE(1) * xpE(1);

	denom1 = 1 / (std::sqrt(a + b + c + d) + 1e-8);
}

double computePseudoHuberWeight(const opengv::bearingVector_t  & f, const opengv::bearingVector_t  & fprime, const opengv::essential_t & E, const double & th)
{
	double denom1, num;
	SampsonL1_Eigen(f, fprime, E, denom1, num);
	const double weight = poselib::costPseudoHuber(num*denom1, th);
	return denom1*weight; /* multiply by 1/dominator of the Sampson L1-distance to eliminate
							  * it from the dominator of the pseudo-huber weight value (which is
							  * actually sqrt(cost_pseudo_huber)/Sampson_L1_distance). This is
							  * necessary because the SVD calculates the least squared algebraic
							  * error of the fundamental matrix (x2^T*F*x1)^2. This algebraic
							  * error is the same as the numerator of the Sampson distance and
							  * should be replaced by the pseudo-huber cost function during SVD.
							  */
}

double computeTorrWeight(const opengv::bearingVector_t  & f, const opengv::bearingVector_t  & fprime, const opengv::essential_t & E)
{
	double rx, ry, ryc, rxc;
	// compute weight (ref: torr dissertation, eqn. 2.25)
	rxc = E.col(0).transpose() * fprime;
	ryc = E.col(1).transpose() * fprime;
	rx = E.row(0) * f;
	ry = E.row(1) * f;

	/*pt = input_points_ + 6 * pt_index;
	rxc = (*model) * (*(pt + 3)) + (*(model + 3)) * (*(pt + 4)) + (*(model + 6));
	ryc = (*(model + 1)) * (*(pt + 3)) + (*(model + 4)) * (*(pt + 4)) + (*(model + 7));
	rx = (*model) * (*(pt)) + (*(model + 1)) * (*(pt + 1)) + (*(model + 2));
	ry = (*(model + 3)) * (*(pt)) + (*(model + 4)) * (*(pt + 1)) + (*(model + 5));*/

	double weight = 1 / sqrt(rxc*rxc + ryc*ryc + rx*rx + ry*ry);
	return weight;
}

opengv::essential_t eightpt_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter)
{
	opengv::Indices idx(adapter.getNumberCorrespondences());
	return eightpt_weight(adapter, idx);
}

opengv::essential_t eightpt_weight(
	const opengv::relative_pose::CentralRelativeWeightingAdapter & adapter,
	const opengv::Indices & indices)
{
	size_t numberCorrespondences = indices.size();
	assert(numberCorrespondences > 7);

	Eigen::MatrixXd A(numberCorrespondences, 9);

	double weightnorm = 0;
	for (size_t i = 0; i < numberCorrespondences; i++)
	{
		weightnorm += std::pow(adapter.getWeight(indices[i]), 2);
	}
	weightnorm = std::sqrt(weightnorm);

	for (size_t i = 0; i < numberCorrespondences; i++)
	{
		//bearingVector_t f1 = adapter.getBearingVector1(indices[i]);
		//bearingVector_t f2 = adapter.getBearingVector2(indices[i]);
		//The eight-point essentially computes the inverse transformation, which is
		//why we invert the input here
		opengv::bearingVector_t f1 = adapter.getBearingVector2(indices[i]);
		opengv::bearingVector_t f2 = adapter.getBearingVector1(indices[i]);
		double weight = adapter.getWeight(indices[i]);

		//normal 8pt algorithm
		A.block<1, 3>(i, 0) = f2[0] * f1.transpose();
		A.block<1, 3>(i, 3) = f2[1] * f1.transpose();
		A.block<1, 3>(i, 6) = f2[2] * f1.transpose();
		//additional weighting:
		A.row(i) *= weight / weightnorm;
	}

	Eigen::JacobiSVD< Eigen::MatrixXd > SVD(
		A,
		Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<double, 9, 1> f = SVD.matrixV().col(8);

	Eigen::MatrixXd F_temp(3, 3);
	F_temp.col(0) = f.block<3, 1>(0, 0);
	F_temp.col(1) = f.block<3, 1>(3, 0);
	F_temp.col(2) = f.block<3, 1>(6, 0);
	opengv::essential_t F = F_temp.transpose();

	Eigen::JacobiSVD< Eigen::MatrixXd > SVD2(
		F,
		Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
	S(0, 0) = SVD2.singularValues()[0];
	S(1, 1) = SVD2.singularValues()[1];

	Eigen::Matrix3d U = SVD2.matrixU();
	Eigen::Matrix3d Vtr = SVD2.matrixV().transpose();

	opengv::essential_t essential = U * S * Vtr;
	return essential;
}