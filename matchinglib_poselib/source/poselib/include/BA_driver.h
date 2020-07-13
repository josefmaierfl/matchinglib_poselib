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
 FILE: BA_driver.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++

 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: March 2014

 LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This source file provides an interface to the sba - Generic Sparse Bundle Adjustment Package
			  Based on Levenberg-Marquardt Algorithm. Most of the code is from the demo that comes with sba.
**********************************************************************************************************/

#pragma once

/* If 1, debug information is available
 * if 0, no debug information is available*/
#define BA_DEBUG 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <math.h>
#if BA_DEBUG
#include <time.h>
#endif

#include "sba.h"

#include "imgproj.h"

#include "poselib/poselib_api.h"

namespace poselib
{

/* --------------------------- Defines --------------------------- */

#define COST_LEASTSQUARES		0
#define COST_PSEUDOHUBER		1

static const double ROBUST_THRESH = 0.005; //Threshold for the different cost functions (e.g. pseudo-huber)
										   //in image coordinates (use a different threshold in the functions
										   //when using camera coordinates)

#define SBA_MAX_REPROJ_ERROR    4.0 // max motion only reprojection error

#define BA_NONE                 -1
#define BA_MOTSTRUCT            0
#define BA_MOT                  1
#define BA_STRUCT               2
#define BA_MOT_MOTSTRUCT        3


typedef struct BAinfo {
		double InitSumSquaredErr; //Initial squared norm errors to the image point measurements -> ||e0||^2
		double SlnSumSquaredErr; //Final squared norm errors to the image point measurements -> ||e||^2
		double FinalSumNormJacErr; //Norm of transposed Jacobi matrix multiplied with the error -> ||(J^T)*e||
		double SumSquaredParmCorrTerms; //Final squared norm of the parameter correction vector -> ||sigma||^2
		double DumpTermDivJacTJacMax; /* Final Jacobi matrix damping term devided by maximum diagonal
									   * term of the Jacobi normal matrix ->mu/max_k(((J^T)J)_kk) */
		double numIterations; //Total number of iterations of the SBA
		double terminatingReason; /* Reason for terminating:
								   * 1: stopped by small ||(J(p)^T)e||
								   * 2: stopped by small norm of parameter correction vector (||sigma||)
								   * 3: stopped by maximum iterations
								   * 4: stopped by small relative reduction in ||e||
								   * 5: stopped by small ||e||
								   * 6: Stopped due to excessive failed attemts to increase damping for getting
								   *	a positive definite normal equations matrix. Typically this indicates a
								   *	programming error in the user-supplied Jacobian.
								   * 7: Stopped due to infinite values in the coordinates of the set of predicted
								   *	projections. This signals a programming error in the user-supplied projection
								   *	function.
							       */
		double numErrCalcs; //Number, how often the reprojection function was called (number of error calculations)
		double numJacCalcs; //Number, how often the Jacobi calculation function was called
		double numNormEquSolve; //Total number of times that the normal equations were solved
		double firstClapackError; // first CLAPACK Error encountered:
								  //       0: no error,
								  //       d: leading minor of order d is not positive definite
								  //      -d: the d-th diagonal element of A is zero (singular matrix)
} BAinfo;

/* --------------------------- Classes --------------------------- */

class POSELIB_API SBAdriver
{
private:
	int expert;
	int analyticjac;
	int verbose;
	double opts[SBA_OPTSSZ];
	int prnt;
	int howto;
	bool fixedcal;
	int nccalib;
	int nconstframes;
	int ncdist;
	int nconst3Dpts;
	double info[SBA_INFOSZ];
	bool useInputVarsAsOutput;
	int costfunc;
	double costThresh;
	bool cam0IntrVarRtFix;

	std::vector<double *> Rquats_out;
	std::vector<double *> trans_out;
	std::vector<double *> intrParms_out;
	std::vector<double *> dist_out;
	double *pts3D_out;
public:


	/* Constructor.
	 *
	 * bool BAfixedcal					Input  -> If the camera intrinsics are provided to perform_sba
	 *											  this variable specifies if they should be fixed or optimized
	 *											  during BA.
	 * int BAhowto						Input  -> Specifies, which parameters should be optimized:
	 *											  BA_MOTSTRUCT: Motion and structure are opimized. Thus,
	 *															the structure elements (3D points), the camera
	 *															rotation, translation and depending
	 *															on other options, the intrinsics and distortion
	 *															are optimized.
	 *											  BA_MOT:		Only the motion is optimized. Thus,
	 *															the camera rotation, translation and depending
	 *															on other options, the intrinsics and distortion
	 *															are optimized.
	 *											  BA_STRUCT:	Only the structure elements are optimized.
	 *											  BA_MOT_MOTSTRUCT: In the first place only the motion (see above)
	 *															is optimized. But if the error after optimization
	 *															is too large, both, the motion and structure
	 *															(see above) are optimized.
	 * int costfunction					Input  -> Specifies, which cost function should be used during BA:
	 *											  COST_LEASTSQUARES: The original least squares cost function is
	 *																 used.
	 *											  COST_PSEUDOHUBER: The robust pseudo-huber cost function is used.
	 *																This cost function is computational more
	 *																expensive because the least squares cost
	 *																function is part of the core algorithm of SBA
	 *																and is performed anyway - this has to be
	 *																reversed while applying the new cost function.
	 * double cost_tresh				Input  -> The threshold for the cost function. The default value should only
	 *											  be used when working in image coordinates and with the pseudo-huber
	 *											  cost function. Otherwise, please use a different threshold.
	 * bool BAuseInputVarsAsOutput		Input  -> If true, the results from BA are written back to the input
	 *											  variables of the function perform_sba. If false, the results
	 *											  from SBA can be read via the different get-methods. In this
	 *											  case, be careful, because the memory which holds the
	 *											  results from BA is deallocated after the lifetime of this
	 *											  object.
	 * int SlnPrnt						Input  -> Specifies, which results should be written back to the input
	 *											  variables (if BAuseInputVarsAsOutput = true) or which results
	 *											  are available through the get-methods (if
	 *											  BAuseInputVarsAsOutput = false). The options are the same as for
	 *											  BAhowto. This option can help saving memory or prevents
	 *											  writing back data.
	 * int BAnccalib					Input  -> Specifies which intrinsics should be kept fixed during BA (if
	 *											  intrinsics are provided to perform_sba):
	 *											  0: all parameters are free
     *											  1: skew is fixed to its initial value, all other parameters vary
	 *												 (i.e. fu, u0, v0, ar)
     *											  2: skew and aspect ratio are fixed to their initial values, all
	 *												 other parameters vary (i.e. fu, u0, v0)
     *											  3: meaningless
     *											  4: skew, aspect ratio and principal point are fixed to their
	 *												 initial values, only the focal length varies (i.e. fu)
     *											  5: all intrinsics are kept fixed to their initial values
     *											  >5: meaningless
     *											  Used only when calibration varies among cameras
	 * int BAncdist						Input  -> Specifies which distortion coeffitients of Bouguet's model
	 *											  should be kept fixed during BA (if distortion parameters are
	 *											  provided to perform_sba):
	 *											  0: all parameters are free
     *											  1: 6th order radial distortion term (kc[4]) is fixed
     *											  2: 6th order radial distortion and one of the tangential
	 *												 distortion terms (kc[3]) are fixed
     *											  3: 6th order radial distortion and both tangential distortion
	 *												 terms (kc[3], kc[2]) are fixed [i.e., only 2nd & 4th order
	 *												 radial dist.]
     *											  4: 4th & 6th order radial distortion terms and both tangential
	 *												 distortion ones are fixed [i.e., only 2nd order radial dist.]
     *											  5: all distortion parameters are kept fixed to their initial
	 *												 values
     *											  >5: meaningless
     *											  Used only when calibration varies among cameras and distortion
	 *											  is to be estimated
	 * int BAnconstframes				Input  -> Number of frames or cameras beginning from the first for which
	 *											  their parameters (exrinsic and intrinsic) should be kept fixed.
	 *											  This is, e.g., useful when the world's coordinate frame is
	 *											  aligned with taht of the first camera, therefore the (projective)
	 *											  first camera matrix should be kept fixed to [I|0].
	 * int BAnconst3Dpts				Input  -> Number of structure elements (3D points) beginning from the first
	 *											  should be kept fixed. This is, e.g., useful if a few structure
	 *											  elements are known to be exact.
	 * bool BAcam0IntrVarRtFix          Input  -> If true [Default = false] and BAnconstframes = 0, R & t are kept fixed
	 *                                            for the first camera but the intrinsics and distortion coefficients
	 *                                            are allowed to vary.
	 */
	SBAdriver(bool BAfixedcal = true,
			  int BAhowto = BA_MOTSTRUCT,
			  int costfunction = COST_PSEUDOHUBER,
			  double cost_tresh = ROBUST_THRESH,
			  bool BAuseInputVarsAsOutput = true,
			  int SlnPrnt = BA_MOT,
			  int BAnccalib = 1,
			  int BAncdist = 0,
			  int BAnconstframes = 1,
			  int BAnconst3Dpts = 0,
			  bool BAcam0IntrVarRtFix = false)
	: fixedcal(BAfixedcal),
	  howto(BAhowto),
	  costfunc(costfunction),
	  costThresh(cost_tresh),
	  useInputVarsAsOutput(BAuseInputVarsAsOutput),
	  prnt(SlnPrnt),
	  nccalib(BAnccalib),
	  ncdist(BAncdist),
	  nconstframes(BAnconstframes),
	  nconst3Dpts(BAnconst3Dpts),
      cam0IntrVarRtFix(BAcam0IntrVarRtFix),
	  expert(1), //If 1, the user specific expert drivers are used (all data is supplied at once to the functions for
				 //calculating the jacobian and the projections). If 0, the functions are called for every single
				 //structure element.
	  analyticjac(1), //if 1, a user-specified function for calculating the jacobian matrix (analytically) is called.
					  //Otherwise, it is calculated numerically.
	  verbose(0) //Verbosity level: The higher the number the more debug information is printed to the std-output
	{
		opts[0] = SBA_INIT_MU; //multiplication factor (tau) for the damping term (u = max_i=1...n(A_ii) with the
							   //normal equation A_ii = J^T*J) of the Levenberg-Marquardt algorithm
		opts[1] = SBA_STOP_THRESH; //algorithm terminates if the magnitude of the gradient drops below this threshold
		opts[2] = SBA_STOP_THRESH; //algorithm terminates if the relative magnitude of the parameter correction
								   //vector (sigma) drops below a threshold involving this parameter
		opts[3] = SBA_STOP_THRESH; //algorithm terminates if the magnitude of the residual (epsilon) drops below this
								   //threshold
		/* uncomment the following line in the function SBAdriver::perform_sba to force termination
		if the average reprojection error drops below 0.05: opts[3]=0.05*numprojs; */
		//opts[4] = 0.0; //algorithm terminates if the relative reduction in the magnitude of the residual (epsilon)
					   //drops below this threshold
		//uncomment to force termination if the relative reduction in the RMS reprojection error drops below 1E-05:
		opts[4]=1E-05;
	}

	//Destructor
	~SBAdriver()
	{
		for(size_t j = 0; j < Rquats_out.size(); ++j)
			free(Rquats_out.at(j));
		for(size_t j = 0; j < trans_out.size(); ++j)
			free(trans_out.at(j));
		for(size_t j = 0; j < intrParms_out.size(); ++j)
			free(intrParms_out.at(j));
		for(size_t j = 0; j < dist_out.size(); ++j)
			free(dist_out.at(j));
		Rquats_out.clear();
		trans_out.clear();
		intrParms_out.clear();
		dist_out.clear();
	}

	void setFixedIntrCal(bool BAfixedcal)
	{
		fixedcal = BAfixedcal;
	}

	void setBAmethode(int BAhowto)
	{
		howto = BAhowto;
	}

	void setOutputData(int SlnPrnt)
	{
		prnt = SlnPrnt;
	}

	void setFixedIntrPars(int BAnccalib)
	{
		nccalib = BAnccalib;
	}

	void setFixedDistPars(int BAncdist)
	{
		ncdist = BAncdist;
	}

	void setFixedCamsAndPoints(int BAnconstframes, int BAnconst3Dpts = 0)
	{
		nconstframes = BAnconstframes;
		nconst3Dpts = BAnconst3Dpts;
	}

	void allowCam0VaryIntrinsics(){
        cam0IntrVarRtFix = true;
        nconstframes = 0;
	}

	void changeSBAmethode(int BAexpert, int BAanalyticjac = 1)
	{
		expert = BAexpert;
		analyticjac = BAanalyticjac;
	}

	void setVerbosityLevel(int BAverbose)
	{
		verbose = BAverbose;
	}

	void setBAoptions(double BAopts[SBA_OPTSSZ])
	{
		for(size_t i = 0; i < SBA_OPTSSZ; ++i)
			opts[i] = BAopts[i];
	}

	/* Returns information to the already performed BA.
	 * For info on the output structure see the typedef of BAinfo
	 */
	BAinfo getFinalSBAinfo()
	{
		BAinfo sln_info;

		sln_info.InitSumSquaredErr = info[0];
		sln_info.SlnSumSquaredErr = info[1];
		sln_info.FinalSumNormJacErr = info[2];
		sln_info.SumSquaredParmCorrTerms = info[3];
		sln_info.DumpTermDivJacTJacMax = info[4];
		sln_info.numIterations = info[5];
		sln_info.terminatingReason = info[6];
		sln_info.numErrCalcs = info[7];
		sln_info.numJacCalcs = info[8];
		sln_info.numNormEquSolve = info[9];
		sln_info.firstClapackError = info[10];

		return sln_info;
	}

	int getRotQuats(std::vector<double *> *Rquats)
	{
		if(useInputVarsAsOutput || Rquats_out.size() == 0)
			return -1;
		*Rquats = Rquats_out;
		return 0;
	}

	int getTranslation(std::vector<double *> *trans)
	{
		if(useInputVarsAsOutput || trans_out.size() == 0)
			return -1;
		*trans = trans_out;
		return 0;
	}

	int getCamIntrinsics(std::vector<double *> *intrParms)
	{
		if(useInputVarsAsOutput || intrParms_out.size() == 0)
			return -1;
		*intrParms = intrParms_out;
		return 0;
	}

	int getDistortionCoefs(std::vector<double *> *dist)
	{
		if(useInputVarsAsOutput || dist_out.size() == 0)
			return -1;
		*dist = dist_out;
		return 0;
	}

	//Performs Bundle Adjustment (BA) by a Sparse Bundleadjustment Algorithm (SBA)
	int perform_sba(std::vector<double *> & Rquats,
			   std::vector<double *> & trans,
			   std::vector<double *> pts2D,
			   std::vector<int> num2Dpts,
			   double *pts3D,
			   int numpts3D,
			   std::vector<char *> *mask2Dpts = NULL,
			   std::vector<double *> *intrParms = NULL,
			   std::vector<double *> *dist = NULL,
			   std::vector<double *> *cov2Dpts = NULL);
};

/* --------------------- Function prototypes --------------------- */

//Calculates the weight using the pseudo-huber cost function.
double costPseudoHuber(const double d, const double thresh);

//#ifdef __cplusplus
//extern "C" {
//#endif
///* in imgproj.c */
//extern void calcImgProj(double a[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]);
//extern void calcImgProjNoK(double qr0[4],double v[3],double t[3],double M[3], double n[2]);
//extern void calcImgProjFullR(double a[5], double qr0[4], double t[3], double M[3], double n[2]);
//extern void calcImgProjFullRnoK(double qr0[4],double t[3],double M[3], double n[2]);
//extern void calcImgProjJacKRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11], double jacmS[2][3]);
//extern void calcImgProjJacKRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11]);
//extern void calcImgProjJacS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);
//extern void calcImgProjJacSnoK(double qr0[4],double v[3],double t[3], double M[3],double jacmS[2][3]);
//extern void calcImgProjJacRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6], double jacmS[2][3]);
//extern void calcImgProjJacRTSnoK(double qr0[4],double v[3],double t[3], double M[3],double jacmRT[2][6],double jacmS[2][3]);
//extern void calcImgProjJacRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6]);
//extern void calcImgProjJacRTnoK(double qr0[4],double v[3],double t[3], double M[3],double jacmRT[2][6]);
//extern void calcDistImgProj(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]);
//extern void calcDistImgProjFullR(double a[5], double kc[5], double qr0[4], double t[3], double M[3], double n[2]);
//extern void calcDistImgProjJacKDRTS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16], double jacmS[2][3]);
//extern void calcDistImgProjJacKDRT(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16]);
//extern void calcDistImgProjJacS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);
//
//#ifdef __cplusplus
//}
//#endif

}
