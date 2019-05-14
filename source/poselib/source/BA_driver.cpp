/**********************************************************************************************************
 FILE: BA_driver.cpp

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: March 2014

 LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This source file provides an interface to the sba - Generic Sparse Bundle Adjustment Package
			  Based on Levenberg-Marquardt Algorithm. Most of the code is from the demo that comes with sba.
**********************************************************************************************************/

#include <cmath>

#include "BA_driver.h"

namespace poselib
{

/* --------------------------- Defines --------------------------- */

#define CLOCKS_PER_MSEC (CLOCKS_PER_SEC/1000.0)

#define MAXITER         100
#define MAXITER2        150

#define FULLQUATSZ     4


/* pointers to additional data, used for computed image projections and their jacobians */
struct globs_{
	double *rot0params; /* initial rotation parameters, combined with a local rotation parameterization */
	double *intrcalib; /* the 5 intrinsic calibration parameters in the order [fu, u0, v0, ar, skew],
                      * where ar is the aspect ratio fv/fu.
					  * The parameters for more than 1 camera can be specified.
                      * Used only when calibration is fixed for all cameras;
                      * otherwise, it is null and the intrinsic parameters are
                      * included in the set of motion parameters for each camera
                      */
  int nccalib; /* number of calibration parameters that must be kept constant.
                * 0: all parameters are free 
                * 1: skew is fixed to its initial value, all other parameters vary (i.e. fu, u0, v0, ar) 
                * 2: skew and aspect ratio are fixed to their initial values, all other parameters vary (i.e. fu, u0, v0)
                * 3: meaningless
                * 4: skew, aspect ratio and principal point are fixed to their initial values, only the focal length varies (i.e. fu)
                * 5: all intrinsics are kept fixed to their initial values
                * >5: meaningless
                * Used only when calibration varies among cameras
                */

  int ncdist; /* number of distortion parameters in Bouguet's model that must be kept constant.
               * 0: all parameters are free 
               * 1: 6th order radial distortion term (kc[4]) is fixed
               * 2: 6th order radial distortion and one of the tangential distortion terms (kc[3]) are fixed
               * 3: 6th order radial distortion and both tangential distortion terms (kc[3], kc[2]) are fixed [i.e., only 2nd & 4th order radial dist.]
               * 4: 4th & 6th order radial distortion terms and both tangential distortion ones are fixed [i.e., only 2nd order radial dist.]
               * 5: all distortion parameters are kept fixed to their initial values
               * >5: meaningless
               * Used only when calibration varies among cameras and distortion is to be estimated
               */
  int cnp, pnp, mnp; /* dimensions */

  int calibcams; /* Number of cams for which their calibration parameters are stored in intrcalib (must be 0, 1 or number of all cameras) 
				  * 0: The 2D-projections are assumed to be already in the world coordinate frame (no intrinsics are used)
				  * 1: The provided intrinsics are the same for all cams
				  * n: Every camera has its own fixed intrinsics
				  */

	double *ptparams; /* needed only when bundle adjusting for camera parameters only */
	double *camparams; /* needed only when bundle adjusting for structure parameters only */
	double *imgpts; /* image measurements (2D-projections) - needed only for a different cost function (e.g. pseudo-huber) than least squares */
	double thresh; /* The threshold for the different (e.g. pseudo-huber) cost functions */
} globs;

/* unit quaternion from vector part */
#define _MK_QUAT_FRM_VEC(q, v){                                     \
  (q)[1]=(v)[0]; (q)[2]=(v)[1]; (q)[3]=(v)[2];                      \
  (q)[0]=sqrt(1.0 - (q)[1]*(q)[1] - (q)[2]*(q)[2]- (q)[3]*(q)[3]);  \
}


/* --------------------- Function prototypes --------------------- */

int writeMotToOutput(double *motstruct, double *initrot, int cnp, int nframes, 
					 std::vector<double *> *Rquats_out, std::vector<double *> *trans_out, bool useInpAsOutp);

void writeStructToOutput(double *motstruct, int cnp, int pnp, int nframes, int numpts3D, double *pts3D_out);

int writeIntrinsicsToOutput(double *motstruct, int cnp, int nframes, std::vector<double *> *intrParms_out, bool useInpAsOutp);

int writeDistToOutput(double *motstruct, int cnp, int nframes, std::vector<double *> *dist_out, bool useInpAsOutp);

void convertCostToPseudoHuber(double *ImgProjEsti, double *ImgProjMeas, int mnp, double thresh = ROBUST_THRESH);

inline double sqr(const double var);

/* -------------------------- Functions -------------------------- */



/*
 * multiplication of the two quaternions in q1 and q2 into p
 */
inline static void quatMult(double q1[FULLQUATSZ], double q2[FULLQUATSZ], double p[FULLQUATSZ])
{
  p[0]=q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3];
  p[1]=q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2];
  p[2]=q1[0]*q2[2]+q2[0]*q1[2]+q2[1]*q1[3]-q1[1]*q2[3];
  p[3]=q1[0]*q2[3]+q2[0]*q1[3]+q1[1]*q2[2]-q2[1]*q1[2];
}

/*
 * fast multiplication of the two quaternions in q1 and q2 into p
 * this is the second of the two schemes derived in pg. 8 of
 * T. D. Howell, J.-C. Lafon, The complexity of the quaternion product, TR 75-245, Cornell Univ., June 1975.
 *
 * total additions increase from 12 to 27 (28), but total multiplications decrease from 16 to 9 (12)
 */
inline static void quatMultFast(double q1[FULLQUATSZ], double q2[FULLQUATSZ], double p[FULLQUATSZ])
{
double t1, t2, t3, t4, t5, t6, t7, t8, t9;
//double t10, t11, t12;

  t1=(q1[0]+q1[1])*(q2[0]+q2[1]);
  t2=(q1[3]-q1[2])*(q2[2]-q2[3]);
  t3=(q1[1]-q1[0])*(q2[2]+q2[3]);
  t4=(q1[2]+q1[3])*(q2[1]-q2[0]);
  t5=(q1[1]+q1[3])*(q2[1]+q2[2]);
  t6=(q1[1]-q1[3])*(q2[1]-q2[2]);
  t7=(q1[0]+q1[2])*(q2[0]-q2[3]);
  t8=(q1[0]-q1[2])*(q2[0]+q2[3]);

#if 0
  t9 =t5+t6;
  t10=t7+t8;
  t11=t5-t6;
  t12=t7-t8;

  p[0]= t2 + 0.5*(-t9+t10);
  p[1]= t1 - 0.5*(t9+t10);
  p[2]=-t3 + 0.5*(t11+t12);
  p[3]=-t4 + 0.5*(t11-t12);
#endif

  /* following fragment it equivalent to the one above */
  t9=0.5*(t5-t6+t7+t8);
  p[0]= t2 + t9-t5;
  p[1]= t1 - t9-t6;
  p[2]=-t3 + t9-t8;
  p[3]=-t4 + t9-t7;
}


/* Routines to estimate the estimated measurement vector (i.e. "func") and
 * its sparse jacobian (i.e. "fjac") needed in BA. Code below makes use of the
 * routines calcImgProj() and calcImgProjJacXXX() which
 * compute the predicted projection & jacobian of a SINGLE 3D point (see imgproj.c).
 * In the terminology of TR-340, these routines compute Q and its jacobians A=dQ/da, B=dQ/db.
 * Notice also that what follows is two pairs of "func" and corresponding "fjac" routines.
 * The first is to be used in full (i.e. motion + structure) BA, the second in 
 * motion only BA.
 */

static const double zerorotquat[FULLQUATSZ]={1.0, 0.0, 0.0, 0.0};

/****************************************************************************************/
/* MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR VARYING CAMERA POSE AND 3D STRUCTURE */
/****************************************************************************************/

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE SIMPLE DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given the parameter vectors aj and bi of camera j and point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projRTS(int j, int i, double *aj, double *bi, double *xij, void *adata)
{
  int KpMult;
  double *Kparms, *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  if(gl->calibcams)
  {
	  KpMult = gl->calibcams > 1 ? 5:0;
	  Kparms=gl->intrcalib + j*KpMult;

	  calcImgProj(Kparms, pr0, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
  }
  else
	  calcImgProjNoK(pr0, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
}

/* Given the parameter vectors aj and bi of camera j and point i, computes in Aij, Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projRTS_jac(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata)
{
  int KpMult;
  double *Kparms, *pr0;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  if(gl->calibcams)
  {
	  KpMult = gl->calibcams > 1 ? 5:0;
	  Kparms=gl->intrcalib + j*KpMult;

	  calcImgProjJacRTS(Kparms, pr0, aj, aj+3, bi, (double (*)[6])Aij, (double (*)[3])Bij); // 3 is the quaternion's vector part length
  }
  else
	  calcImgProjJacRTSnoK(pr0, aj, aj+3, bi, (double (*)[6])Aij, (double (*)[3])Bij);
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projRT(int j, int i, double *aj, double *xij, void *adata)
{
  int pnp, KpMult;

  double *Kparms, *pr0, *ptparams;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  ptparams=gl->ptparams;

  if(gl->calibcams)
  {
	  KpMult = gl->calibcams > 1 ? 5:0;
	  Kparms=gl->intrcalib + j*KpMult;

	  calcImgProj(Kparms, pr0, aj, aj+3, ptparams+i*pnp, xij); // 3 is the quaternion's vector part length
  }
  else
	  calcImgProjNoK(pr0, aj, aj+3, ptparams+i*pnp, xij);
}

/* Given the parameter vector aj of camera j, computes in Aij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
  int pnp, KpMult;

  double *Kparms, *ptparams, *pr0;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  ptparams=gl->ptparams;

  if(gl->calibcams)
  {
	  KpMult = gl->calibcams > 1 ? 5:0;
	  Kparms=gl->intrcalib + j*KpMult;

	  calcImgProjJacRT(Kparms, pr0, aj, aj+3, ptparams+i*pnp, (double (*)[6])Aij); // 3 is the quaternion's vector part length
  }
  else
	  calcImgProjJacRTnoK(pr0, aj, aj+3, ptparams+i*pnp, (double (*)[6])Aij);
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given the parameter vector bi of point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projS(int j, int i, double *bi, double *xij, void *adata)
{
  int cnp, KpMult;

  double *Kparms, *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  if(gl->calibcams)
  {
	  KpMult = gl->calibcams > 1 ? 5:0;
	  Kparms=gl->intrcalib + j*KpMult;

	  calcImgProjFullR(Kparms, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
	  //calcImgProj(Kparms, (double *)zerorotquat, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
  }
  else
  {
	  calcImgProjFullRnoK(aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
	  //calcImgProjNoK((double *)zerorotquat, aj, aj+3, bi, xij); // 3 is the quaternion's vector part length
  }
}

/* Given the parameter vector bi of point i, computes in Bij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projS_jac(int j, int i, double *bi, double *Bij, void *adata)
{
  int cnp, KpMult;

  double *Kparms, *camparams, *aj;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  if(gl->calibcams)
  {
	  KpMult = gl->calibcams > 1 ? 5:0;
	  Kparms=gl->intrcalib + j*KpMult;

	  calcImgProjJacS(Kparms, (double *)zerorotquat, aj, aj+3, bi, (double (*)[3])Bij); // 3 is the quaternion's vector part length
  }
  else
	  calcImgProjJacSnoK((double *)zerorotquat, aj, aj+3, bi, (double (*)[3])Bij);
}

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE EXPERT DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsRTS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *pqr, *pt, *ppt, *pmeas, *Kparms, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz, KpMult, imgidx;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  if(gl->calibcams)
  {
    Kparms=gl->intrcalib;
    KpMult = gl->calibcams > 1 ? 5:0;
  }

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=pa+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    if(gl->calibcams)
    {
      Kparms += j*KpMult; //intrinsic parameters of the m (different) cameras
      for(i=0; i<nnz; ++i){
        ppt=pb + rcsubs[i]*pnp;
        imgidx = idxij->val[rcidxs[i]]*mnp;
        pmeas=hx + imgidx; // set pmeas to point to hx_ij


        calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
        //calcImgProj(Kparms, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas

        if(gl->imgpts)
          convertCostToPseudoHuber(pmeas,gl->imgpts+imgidx,mnp,gl->thresh);
      }
    }
    else
    {
      for(i=0; i<nnz; ++i){
        ppt=pb + rcsubs[i]*pnp;
        imgidx = idxij->val[rcidxs[i]]*mnp;
        pmeas=hx + imgidx; // set pmeas to point to hx_ij

        calcImgProjFullRnoK(trot, pt, ppt, pmeas); // evaluate Q in pmeas
        //calcImgProjNoK(pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas

        if(gl->imgpts)
          convertCostToPseudoHuber(pmeas,gl->imgpts+imgidx,mnp,gl->thresh);
      }
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm, B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where A_ij=dx_ij/db_j and B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 */
static void img_projsRTS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j;
  int cnp, pnp, mnp, KpMult;
  double *pa, *pb, *pqr, *pt, *ppt, *pA, *pB, *Kparms, *pr0;
  //int n;
  int m, nnz, Asz, Bsz, ABsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;

  if(gl->calibcams)
  {
	  Kparms=gl->intrcalib;
	  KpMult = gl->calibcams > 1 ? 5:0;
  }

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=pa+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

	if(gl->calibcams)
	{
		Kparms += j*KpMult; //intrinsic parameters of the m (different) cameras
		for(i=0; i<nnz; ++i){
		  ppt=pb + rcsubs[i]*pnp;
		  pA=jac + idxij->val[rcidxs[i]]*ABsz; // set pA to point to A_ij
		  pB=pA  + Asz; // set pB to point to B_ij

		  calcImgProjJacRTS(Kparms, pr0, pqr, pt, ppt, (double (*)[6])pA, (double (*)[3])pB); // evaluate dQ/da, dQ/db in pA, pB
		}
	}
	else
	{
		for(i=0; i<nnz; ++i){
		  ppt=pb + rcsubs[i]*pnp;
		  pA=jac + idxij->val[rcidxs[i]]*ABsz; // set pA to point to A_ij
		  pB=pA  + Asz; // set pB to point to B_ij

		  calcImgProjJacRTSnoK(pr0, pqr, pt, ppt, (double (*)[6])pA, (double (*)[3])pB); // evaluate dQ/da, dQ/db in pA, pB
		}
	}
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
 * The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
 * where hx_ij is the predicted projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp, KpMult;
  double *pqr, *pt, *ppt, *pmeas, *Kparms, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  if(gl->calibcams)
  {
	  Kparms=gl->intrcalib;
	  KpMult = gl->calibcams > 1 ? 5:0;
  }
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=p+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

	if(gl->calibcams)
	{
		Kparms += j*KpMult; //intrinsic parameters of the m (different) cameras
		for(i=0; i<nnz; ++i){
			ppt=ptparams + rcsubs[i]*pnp;
		  pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

		  calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
		  //calcImgProj(Kparms, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
		}
	}
	else
	{
		for(i=0; i<nnz; ++i){
			ppt=ptparams + rcsubs[i]*pnp;
		  pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

		  calcImgProjFullRnoK(trot, pt, ppt, pmeas); // evaluate Q in pmeas
		  //calcImgProjNoK(pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
		}
	}
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
 * the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
 * where A_ij=dx_ij/db_j (see HZ).
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 */
static void img_projsRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j;
  int cnp, pnp, mnp, KpMult;
  double *pqr, *pt, *ppt, *pA, *Kparms, *ptparams, *pr0;
  //int n;
  int m, nnz, Asz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  if(gl->calibcams)
  {
	  Kparms=gl->intrcalib;
	  KpMult = gl->calibcams > 1 ? 5:0;
  }
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=p+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

	if(gl->calibcams)
	{
		Kparms += j*KpMult; //intrinsic parameters of the m (different) cameras
		for(i=0; i<nnz; ++i){
		  ppt=ptparams + rcsubs[i]*pnp;
		  pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij

		  calcImgProjJacRT(Kparms, pr0, pqr, pt, ppt, (double (*)[6])pA); // evaluate dQ/da in pA
		}
	}
	else
	{
		for(i=0; i<nnz; ++i){
		  ppt=ptparams + rcsubs[i]*pnp;
		  pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij

		  calcImgProjJacRTnoK(pr0, pqr, pt, ppt, (double (*)[6])pA); // evaluate dQ/da in pA
		}
	}
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp, KpMult;
  double *pqr, *pt, *ppt, *pmeas, *Kparms, *camparams, trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  if(gl->calibcams)
  {
	  Kparms=gl->intrcalib;
	  KpMult = gl->calibcams > 1 ? 5:0;
  }
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=camparams+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements
    _MK_QUAT_FRM_VEC(trot, pqr);

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

	if(gl->calibcams)
	{
		Kparms += j*KpMult; //intrinsic parameters of the m (different) cameras
		for(i=0; i<nnz; ++i){
		  ppt=p + rcsubs[i]*pnp;
		  pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

		  calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
		  //calcImgProj(Kparms, (double *)zerorotquat, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
		}
	}
	else
	{
		for(i=0; i<nnz; ++i){
		  ppt=p + rcsubs[i]*pnp;
		  pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

		  calcImgProjFullRnoK(trot, pt, ppt, pmeas); // evaluate Q in pmeas
		  //calcImgProjNoK((double *)zerorotquat, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
		}
	}
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 */
static void img_projsS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j;
  int cnp, pnp, mnp, KpMult;
  double *pqr, *pt, *ppt, *pB, *Kparms, *camparams;
  //int n;
  int m, nnz, Bsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  if(gl->calibcams)
  {
	  Kparms=gl->intrcalib;
	  KpMult = gl->calibcams > 1 ? 5:0;
  }
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;
  Bsz=mnp*pnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pqr=camparams+j*cnp;
    pt=pqr+3; // quaternion vector part has 3 elements

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

	if(gl->calibcams)
	{
		Kparms += j*KpMult; //intrinsic parameters of the m (different) cameras
		for(i=0; i<nnz; ++i){
		  ppt=p + rcsubs[i]*pnp;
		  pB=jac + idxij->val[rcidxs[i]]*Bsz; // set pB to point to B_ij

		  calcImgProjJacS(Kparms, (double *)zerorotquat, pqr, pt, ppt, (double (*)[3])pB); // evaluate dQ/da in pB
		}
	}
	else
	{
		for(i=0; i<nnz; ++i){
		  ppt=p + rcsubs[i]*pnp;
		  pB=jac + idxij->val[rcidxs[i]]*Bsz; // set pB to point to B_ij

		  calcImgProjJacSnoK((double *)zerorotquat, pqr, pt, ppt, (double (*)[3])pB); // evaluate dQ/da in pB
		}
	}
  }
}

/****************************************************************************************************/
/* MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR VARYING CAMERA INTRINSICS, POSE AND 3D STRUCTURE */
/****************************************************************************************************/

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE SIMPLE DRIVERS ***/

/* A note about the computation of Jacobians below:
 *
 * When performing BA that includes the camera intrinsics, it would be
 * very desirable to allow for certain parameters such as skew, aspect
 * ratio and principal point to be fixed. The straighforward way to
 * implement this would be to code a separate version of the Jacobian
 * computation routines for each subset of non-fixed parameters. Here,
 * this is bypassed by developing only one set of Jacobian computation
 * routines which estimate the former for all 5 intrinsics and then set
 * the columns corresponding to fixed parameters to zero.
 */

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given the parameter vectors aj and bi of camera j and point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKRTS(int j, int i, double *aj, double *bi, double *xij, void *adata)
{
  double *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProj(aj, pr0, aj+5, aj+5+3, bi, xij); // 5 for the calibration + 3 for the quaternion's vector part
}

/* Given the parameter vectors aj and bi of camera j and point i, computes in Aij, Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projKRTS_jac(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata)
{
struct globs_ *gl;
double *pr0;
int ncK;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  calcImgProjJacKRTS(aj, pr0, aj+5, aj+5+3, bi, (double (*)[5+6])Aij, (double (*)[3])Bij); // 5 for the calibration + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  gl=(struct globs_ *)adata;
  ncK=gl->nccalib;
  if(ncK){
    int cnp, mnp, j0;

    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-ncK;

    for(i=0; i<mnp; ++i, Aij+=cnp)
      for(j=j0; j<5; ++j)
        Aij[j]=0.0; // Aij[i*cnp+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKRT(int j, int i, double *aj, double *xij, void *adata)
{
  int pnp;

  double *ptparams, *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProj(aj, pr0, aj+5, aj+5+3, ptparams+i*pnp, xij); // 5 for the calibration + 3 for the quaternion's vector part
}

/* Given the parameter vector aj of camera j, computes in Aij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projKRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
struct globs_ *gl;
double *ptparams, *pr0;
int pnp, ncK;
  
  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcImgProjJacKRT(aj, pr0, aj+5, aj+5+3, ptparams+i*pnp, (double (*)[5+6])Aij); // 5 for the calibration + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  ncK=gl->nccalib;
  if(ncK){
    int cnp, mnp, j0;

    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-ncK;

    for(i=0; i<mnp; ++i, Aij+=cnp)
      for(j=j0; j<5; ++j)
        Aij[j]=0.0; // Aij[i*cnp+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given the parameter vector bi of point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKS(int j, int i, double *bi, double *xij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcImgProjFullR(aj, aj+5, aj+5+3, bi, xij); // 5 for the calibration + 3 for the quaternion's vector part
  //calcImgProj(aj, (double *)zerorotquat, aj+5, aj+5+3, bi, xij); // 5 for the calibration + 3 for the quaternion's vector part
}

/* Given the parameter vector bi of point i, computes in Bij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projKS_jac(int j, int i, double *bi, double *Bij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcImgProjJacS(aj, (double *)zerorotquat, aj+5, aj+5+3, bi, (double (*)[3])Bij); // 5 for the calibration + 3 for the quaternion's vector part
}

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE EXPERT DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKRTS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *pqr, *pt, *ppt, *pmeas, *pcalib, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(pcalib, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(pcalib, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm, B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where A_ij=dx_ij/db_j and B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 */
static void img_projsKRTS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j, ii, jj;
  int cnp, pnp, mnp, ncK;
  double *pa, *pb, *pqr, *pt, *ppt, *pA, *pB, *pcalib, *pr0;
  //int n;
  int m, nnz, Asz, Bsz, ABsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*ABsz; // set pA to point to A_ij
      pB=pA  + Asz; // set pB to point to B_ij

      calcImgProjJacKRTS(pcalib, pr0, pqr, pt, ppt, (double (*)[5+6])pA, (double (*)[3])pB); // evaluate dQ/da, dQ/db in pA, pB

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0=5-ncK;

        for(ii=0; ii<mnp; ++ii, pA+=cnp)
          for(jj=jj0; jj<5; ++jj)
            pA[jj]=0.0; // pA[ii*cnp+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
 * The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
 * where hx_ij is the predicted projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
	    ppt=ptparams + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(pcalib, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(pcalib, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
 * the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
 * where A_ij=dx_ij/db_j (see HZ).
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 */
static void img_projsKRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j, ii, jj;
  int cnp, pnp, mnp, ncK;
  double *pqr, *pt, *ppt, *pA, *pcalib, *ptparams, *pr0;
  //int n;
  int m, nnz, Asz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=ptparams + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij

      calcImgProjJacKRT(pcalib, pr0, pqr, pt, ppt, (double (*)[5+6])pA); // evaluate dQ/da in pA

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0;

        jj0=5-ncK;
        for(ii=0; ii<mnp; ++ii, pA+=cnp)
          for(jj=jj0; jj<5; ++jj)
            pA[jj]=0.0; // pA[ii*cnp+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *camparams, trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    _MK_QUAT_FRM_VEC(trot, pqr);

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcImgProjFullR(pcalib, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcImgProj(pcalib, (double *)zerorotquat, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 */
static void img_projsKS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pB, *pcalib, *camparams;
  //int n;
  int m, nnz, Bsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;
  Bsz=mnp*pnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pqr=pcalib+5;
    pt=pqr+3; // quaternion vector part has 3 elements

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pB=jac + idxij->val[rcidxs[i]]*Bsz; // set pB to point to B_ij

      calcImgProjJacS(pcalib, (double *)zerorotquat, pqr, pt, ppt, (double (*)[3])pB); // evaluate dQ/da in pB
    }
  }
}

/****************************************************************************************************************/
/* MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR VARYING CAMERA INTRINSICS, DISTORTION, POSE AND 3D STRUCTURE */
/****************************************************************************************************************/

/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE SIMPLE DRIVERS ***/

/* A note about the computation of Jacobians below:
 *
 * When performing BA that includes the camera intrinsics & distortion, it would be
 * very desirable to allow for certain parameters such as skew, aspect ratio and principal
 * point (also high order radial distortion, tangential distortion), to be fixed. The
 * straighforward way to implement this would be to code a separate version of the
 * Jacobian computation routines for each subset of non-fixed parameters. Here,
 * this is bypassed by developing only one set of Jacobian computation
 * routines which estimate the former for all 5 intrinsics and all 5 distortion
 * coefficients and then set the columns corresponding to fixed parameters to zero.
 */

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given the parameter vectors aj and bi of camera j and point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKDRTS(int j, int i, double *aj, double *bi, double *xij, void *adata)
{
  double *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcDistImgProj(aj, aj+5, pr0, aj+5+5, aj+5+5+3, bi, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}

/* Given the parameter vectors aj and bi of camera j and point i, computes in Aij, Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projKDRTS_jac(int j, int i, double *aj, double *bi, double *Aij, double *Bij, void *adata)
{
struct globs_ *gl;
double *pA, *pr0;
int nc;

  gl=(struct globs_ *)adata;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
  calcDistImgProjJacKDRTS(aj, aj+5, pr0, aj+5+5, aj+5+5+3, bi, (double (*)[5+5+6])Aij, (double (*)[3])Bij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  gl=(struct globs_ *)adata;
  nc=gl->nccalib;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[j]=0.0; // pA[i*cnp+j]=0.0;
  }

  /* clear the columns of the Jacobian corresponding to fixed distortion parameters */
  nc=gl->ncdist;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[5+j]=0.0; // pA[i*cnp+5+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given the parameter vector aj of camera j, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKDRT(int j, int i, double *aj, double *xij, void *adata)
{
  int pnp;

  double *ptparams, *pr0;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcDistImgProj(aj, aj+5, pr0, aj+5+5, aj+5+5+3, ptparams+i*pnp, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}

/* Given the parameter vector aj of camera j, computes in Aij
 * the jacobian of the predicted projection of point i on image j
 */
static void img_projKDRT_jac(int j, int i, double *aj, double *Aij, void *adata)
{
struct globs_ *gl;
double *pA, *ptparams, *pr0;
int pnp, nc;
  
  gl=(struct globs_ *)adata;
  pnp=gl->pnp;
  ptparams=gl->ptparams;
  pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

  calcDistImgProjJacKDRT(aj, aj+5, pr0, aj+5+5, aj+5+5+3, ptparams+i*pnp, (double (*)[5+5+6])Aij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part

  /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
  nc=gl->nccalib;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[j]=0.0; // pA[i*cnp+j]=0.0;
  }
  nc=gl->ncdist;
  if(nc){
    int cnp, mnp, j0;

    pA=Aij;
    cnp=gl->cnp;
    mnp=gl->mnp;
    j0=5-nc;

    for(i=0; i<mnp; ++i, pA+=cnp)
      for(j=j0; j<5; ++j)
        pA[5+j]=0.0; // pA[i*cnp+5+j]=0.0;
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given the parameter vector bi of point i, computes in xij the
 * predicted projection of point i on image j
 */
static void img_projKDS(int j, int i, double *bi, double *xij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcDistImgProjFullR(aj, aj+5, aj+5+5, aj+5+5+3, bi, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
  //calcDistImgProj(aj, aj+5, (double *)zerorotquat, aj+5+5, aj+5+5+3, bi, xij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}

/* Given the parameter vector bi of point i, computes in Bij the
 * jacobian of the predicted projection of point i on image j
 */
static void img_projKDS_jac(int j, int i, double *bi, double *Bij, void *adata)
{
  int cnp;

  double *camparams, *aj;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp;
  camparams=gl->camparams;
  aj=camparams+j*cnp;

  calcDistImgProjJacS(aj, aj+5, (double *)zerorotquat, aj+5+5, aj+5+5+3, bi, (double (*)[3])Bij); // 5 for the calibration + 5 for the distortion + 3 for the quaternion's vector part
}


/*** MEASUREMENT VECTOR AND JACOBIAN COMPUTATION FOR THE EXPERT DRIVERS ***/

/* FULL BUNDLE ADJUSTMENT, I.E. SIMULTANEOUS ESTIMATION OF CAMERA AND STRUCTURE PARAMETERS */

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKDRTS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pa, *pb, *pqr, *pt, *ppt, *pmeas, *pcalib, *pdist, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcDistImgProjFullR(pcalib, pdist, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcDistImgProj(pcalib, pdist, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points and the parameters of m cameras, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm, B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where A_ij=dx_ij/db_j and B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the A_ij, B_ij might be missing
 *
 */
static void img_projsKDRTS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j, ii, jj;
  int cnp, pnp, mnp, ncK, ncD;
  double *pa, *pb, *pqr, *pt, *ppt, *pA, *pB, *ptr, *pcalib, *pdist, *pr0;
  //int n;
  int m, nnz, Asz, Bsz, ABsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;
  ncD=gl->ncdist;

  //n=idxij->nr;
  m=idxij->nc;
  pa=p; pb=p+m*cnp;
  Asz=mnp*cnp; Bsz=mnp*pnp; ABsz=Asz+Bsz;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=pa+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=pb + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*ABsz; // set pA to point to A_ij
      pB=pA  + Asz; // set pB to point to B_ij

      calcDistImgProjJacKDRTS(pcalib, pdist, pr0, pqr, pt, ppt, (double (*)[5+5+6])pA, (double (*)[3])pB); // evaluate dQ/da, dQ/db in pA, pB

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0=5-ncK;

        ptr=pA;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[jj]=0.0; // ptr[ii*cnp+jj]=0.0;
      }

      /* clear the columns of the Jacobian corresponding to fixed distortion parameters */
      if(ncD){
        int jj0=5-ncD;

        ptr=pA;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[5+jj]=0.0; // ptr[ii*cnp+5+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR CAMERA PARAMETERS ONLY */

/* Given a parameter vector p made up of the parameters of m cameras, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images.
 * The measurements are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T,
 * where hx_ij is the predicted projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKDRT_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *pdist, *ptparams, *pr0, lrot[FULLQUATSZ], trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate
    _MK_QUAT_FRM_VEC(lrot, pqr);
    quatMultFast(lrot, pr0, trot); // trot=lrot*pr0

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
	    ppt=ptparams + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcDistImgProjFullR(pcalib, pdist, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcDistImgProj(pcalib, pdist, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the parameters of m cameras, compute in jac
 * the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (A_11, ..., A_1m, ..., A_n1, ..., A_nm),
 * where A_ij=dx_ij/db_j (see HZ).
 * Notice that depending on idxij, some of the A_ij might be missing
 *
 */
static void img_projsKDRT_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j, ii, jj;
  int cnp, pnp, mnp, ncK, ncD;
  double *pqr, *pt, *ppt, *pA, *ptr, *pcalib, *pdist, *ptparams, *pr0;
  //int n;
  int m, nnz, Asz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  ncK=gl->nccalib;
  ncD=gl->ncdist;
  ptparams=gl->ptparams;

  //n=idxij->nr;
  m=idxij->nc;
  Asz=mnp*cnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=p+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    pr0=gl->rot0params+j*FULLQUATSZ; // full quat for initial rotation estimate

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=ptparams + rcsubs[i]*pnp;
      pA=jac + idxij->val[rcidxs[i]]*Asz; // set pA to point to A_ij

      calcDistImgProjJacKDRT(pcalib, pdist, pr0, pqr, pt, ppt, (double (*)[5+5+6])pA); // evaluate dQ/da in pA

      /* clear the columns of the Jacobian corresponding to fixed calibration parameters */
      if(ncK){
        int jj0;

        ptr=pA;
        jj0=5-ncK;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[jj]=0.0; // ptr[ii*cnp+jj]=0.0;
      }

      /* clear the columns of the Jacobian corresponding to fixed distortion parameters */
      if(ncD){
        int jj0;

        ptr=pA;
        jj0=5-ncD;
        for(ii=0; ii<mnp; ++ii, ptr+=cnp)
          for(jj=jj0; jj<5; ++jj)
            ptr[5+jj]=0.0; // ptr[ii*cnp+5+jj]=0.0;
      }
    }
  }
}

/* BUNDLE ADJUSTMENT FOR STRUCTURE PARAMETERS ONLY */

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * hx the prediction of the measurements, i.e. the projections of 3D points in the m images. The measurements
 * are returned in the order (hx_11^T, .. hx_1m^T, ..., hx_n1^T, .. hx_nm^T)^T, where hx_ij is the predicted
 * projection of the i-th point on the j-th camera.
 * Notice that depending on idxij, some of the hx_ij might be missing
 *
 */
static void img_projsKDS_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pmeas, *pcalib, *pdist, *camparams, trot[FULLQUATSZ];
  //int n;
  int m, nnz;
  struct globs_ *gl;

  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements
    _MK_QUAT_FRM_VEC(trot, pqr);

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pmeas=hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

      calcDistImgProjFullR(pcalib, pdist, trot, pt, ppt, pmeas); // evaluate Q in pmeas
      //calcDistImgProj(pcalib, pdist, (double *)zerorotquat, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
    }
  }
}

/* Given a parameter vector p made up of the 3D coordinates of n points, compute in
 * jac the jacobian of the predicted measurements, i.e. the jacobian of the projections of 3D points in the m images.
 * The jacobian is returned in the order (B_11, ..., B_1m, ..., B_n1, ..., B_nm),
 * where B_ij=dx_ij/db_i (see HZ).
 * Notice that depending on idxij, some of the B_ij might be missing
 *
 */
static void img_projsKDS_jac_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *jac, void *adata)
{
  int i, j;
  int cnp, pnp, mnp;
  double *pqr, *pt, *ppt, *pB, *pcalib, *pdist, *camparams;
  //int n;
  int m, nnz, Bsz;
  struct globs_ *gl;
  
  gl=(struct globs_ *)adata;
  cnp=gl->cnp; pnp=gl->pnp; mnp=gl->mnp;
  camparams=gl->camparams;

  //n=idxij->nr;
  m=idxij->nc;
  Bsz=mnp*pnp;

  for(j=0; j<m; ++j){
    /* j-th camera parameters */
    pcalib=camparams+j*cnp;
    pdist=pcalib+5;
    pqr=pdist+5;
    pt=pqr+3; // quaternion vector part has 3 elements

    nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

    for(i=0; i<nnz; ++i){
      ppt=p + rcsubs[i]*pnp;
      pB=jac + idxij->val[rcidxs[i]]*Bsz; // set pB to point to B_ij

      calcDistImgProjJacS(pcalib, pdist, (double *)zerorotquat, pqr, pt, ppt, (double (*)[3])pB); // dQ/db in pB
    }
  }
}


/* Performs Bundle Adjustment (BA) by a Sparse Bundleadjustment Algorithm (SBA) for 2 or more cameras and 
 * corresponding structures (euclidean 3D points). The methode of operation can be adjusted beforehand by
 * the class constructor or the different set-methods. If not every structure element is visable in all
 * views (cameras), a mask for each view (vector of maskes) has to be provided for the algorithm which
 * specifies, which structure element is visible (true) and which not (false). If the intrinsic parameters
 * are provided to this function, it is assumed that the provided image projections are in image coordinates.
 * Otherwise, the 2D image projections must be in camera coordinates (K^(-1)*x). Moreover, if only one
 * set of camera intrinsics (vector size = 1) is provided, it is assumed that all frames are from the same
 * camera. In this case, the intrinsics must be fixed (fixedcal = true) and are not optimized. Moreover,
 * all image projections provided to this function must be undistorted except the number of provided 
 * not fixed intrinsics equals the number of cams. The following table gives an overview of the possible 
 * number of different camera intriniscs (different cameras) in relation to the fixation (during BA) of 
 * intriniscs, the possible use of distortion parameters (for optimization) and the input format of the 2D 
 * projections (image or camera coordinate system):
 *	| num. of diff. intr	|	fix of intrinsics	|	distortion BA possible	|	Coord. system	|	Possible
 *	|			0			|			0			|				0			|		camera		|		1
 *	|			0			|			1			|				0			|		camera		|		1
 *	|			1			|			0			|				0			|		image		|		0
 *	|			1			|			1			|				0			|		image		|		1
 *	|			n			|			0			|				1			|		image		|		1
 *	|			n			|			1			|				0			|		image		|		1
 * For mor information please see the comments to the class constructor and the sba website.
 *
 * vector<double *> & Rquats		Input & Output  -> Vector of pointers to quaternions of the form
 *											  [w,x,y,z]. Each vector element represents data for one frame
 *											  (or camera) beginning with the first one (typically [1,0,0,0].
 *											  Rotation quaternions have the scalar part as their first 
 *											  element, i.e. a rotation by angle TH around the unit vector 
 *											  U=(Ux, Uy, Uz) should be specified as
 *											  cos(TH/2) Ux*sin(TH/2) Uy*sin(TH/2) Uz*sin(TH/2). If the variable 
 *											  useInputVarsAsOutput = 1, the result from BA is written back to this
 *											  memory adresses
 * vector<double *> & trans			Input & Output  -> Vector of pointers to translations of the form [x,y,z].
 *											  Each vector element represents data for one frame (or camera) 
 *											  beginning with the first one (typically [0,0,0]).
 *											  If the variable useInputVarsAsOutput = 1, the result from BA is
 *											  written back to this memory adresses
 * vector<double *> pts2D			Input  -> Vector of pointers to the image projections per frame of the
 *											  form [x1,y1,x2,y2,...xn,yn]. 
 *											  Each vector element represents data for one frame (or camera) 
 *											  beginning with the first one. The number of image projections 
 *											  per frame is provided by num2Dpts. If number of image projections
 *											  differs from frame to frame, a mask has to be provided for each
 *											  frame with the vector mask2Dpts. If no camera intrinsics are 
 *											  provided to this function, the coordinates must be in the
 *											  camera coordinate system (K^(-1)*x, x ... coordinate in image 
 *											  coordinate system). Moreover, the image projections must be 
 *											  undistorted except the number of provided not fixed intrinsics 
 *											  equals the number of cams. The image projections must be in
 *											  the same order as the structure elements (pts3D).
 * vector<int> num2Dpts				Input  -> number of image projections per frame (camera)
 * double *pts3D					Input & Output  -> Pointer to the structure elements (euclidean 3D points)
 *											  of the form [X1,Y1,Z1,X2,Y2,Z2,...Xn,Yn,Zn]. If the variable
 *											  useInputVarsAsOutput = 1, the result from BA is written back
 *											  to this memory adress
 * int numpts3D						Input  -> number of structure elements
 * vector<char *> *mask2Dpts		Input  -> If not NULL, this variable points to a vector of pointers
 *											  whereas each of these pointers holds a mask for a specific
 *											  frame which provides information if a given structure element
 *											  (pts3D) is visible in the corresponding frame. Thus, the size
 *											  of each mask must equal the size of structure elements.
 * vector<double *> *intrParms		Input & Output  -> Pointer to the vector of camera intriniscs (stored
 *											  in arrays with size 5 -> [fu,u0,v0,ar,s] with ar=fv/fu). 
 *											  Each vector element represents data for one frame (or camera) 
 *											  beginning with the first one. If this pointer is NULL, the 
 *											  image projections must be in the camera coordinate system 
 *											  (K^(-1)*x). Otherwise, the must be in the image coordinate
 *											  system. If the variable useInputVarsAsOutput = 1, the result 
 *											  from BA is written back to this memory adresses
 * vector<double *> *dist			Input & Output  -> Pointer to the vector of distortion coeffitients for
 *											  each camera (stored in arrays with size 5 -> [k1,k2,k3,k4,k5] 
 *											  with k1 (2nd order), k2 (4th order) and k5 (6th order) beeing 
 *											  the radial and k3 and k4 beeing the tangential distortion 
 *											  coeffitients. If this pointer is NULL, the image projections 
 *											  must be undistorted. The employed distortion model is the one 
 *											  used by Bouguet, see
 *											  http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html.
 *											  More specifically, assume x_n=(x, y)=(X/Z, Y/Z) is the pinhole 
 *											  image projection and let r^2=x^2+y^2. The distorted image 
 *											  projection is defined as 
 *											  x_d=(1 + kc[0]*r^2 + kc[1]*r^4 + kc[4]*r^6)*x_n + dx, with
 *											  dx=(2*kc[2]*x*y + kc[3]*(r^2+2*x^2), kc[2]*(r^2+2*y^2) + 2*kc[3]*x*y).
 *											  The distorted point in pixel coordinates is given by K*x_d, 
 *											  K being the intrinsics. Note that zero-based indexing is used 
 *											  above, Bouguet's page conforms to Matlab's convention and uses 
 *											  one-based indexing! If the variable useInputVarsAsOutput = 1, 
 *											  the result from BA is written back to this memory adresses
 * vector<double *> *cov2Dpts		Input  -> Pointer to a vector of pointers containing arrays of covariances
 *											  for each provided image projection of the form
 *											  [covx1^2,covx1y1,covx1y1,covy1^2,covx2^2,covx2y2,covx2y2,covy2^2,...
 *											  covxn^2,covxnyn,covxnyn,covyn^2]. Each vector element represents 
 *											  data for one frame (or camera) beginning with the first one. If
 *											  this pointer is NULL, no covariances are used.
 *
 * Return value:					 0:		  Everything ok
 *									-1:		  Structure of input variables or options is wrong
 *									-2:		  Memory allocation failed
 *									-3:		  Error in SBA
 */
int SBAdriver::perform_sba(std::vector<double *> & Rquats, 
			   std::vector<double *> & trans, 
			   std::vector<double *> pts2D,
			   std::vector<int> num2Dpts, 
			   double *pts3D, 
			   int numpts3D,
			   std::vector<char *> *mask2Dpts,
			   std::vector<double *> *intrParms, 
			   std::vector<double *> *dist,
			   std::vector<double *> *cov2Dpts)
{
  int cnp=6, /* 3 rot params + 3 trans params */
	  pnp=3, /* euclidean 3D points */
      mnp=2; /* image points are 2D */
  double *motstruct, *motstruct_copy, *imgpts, *covimgpts, *initrot;
  double *imgpts_tmp = 0;
  double *covimgpts_tmp = 0;

  char *vmask;
  //double opts[SBA_OPTSSZ], info[SBA_INFOSZ];
  double phi;
  //int expert, analyticjac, verbose=0;
  int n;
  bool havedist = false;
  bool motstruct_after_mot = false;
  int nframes, numprojs, nvars;
  int ret = 0;
  int i;
  
  #if BA_DEBUG 
  char tbuf[32];
  static char *howtoname[]={"BA_MOTSTRUCT", "BA_MOT", "BA_STRUCT", "BA_MOT_MOTSTRUCT"};
  clock_t start_time, end_time;
  #endif

  if((pts3D == nullptr) || (numpts3D <= 0))
	  return -1;
  if((Rquats.size() != trans.size()) || (Rquats.size() != pts2D.size()) || (Rquats.size() != num2Dpts.size()))
	  return -1;
  if(intrParms && !fixedcal)
  {
	  if(Rquats.size() != (*intrParms).size())
		  return -1;
	  if(dist && (Rquats.size() != (*dist).size()))
		  return -1;
  }
  if(intrParms && fixedcal)
  {
	  if(((*intrParms).size() > 1) && ((*intrParms).size() != Rquats.size()))
		  return -1;
	  if(dist)
		  return -1;
  }
  if(dist && !intrParms)
	  return -1;
  if(mask2Dpts && (mask2Dpts->size() != Rquats.size()))
	  return -1;
  if(cov2Dpts && (cov2Dpts->size() != Rquats.size()))
	  return -1;
  /*if(((prnt == BA_MOTSTRUCT) || (prnt == BA_MOT_MOTSTRUCT)) && 
	  ((Rquats_out == NULL) || (trans_out == NULL) || (pts3D_out == NULL) ||
	  ((intrParms != NULL) && (intrParms_out == NULL)) ||
	  ((dist != NULL) && (dist_out == NULL))))
	  return -1;
  if((prnt == BA_MOT) && ((Rquats_out == NULL) || (trans_out == NULL) ||
	  ((intrParms != NULL) && (intrParms_out == NULL)) ||
	  ((dist != NULL) && (dist_out == NULL))))
	  return -1;
  if((prnt == BA_STRUCT) && (pts3D_out == NULL))
	  return -1;*/


  /* Notice the various BA options demonstrated below */

  /* minimize motion & structure, motion only, or
   * motion and possibly motion & structure in a 2nd pass?
   */
  //howto=BA_MOTSTRUCT;
  //howto=BA_MOT;
  //howto=BA_STRUCT;
  //howto=BA_MOT_MOTSTRUCT;

  /* simple or expert drivers? */
  //expert=0;
  //expert=1;

  /* analytic or approximate jacobian? */
  //analyticjac=0;
  //analyticjac=1;

  /* print motion & structure estimates,
   * motion only or structure only upon completion?
   */
  //prnt=BA_NONE;
  //prnt=BA_MOTSTRUCT;
  //prnt=BA_MOT;
  //prnt=BA_STRUCT;

  nframes = (int)Rquats.size();
  numprojs = 0;
  for(i=0; i<(int)num2Dpts.size();++i)
	  numprojs += num2Dpts[i];
  if(intrParms && !fixedcal)
	  cnp+=5;
  if(dist)
  {
	  cnp+=5;
	  havedist = true;
  }

  //Allocate memory for the BA parameter vector
  motstruct=(double *)malloc((nframes*cnp + numpts3D*pnp)*sizeof(double));
  if(motstruct== nullptr){
    /*fprintf(stderr, "memory allocation for 'motstruct' failed\n");
    exit(1);*/
	  return -2; //Memory allocation failed
  }

  //Allocate memory for the initial rotations (BA calculates the best fitting difference to these rotations)
  initrot=(double *)malloc((nframes*FULLQUATSZ)*sizeof(double)); // Note: this assumes quaternions for rotations!
  if(initrot== nullptr){
    /*fprintf(stderr, "memory allocation for 'initrot' failed\n");
    exit(1);*/
	  free(motstruct);
	  return -2; //Memory allocation failed
  }

  //Store the camera parameters to the BA parameter vector (initial BA start values)
  double *motstruct_tmp = motstruct;
  double *initrot_tmp = initrot;
  for(i = 0; i < nframes; ++i)
  {
	  if(intrParms && !fixedcal)
	  {
		  memcpy(motstruct_tmp,(*intrParms)[i],5*sizeof(double));
		  motstruct_tmp += 5;
		  if(dist)
		  {
			  memcpy(motstruct_tmp,(*dist)[i],5*sizeof(double));
			  motstruct_tmp += 5;
		  }
	  }
	  
	  /* rotation */
	  /* normalize and ensure that the quaternion's scalar component is non-negative;
	   * if not, negate the quaternion since two quaternions q and -q represent the
	   * same rotation
	   */
	  double mag, sg;
	  mag = sqrt(Rquats[i][0] * Rquats[i][0] + Rquats[i][1] * Rquats[i][1]
			   + Rquats[i][2] * Rquats[i][2] + Rquats[i][3] * Rquats[i][3]);
	  sg = (*Rquats[i] >= 0.0) ? 1.0 : -1.0;
	  mag = sg/mag;
	  initrot_tmp[0] = Rquats[i][0]*mag;
	  initrot_tmp[1] = Rquats[i][1]*mag;
	  initrot_tmp[2] = Rquats[i][2]*mag;
	  initrot_tmp[3] = Rquats[i][3]*mag;

	  if(howto!=BA_STRUCT)
	  {
		  //the initial difference to the approximated (initial) rotation is zero
		  /* initialize the local rotation estimates to 0, corresponding to local quats (1, 0, 0, 0) 
		   * A local rotation estimate corresponds to the difference to initrot.
		   * For BA quaternion vectors are used (normalized quaternion without scalar component) */
		  motstruct_tmp[0]=motstruct_tmp[1]=motstruct_tmp[2]=0.0; // clear rotation
	  }
	  else
	  {
		  //If only the structure is refined, this is the final rotation
		  motstruct_tmp[0] = initrot_tmp[1];
		  motstruct_tmp[1] = initrot_tmp[2];
		  motstruct_tmp[2] = initrot_tmp[3];
	  }
	  motstruct_tmp += 3;
	  initrot_tmp += 4;

	  memcpy(motstruct_tmp,trans[i],3*sizeof(double));
	  motstruct_tmp += 3;
  }
  memcpy(motstruct_tmp, pts3D, numpts3D*pnp*sizeof(double));
  /* note that if howto==BA_STRUCT the rotation parts of motstruct actually equal the initial rotations! */


  imgpts=(double *)malloc(numprojs*mnp*sizeof(double));
  if(imgpts== nullptr){
    /*fprintf(stderr, "memory allocation for 'imgpts' failed\n");
    exit(1);*/
	  free(motstruct);
	  free(initrot);
	  return -2; //Memory allocation failed
  }

  vmask = (char *)malloc(numpts3D * nframes * sizeof(char));
  if(vmask==nullptr){
    /*fprintf(stderr, "memory allocation for 'vmask' failed\n");
    exit(1);*/
	ret = -2; //Memory allocation failed
	goto cleanup;
  }

  if(cov2Dpts != nullptr)
  {
	    covimgpts=(double *)malloc(numprojs*mnp*mnp*sizeof(double));
		if(covimgpts==nullptr){
		  /*fprintf(stderr, "memory allocation for 'covimgpts' failed\n");
		  exit(1);*/
		  ret = -2; //Memory allocation failed
		  goto cleanup;
		}
  }
  else
	  covimgpts = nullptr;

  imgpts_tmp = imgpts;
  covimgpts_tmp = covimgpts;
  
  if(mask2Dpts == nullptr)
  {
	  memset(vmask, 1, numpts3D * nframes * sizeof(char)); /* set whole vmask to 1 */
	  for(i = 0; i < numpts3D; ++i)
	  {
		  for(int j = 0; j < nframes; ++j)
		  {
			  memcpy(imgpts_tmp,pts2D[j]+i*mnp,mnp*sizeof(double));
			  imgpts_tmp += mnp;
		  }
	  }
	  if(cov2Dpts != nullptr)
	  {
		  int mnp2 = mnp*mnp;
		  for(i = 0; i < numpts3D; ++i)
		  {
			  for(int j = 0; j < nframes; ++j)
			  {
				  memcpy(covimgpts_tmp,(*cov2Dpts)[j]+i*mnp2,mnp2*sizeof(double));
				  covimgpts_tmp += mnp2;
			  }
		  }
	  }
  }
  else
  {
	  memset(vmask, 0, numpts3D * nframes * sizeof(char)); /* clear vmask */
  
	  int *pts2D_pt_cnt = (int *)malloc(nframes*sizeof(int));
	  if(pts2D_pt_cnt == nullptr)
	  {
		  ret = -2; //Memory allocation failed
		  goto cleanup;
	  }
	  memset(pts2D_pt_cnt,0,nframes*sizeof(int));
	  for(i = 0; i < numpts3D; ++i)
	  {
		  for(int j = 0; j < nframes; ++j)
		  {
			  if(num2Dpts[j] == 0) continue;
			  
			  if((*mask2Dpts)[j][i])
			  {
				  memcpy(imgpts_tmp,pts2D[j]+pts2D_pt_cnt[j],mnp*sizeof(double));
				  vmask[i*nframes+j] = 1;

				  if(cov2Dpts != nullptr)
				  {
					  memcpy(covimgpts_tmp,(*cov2Dpts)[j]+pts2D_pt_cnt[j]*mnp,mnp*mnp*sizeof(double));
					  covimgpts_tmp += mnp*mnp;
				  }

				  imgpts_tmp += mnp;
				  pts2D_pt_cnt[j] += mnp;
			  }
		  }
	  }
	  free(pts2D_pt_cnt);
  }
  
  if(costfunc)
  {
	  globs.imgpts = imgpts;
	  globs.thresh = costThresh;
  }
  else
	  globs.imgpts = nullptr;

  /* set up globs structure */
  globs.cnp=cnp; globs.pnp=pnp; globs.mnp=mnp;
  globs.rot0params=initrot;

  if(intrParms && fixedcal)
  {
	  if((*intrParms).size() == 1)
	  {
		  globs.calibcams = 1;

		  globs.intrcalib=(double *)malloc(5*sizeof(double));
		  if(globs.intrcalib==nullptr){
			  /*fprintf(stderr, "memory allocation for 'globs.intrcalib' failed\n");
			  exit(1);*/
			  ret = -2; //Memory allocation failed
			  goto cleanup;
		  }
		  memcpy(globs.intrcalib,(*intrParms)[0],5*sizeof(double));
	  }
	  else
	  {
		  globs.calibcams = nframes;

		  globs.intrcalib=(double *)malloc(5*nframes*sizeof(double));
		  if(globs.intrcalib==nullptr){
			  /*fprintf(stderr, "memory allocation for 'globs.intrcalib' failed\n");
			  exit(1);*/
			  ret = -2; //Memory allocation failed
			  goto cleanup;
		  }
		  for(i = 0; i < nframes; ++i)
			  memcpy(globs.intrcalib + i*5,(*intrParms)[i],5*sizeof(double));
	  }
	  havedist = false;
  }
  else //if(!intrParms && fixedcal)
  {
	  globs.calibcams = 0;
	  globs.intrcalib=nullptr;
	  globs.nccalib = nccalib;
	  if(cnp==16){ // 16 = 5+5+6
		  havedist = true; /* with distortion */
		  globs.ncdist = ncdist; /* number of distortion params to keep fixed, must be between 0 and 5 */
	  }
	  else{
		  havedist = false;
		  globs.ncdist=-9999;
	  }
  }
  
  globs.ptparams=nullptr;
  globs.camparams=nullptr;

  /* call sparse LM routine */
  /*opts[0]=SBA_INIT_MU; opts[1]=SBA_STOP_THRESH; opts[2]=SBA_STOP_THRESH;
  opts[3]=SBA_STOP_THRESH;
  //opts[3]=0.05*numprojs; // uncomment to force termination if the average reprojection error drops below 0.05
  opts[4]=0.0;*/
  //opts[4]=1E-05; // uncomment to force termination if the relative reduction in the RMS reprojection error drops below 1E-05

  #if BA_DEBUG
  start_time=clock();
  #endif
  switch(howto){
    case BA_MOTSTRUCT: /* BA for motion & structure */
      nvars=nframes*cnp+numpts3D*pnp;
      if(expert)
        n=sba_motstr_levmar_x(numpts3D, nconst3Dpts, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                            fixedcal? img_projsRTS_x : (havedist? img_projsKDRTS_x : img_projsKRTS_x),
                            analyticjac? (fixedcal? img_projsRTS_jac_x : (havedist? img_projsKDRTS_jac_x : img_projsKRTS_jac_x)) : nullptr,
                            (void *)(&globs), MAXITER2, verbose, opts, info);
      else
        n=sba_motstr_levmar(numpts3D, nconst3Dpts, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                            fixedcal? img_projRTS : (havedist? img_projKDRTS : img_projKRTS),
                            analyticjac? (fixedcal? img_projRTS_jac : (havedist? img_projKDRTS_jac : img_projKRTS_jac)) : nullptr,
                            (void *)(&globs), MAXITER2, verbose, opts, info);
    break;

    case BA_MOT: /* BA for motion only */
      globs.ptparams=motstruct+nframes*cnp;
      nvars=nframes*cnp;
      if(expert)
        n=sba_mot_levmar_x(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projsRT_x : (havedist? img_projsKDRT_x : img_projsKRT_x),
                          analyticjac? (fixedcal? img_projsRT_jac_x : (havedist? img_projsKDRT_jac_x : img_projsKRT_jac_x)) : nullptr,
                          (void *)(&globs), MAXITER, verbose, opts, info);
      else
        n=sba_mot_levmar(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projRT : (havedist? img_projKDRT : img_projKRT),
                          analyticjac? (fixedcal? img_projRT_jac : (havedist? img_projKDRT_jac : img_projKRT_jac)) : nullptr,
                          (void *)(&globs), MAXITER, verbose, opts, info);
    break;

    case BA_STRUCT: /* BA for structure only */
      globs.camparams=motstruct;
      nvars=numpts3D*pnp;
      if(expert)
        n=sba_str_levmar_x(numpts3D, nconst3Dpts, nframes, vmask, motstruct+nframes*cnp, pnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projsS_x : (havedist? img_projsKDS_x : img_projsKS_x), 
                          analyticjac? (fixedcal? img_projsS_jac_x : (havedist? img_projsKDS_jac_x : img_projsKS_jac_x)) : nullptr,
                          (void *)(&globs), MAXITER, verbose, opts, info);
      else
        n=sba_str_levmar(numpts3D, nconst3Dpts, nframes, vmask, motstruct+nframes*cnp, pnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projS : (havedist? img_projKDS : img_projKS),
                          analyticjac? (fixedcal? img_projS_jac : (havedist? img_projKDS_jac : img_projKS_jac)) : nullptr,
                          (void *)(&globs), MAXITER, verbose, opts, info);
    break;

    case BA_MOT_MOTSTRUCT: /* BA for motion only; if error too large, then BA for motion & structure */
      if((motstruct_copy=(double *)malloc((nframes*cnp + numpts3D*pnp)*sizeof(double)))==nullptr){
        /*fprintf(stderr, "memory allocation failed in sba_driver()!\n");
        exit(1);*/
		ret = -2; //Memory allocation failed
		goto cleanup;
      }

      memcpy(motstruct_copy, motstruct, (nframes*cnp + numpts3D*pnp)*sizeof(double)); // save starting point for later use
      globs.ptparams=motstruct+nframes*cnp;
      nvars=nframes*cnp;

      if(expert)
        n=sba_mot_levmar_x(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                          fixedcal? img_projsRT_x : (havedist? img_projsKDRT_x : img_projsKRT_x),
                          analyticjac? (fixedcal? img_projsRT_jac_x : (havedist? img_projsKDRT_jac_x : img_projsKRT_jac_x)) : nullptr,
                          (void *)(&globs), MAXITER, verbose, opts, info);
      else
        n=sba_mot_levmar(numpts3D, nframes, nconstframes, vmask, motstruct, cnp, imgpts, covimgpts, mnp,
                        fixedcal? img_projRT : (havedist? img_projKDRT : img_projKRT),
                        analyticjac? (fixedcal? img_projRT_jac : (havedist? img_projKDRT_jac : img_projKRT_jac)) : nullptr,
                        (void *)(&globs), MAXITER, verbose, opts, info);

      if((phi=info[1]/numprojs)>SBA_MAX_REPROJ_ERROR){
		#if BA_DEBUG
        fflush(stdout); fprintf(stdout, "Refining structure (motion only error %g)...\n", phi); fflush(stdout);
		#endif
		motstruct_after_mot = true;
        memcpy(motstruct, motstruct_copy, (nframes*cnp + numpts3D*pnp)*sizeof(double)); // reset starting point

        if(expert)
          n=sba_motstr_levmar_x(numpts3D, nconst3Dpts, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                                fixedcal? img_projsRTS_x : (havedist? img_projsKDRTS_x : img_projsKRTS_x),
                                analyticjac? (fixedcal? img_projsRTS_jac_x : (havedist? img_projsKDRTS_jac_x : img_projsKRTS_jac_x)) : nullptr,
                                (void *)(&globs), MAXITER2, verbose, opts, info);
        else
          n=sba_motstr_levmar(numpts3D, nconst3Dpts, nframes, nconstframes, vmask, motstruct, cnp, pnp, imgpts, covimgpts, mnp,
                              fixedcal? img_projRTS : (havedist? img_projKDRTS : img_projKRTS),
                              analyticjac? (fixedcal? img_projRTS_jac : (havedist? img_projKDRTS_jac : img_projKRTS_jac)) : nullptr,
                              (void *)(&globs), MAXITER2, verbose, opts, info);
      }
      free(motstruct_copy);

    break;

    default:
      /*fprintf(stderr, "unknown BA method \"%d\" in sba_driver()!\n", howto);
      exit(1);*/
	  ret = -1;
	  goto cleanup;
  }
  #if BA_DEBUG
  end_time=clock();
  #endif

  if(n==SBA_ERROR) //goto cleanup;
  {
	  ret = -3;
	  goto cleanup;
  }
  
  #if BA_DEBUG
  fflush(stdout);
  fprintf(stdout, "SBA using %d 3D pts, %d frames and %d image projections, %d variables\n", numpts3D, nframes, numprojs, nvars);
  if(havedist) sprintf(tbuf, " (%d fixed)", globs.ncdist);
  fprintf(stdout, "\nMethod %s, %s driver, %s Jacobian, %s covariances, %s distortion%s, %s intrinsics", howtoname[howto],
                  expert? "expert" : "simple",
                  analyticjac? "analytic" : "approximate",
                  covimgpts? "with" : "without",
                  havedist? "variable" : "without",
                  havedist? tbuf : "",
                  fixedcal? "fixed" : "variable");
  if(!fixedcal) fprintf(stdout, " (%d fixed)", globs.nccalib);
  fputs("\n\n", stdout); 
  fprintf(stdout, "SBA returned %d in %g iter, reason %g, error %g [initial %g], %d/%d func/fjac evals, %d lin. systems\n", n,
                    info[5], info[6], info[1]/numprojs, info[0]/numprojs, (int)info[7], (int)info[8], (int)info[9]);
  fprintf(stdout, "Elapsed time: %.2lf seconds, %.2lf msecs\n", ((double) (end_time - start_time)) / CLOCKS_PER_SEC,
                  ((double) (end_time - start_time)) / CLOCKS_PER_MSEC);
  fflush(stdout);
  #endif


  /* refined motion and structure are now in motstruct */

  switch(prnt){
    case BA_NONE:
		goto cleanup;
    break;

    case BA_MOTSTRUCT:
		if(writeMotToOutput(motstruct,initrot,cnp,nframes,
							useInputVarsAsOutput ? &Rquats : &Rquats_out,
							useInputVarsAsOutput ? &trans : &trans_out,
							useInputVarsAsOutput) < 0) goto cleanup;
		writeStructToOutput(motstruct,cnp,pnp,nframes,numpts3D,useInputVarsAsOutput ? pts3D : pts3D_out);
		if(intrParms)
		{
			if(writeIntrinsicsToOutput(motstruct,cnp,nframes,
									   useInputVarsAsOutput ? intrParms : &intrParms_out,
									   useInputVarsAsOutput) < 0)
			{
				/*for(int j = 0; j < nframes; ++j)
				{
					free(Rquats_out.at(j));
					free(trans_out.at(j));
				}
				Rquats_out.clear();
				trans_out.clear();*/
				goto cleanup;
			}
			if(dist)
			{
				if(writeDistToOutput(motstruct,cnp,nframes,useInputVarsAsOutput ? dist : &dist_out,useInputVarsAsOutput) < 0)
				{
					/*for(int j = 0; j < nframes; ++j)
					{
						free(Rquats_out.at(j));
						free(trans_out.at(j));
						free(intrParms_out.at(j));
					}
					Rquats_out.clear();
					trans_out.clear();
					intrParms_out.clear();*/
					goto cleanup;
				}
			}
		}
    break;

    case BA_MOT:
		if(writeMotToOutput(motstruct,initrot,cnp,nframes,
							useInputVarsAsOutput ? &Rquats : &Rquats_out,
							useInputVarsAsOutput ? &trans : &trans_out,
							useInputVarsAsOutput) < 0) goto cleanup;
		if(intrParms)
		{
			if(writeIntrinsicsToOutput(motstruct,cnp,nframes,
									   useInputVarsAsOutput ? intrParms : &intrParms_out,
									   useInputVarsAsOutput) < 0)
			{
				/*for(int j = 0; j < nframes; ++j)
				{
					free(Rquats_out.at(j));
					free(trans_out.at(j));
				}
				Rquats_out.clear();
				trans_out.clear();*/
				goto cleanup;
			}
			if(dist)
			{
				if(writeDistToOutput(motstruct,cnp,nframes,useInputVarsAsOutput ? dist : &dist_out,useInputVarsAsOutput) < 0)
				{
					/*for(int j = 0; j < nframes; ++j)
					{
						free(Rquats_out.at(j));
						free(trans_out.at(j));
						free(intrParms_out.at(j));
					}
					Rquats_out.clear();
					trans_out.clear();
					intrParms_out.clear();*/
					goto cleanup;
				}
			}
		}
    break;

    case BA_STRUCT:
		writeStructToOutput(motstruct,cnp,pnp,nframes,numpts3D,useInputVarsAsOutput ? pts3D : pts3D_out);
    break;

	case BA_MOT_MOTSTRUCT:
		if(writeMotToOutput(motstruct,initrot,cnp,nframes,
							useInputVarsAsOutput ? &Rquats : &Rquats_out,
							useInputVarsAsOutput ? &trans : &trans_out,
							useInputVarsAsOutput) < 0) goto cleanup;
		if(motstruct_after_mot)
			writeStructToOutput(motstruct,cnp,pnp,nframes,numpts3D,useInputVarsAsOutput ? pts3D : pts3D_out);
		if(intrParms)
		{
			if(writeIntrinsicsToOutput(motstruct,cnp,nframes,
									   useInputVarsAsOutput ? intrParms : &intrParms_out,
									   useInputVarsAsOutput) < 0)
			{
				/*for(int j = 0; j < nframes; ++j)
				{
					free(Rquats_out.at(j));
					free(trans_out.at(j));
				}
				Rquats_out.clear();
				trans_out.clear();*/
				goto cleanup;
			}
			if(dist)
			{
				if(writeDistToOutput(motstruct,cnp,nframes,useInputVarsAsOutput ? dist : &dist_out,useInputVarsAsOutput) < 0)
				{
					/*for(int j = 0; j < nframes; ++j)
					{
						free(Rquats_out.at(j));
						free(trans_out.at(j));
						free(intrParms_out.at(j));
					}
					Rquats_out.clear();
					trans_out.clear();
					intrParms_out.clear();*/
					goto cleanup;
				}
			}
		}
	break;

    default:
		ret = -1;
#if BA_DEBUG
        fprintf(stderr, "unknown print option \"%d\" in sba_driver()!\n", prnt);
        //exit(1);
#endif
  }


cleanup:
  /* just in case... */
  globs.intrcalib=nullptr;
  globs.nccalib=0;
  globs.ncdist=0;

  if(motstruct) free(motstruct);
  if(imgpts) free(imgpts);
  if(initrot) free(initrot); 
  globs.rot0params=nullptr;
  if(covimgpts) free(covimgpts);
  if(vmask) free(vmask);
  if(globs.intrcalib) free(globs.intrcalib);

return ret;
}

/* Extracts the rotation and translation from the parameter vector deliverd by the Sparse Bundleadjustment
 * Algorithm (SBA).
 *
 * double *motstruct				Input  -> Pointer to the parameter array from SBA
 * double *initrot					Input  -> Pointer to the array of initial (before SBA) rotation quaternions
 * int cnp							Input  -> Number of parameters per camera in the motstruct array
 * int nframes						Input  -> Number of frames
 * vector<double *> *Rquats_out		Output -> Pointer to the vector of rotation quaternions (stored in arrays
 *											  with size 4 -> [w,x,y,z])
 * vector<double *> *trans_out		Output -> Pointer to the vector of translations (stored in arrays
 *											  with size 3 -> [x,y,z])
 * bool useInpAsOutp				Input  -> Specifies if the data from SBA is copied back to the provided 
 *											  initial data structures (already allocated memory)
 *
 * Return value:					 0:		  Everything ok
 *									-1:		  Memory allocation failed
 */
int writeMotToOutput(double *motstruct, double *initrot, int cnp, int nframes, 
					 std::vector<double *> *Rquats_out, std::vector<double *> *trans_out, bool useInpAsOutp)
{
	for(int i = 0; i < nframes; ++i)
	{

		double *v, qs[FULLQUATSZ], *q0, *prd, *trans;

		/** Get the rotational part **/
		if(useInpAsOutp)
			prd = Rquats_out->at(i);
		else
		{
			prd = (double *)malloc(FULLQUATSZ*sizeof(double));
			if(prd == nullptr)
			{
				for(int j = 0; j < i; ++j)
				{
					free(Rquats_out->at(j));
					free(trans_out->at(j));
				}
				Rquats_out->clear();
				trans_out->clear();
				return -1; //Memory allocation failed
			}
		}

		/* retrieve the vector part */
		v=motstruct + (i+1)*cnp - 6; // note the +1, we access the motion parameters from the right, assuming 3 for translation!
		_MK_QUAT_FRM_VEC(qs, v);

		q0=initrot+i*FULLQUATSZ;
		quatMultFast(qs, q0, prd); // prd=qs*q0

		/* make sure that the scalar part is non-negative */
		if(prd[0] < 0.0)
		{
		// negate since two quaternions q and -q represent the same rotation
			prd[0] *= -1.0;
			prd[1] *= -1.0;
			prd[2] *= -1.0;
			prd[3] *= -1.0;
		}
		if(!useInpAsOutp)
			Rquats_out->push_back(prd);

		/** Get the translational part **/

		if(useInpAsOutp)
			trans = trans_out->at(i);
		else
		{
			trans = (double *)malloc(3*sizeof(double));
			if(trans == nullptr)
			{
				free(Rquats_out->at(i));
				for(int j = 0; j < i; ++j)
				{
					free(Rquats_out->at(j));
					free(trans_out->at(j));
				}
				Rquats_out->clear();
				trans_out->clear();
				return -1; //Memory allocation failed
			}
		}

		v += 3; //skip the 3 parameters for rotation

		memcpy(trans,v,3*sizeof(double));
		if(!useInpAsOutp)
			trans_out->push_back(trans);
	}
	return 0;
}

/* Extracts the structure (e.g. 3D scene points) from the parameter vector deliverd by the Sparse Bundleadjustment
 * Algorithm (SBA).
 *
 * double *motstruct				Input  -> Pointer to the parameter array from SBA
 * int cnp							Input  -> Number of parameters per camera in the motstruct array
 * int pnp							Input  -> Number of parameters per structure element (e.g. 3D scene point)
 * int nframes						Input  -> Number of frames
 * int numpts3D						Input  -> Number of structure elements (e.g. 3D scene points)
 * double *pts3D_out				Output -> Pointer to the array of structure elements
 *
 * Return value:					none
 */
void writeStructToOutput(double *motstruct, int cnp, int pnp, int nframes, int numpts3D, double *pts3D_out)
{
	double *v = motstruct + nframes * cnp;

	memcpy(pts3D_out,v,numpts3D*pnp*sizeof(double));
}

/* Extracts the camera intrinsics from the parameter vector deliverd by the Sparse Bundleadjustment
 * Algorithm (SBA).
 *
 * double *motstruct				Input  -> Pointer to the parameter array from SBA
 * int cnp							Input  -> Number of parameters per camera in the motstruct array
 * int nframes						Input  -> Number of frames
 * vector<double *> *intrParms_out	Output -> Pointer to the vector of camera intriniscs (stored
 *											  in arrays with size 5 -> [fu,u0,v0,ar,s] with ar=fv/fu)
 * bool useInpAsOutp				Input  -> Specifies if the data from SBA is copied back to the provided 
 *											  initial data structures (already allocated memory)
 *
 * Return value:					 0:		  Everything ok
 *									-1:		  Memory allocation failed
 */
int writeIntrinsicsToOutput(double *motstruct, int cnp, int nframes, std::vector<double *> *intrParms_out, bool useInpAsOutp)
{
	for(int i = 0; i < nframes; ++i)
	{
		double *intr, *v;

		v = motstruct + i*cnp;

		if(useInpAsOutp)
			memcpy(intrParms_out->at(i),v,5*sizeof(double));
		else
		{
			intr = (double *)malloc(5*sizeof(double));
			if(intr == nullptr)
			{
				for(int j = 0; j < i; ++j)
				{
					free(intrParms_out->at(j));
				}
				intrParms_out->clear();
				return -1; //Memory allocation failed
			}
			memcpy(intr,v,5*sizeof(double));
			intrParms_out->push_back(intr);
		}
	}
	return 0;
}

/* Extracts the camera intrinsics from the parameter vector deliverd by the Sparse Bundleadjustment
 * Algorithm (SBA).
 *
 * double *motstruct				Input  -> Pointer to the parameter array from SBA
 * int cnp							Input  -> Number of parameters per camera in the motstruct array
 * int nframes						Input  -> Number of frames
 * vector<double *> *dist_out		Output -> Pointer to the vector of distortion coeffitients (stored in 
 *											  arrays with size 5 -> [k1,k2,k3,k4,k5] with k1 (2nd order), 
 *											  k2 (4th order) and k5 (6th order) beeing the radial and
 *											  k3 and k4 beeing the tangential distortion coeffitients.
 * bool useInpAsOutp				Input  -> Specifies if the data from SBA is copied back to the provided 
 *											  initial data structures (already allocated memory)
 *
 * Return value:					 0:		  Everything ok
 *									-1:		  Memory allocation failed
 */
int writeDistToOutput(double *motstruct, int cnp, int nframes, std::vector<double *> *dist_out, bool useInpAsOutp)
{
	for(int i = 0; i < nframes; ++i)
	{
		double *disto, *v;

		v = motstruct + i*cnp + 5;

		if(useInpAsOutp)
			memcpy(dist_out->at(i),v,5*sizeof(double));
		else
		{
			disto = (double *)malloc(5*sizeof(double));
			if(disto == nullptr)
			{
				for(int j = 0; j < i; ++j)
				{
					free(dist_out->at(j));
				}
				dist_out->clear();
				return -1; //Memory allocation failed
			}
			memcpy(disto,v,5*sizeof(double));
			dist_out->push_back(disto);
		}
	}
	return 0;
}


/* Changes the estimated image projection in such a way that the cost function of the SBA is changed from
 * squared norm to pseudo-huber.
 *
 * double *ImgProjEsti				Input & Output  -> Pointer to the estimated image projection
 * double *ImgProjMeas				Input  -> Pointer to the measured image projection
 * int mnp							Input  -> Number of coordinates per image projection
 *
 * Return value:					none
 */
void convertCostToPseudoHuber(double *ImgProjEsti, double *ImgProjMeas, int mnp, double thresh)
{
	double deltanorm = 0;
	double weight;

	for(int i = 0; i < mnp; ++i)
	{
		deltanorm += sqr(ImgProjMeas[i] - ImgProjEsti[i]);
	}
	deltanorm = std::sqrt(deltanorm);

    weight =  costPseudoHuber(deltanorm, thresh);

	//As we cannot influence the squared norm directly with the cost function, we have to do this over the estimated projection
	for(int i = 0; i < mnp; ++i)
	{
		ImgProjEsti[i] = (1-weight) * ImgProjMeas[i] + weight * ImgProjEsti[i];
	}
}

/* Calculates the weight using the pseudo-huber cost function.
 * 
 * const double d				Input  -> Error value
 * const double thresh			Input  -> Threshold
 *
 * Return value:				Weighting factor resulting from the pseudo-huber cost function
 */
double costPseudoHuber(const double d, const double thresh)
{
	const double b_sq = sqr(thresh);
    const double d_abs = std::abs(d) + 1e-12;

    //C(deltanorm) = 2*b^2*(sqrt(1+(delta/b)^2) - 1);
	//weight = sqrt(C(deltanorm))/deltanorm; -> to compensate original least squares algorithm
	//const double weight = 2 * b_sq * (std::sqrt(1 + sqr(d / thresh)) - 1);
    return std::sqrt(2 * b_sq * (std::sqrt(1 + sqr(d_abs / thresh)) - 1)) / d_abs;
}


/* Squares the provided value
 * 
 * const double var				Input  -> Value to be squared
 *
 * Return value:				Squared value
 */
inline double sqr(const double var)
{
	return var * var;
}

}