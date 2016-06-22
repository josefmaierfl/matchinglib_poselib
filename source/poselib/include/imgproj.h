/**********************************************************************************************************
 FILE: imgproj.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: May 2016

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: Functions for calculating the exact derivatives for SBA
**********************************************************************************************************/

#pragma once

/* --------------------------- Defines --------------------------- */


/* --------------------- Function prototypes --------------------- */

#ifdef __cplusplus
extern "C" {
#endif
/* in imgproj.c */
void calcImgProj(double a[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]);
void calcImgProjNoK(double qr0[4],double v[3],double t[3],double M[3], double n[2]);
void calcImgProjFullR(double a[5], double qr0[4], double t[3], double M[3], double n[2]);
void calcImgProjFullRnoK(double qr0[4],double t[3],double M[3], double n[2]);
void calcImgProjJacKRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11], double jacmS[2][3]);
void calcImgProjJacKRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11]);
void calcImgProjJacS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);
void calcImgProjJacSnoK(double qr0[4],double v[3],double t[3], double M[3],double jacmS[2][3]);
void calcImgProjJacRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6], double jacmS[2][3]);
void calcImgProjJacRTSnoK(double qr0[4],double v[3],double t[3], double M[3],double jacmRT[2][6],double jacmS[2][3]);
void calcImgProjJacRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6]);
void calcImgProjJacRTnoK(double qr0[4],double v[3],double t[3], double M[3],double jacmRT[2][6]);
void calcDistImgProj(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]);
void calcDistImgProjFullR(double a[5], double kc[5], double qr0[4], double t[3], double M[3], double n[2]);
void calcDistImgProjJacKDRTS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16], double jacmS[2][3]);
void calcDistImgProjJacKDRT(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16]);
void calcDistImgProjJacS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);

#ifdef __cplusplus
}
#endif