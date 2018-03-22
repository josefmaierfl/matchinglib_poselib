/**********************************************************************************************************
 FILE: eval_start.h

 PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

 CODE: C++
 
 AUTOR: Josef Maier, AIT Austrian Institute of Technology

 DATE: September 2015

 LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

 VERSION: 1.0

 DISCRIPTION: This file provides functionalities for starting the different matching algorithms and to 
			  perform different measurements.
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "matchtestlib\matchtestlib_api.h"

/* --------------------------- Defines --------------------------- */

/* --------------------- Function prototypes --------------------- */

int MATCHTESTLIB_API startTimeMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH, 
						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
						 std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
						 bool useRatioTest, std::string storeResultPath, bool refine = false, double inlRatio = 1.0, 
						 int showResult = -1, int showRefinedResult = -1, std::string storeImgResPath = "", 
						 std::string storeRefResPath = "", std::string idxPars_NMSLIB = "", std::string queryPars_NMSLIB = "");

int MATCHTESTLIB_API startTimeMeasurementDiffInlRats(std::string imgsPath, std::string flowDispHPath, int flowDispH,
										std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
										std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
										bool useRatioTest, std::string storeResultPath, bool useSameKeyPSiAllInl = false,
										int showResult = -1, std::string storeImgResPath = "");

int MATCHTESTLIB_API startInlierRatioMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH,
								std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
								std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
								bool useRatioTest, std::string storeResultPath, int imgidx1, int imgidx2 = -1, bool refine = false, 
								int showResult = -1, int showRefinedResult = -1, std::string storeImgResPath = "", 
								std::string storeRefResPath = "");

int MATCHTESTLIB_API startInlierRatioMeasurementWholeSzene(std::string imgsPath, std::string flowDispHPath, int flowDispH,
										std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
										std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
										bool useRatioTest, std::string storeResultPath, bool refine = false, 
										int showResult = -1, int showRefinedResult = -1, std::string storeImgResPath = "", 
										std::string storeRefResPath = "", std::string idxPars_NMSLIB = "", std::string queryPars_NMSLIB = "");

int MATCHTESTLIB_API startQualPMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH,
						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
						 std::string featureDetector, std::string descriptorExtractor, std::string matcherType,
						 bool useRatioTest, std::string storeResultPath, bool refine = false, double inlRatio = 1.0, 
						 int showResult = -1, int showRefinedResult = -1, std::string storeImgResPath = "", 
						 std::string storeRefResPath = "");

int MATCHTESTLIB_API testGMbSOFthreshold(std::string imgsPath, std::string flowDispHPath, int flowDispH,
						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
						 std::string featureDetector, std::string descriptorExtractor,
						 std::string storeResultPath, int showResult = -1, std::string storeImgResPath = "");

int MATCHTESTLIB_API testGMbSOFsearchRange(std::string imgsPath, std::string flowDispHPath, int flowDispH,
						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
						 std::string featureDetector, std::string descriptorExtractor,
						 std::string storeResultPath, double inlRatio = -1.0, double validationTh = 0.3, int showResult = -1, 
						 std::string storeImgResPath = "");

int MATCHTESTLIB_API testGMbSOFinitMatching(std::string imgsPath, std::string flowDispHPath, int flowDispH,
						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
						 std::string featureDetector, std::string descriptorExtractor,
						 std::string storeResultPath, int showResult = -1, std::string storeImgResPath = "");

int MATCHTESTLIB_API testGMbSOF_CDratios(std::string imgsPath, std::string flowDispHPath, int flowDispH,
						 std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
						 std::string featureDetector, std::string descriptorExtractor,
						 std::string storeResultPath, int showResult = -1, std::string storeImgResPath = "");

int MATCHTESTLIB_API testGTmatches(std::string imgsPath, std::string flowDispHPath, int flowDispH,
				  std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
				  std::string featureDetector, std::string storeResultPath, std::string descriptorExtractorGT = "FREAK", double threshhTh = 64.0);

int MATCHTESTLIB_API startDescriptorTimeMeasurement(std::string imgsPath, std::string flowDispHPath, int flowDispH,
	std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
	std::string featureDetector, std::string descriptorExtractor, std::string storeResultPath);

int MATCHTESTLIB_API generateMissingInitialGTMs(std::string imgsPath, std::string flowDispHPath, int flowDispH,
	std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
	std::string featureDetector, std::string descriptorExtractorGT = "FREAK");

int MATCHTESTLIB_API generateGTMs(std::string imgsPath, std::string flowDispHPath, int flowDispH,
	std::string filePrefImgL, std::string filePrefImgR, std::string filePrefFlowDispH,
	std::string featureDetector, std::string descriptorExtractorGT = "FREAK");