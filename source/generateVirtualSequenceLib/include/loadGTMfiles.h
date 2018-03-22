/**********************************************************************************************************
FILE: loadGTMfiles.h

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: May 2017

LOCATION: TechGate Vienna, Donau-City-Straﬂe 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for loading the GTMs.
**********************************************************************************************************/

#pragma once

#include "glob_includes.h"
#include "generateVirtualSequenceLib\generateVirtualSequenceLib_api.h"

int GENERATEVIRTUALSEQUENCELIB_API showGTM(std::string img_path, std::string l_img_pref, std::string r_img_pref,
	std::string gtm_path, std::string gtm_postfix);