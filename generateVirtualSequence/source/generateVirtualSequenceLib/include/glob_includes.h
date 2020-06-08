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

#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
//#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>       /* isnan, sqrt */
#include <utility>
#include "generateVirtualSequenceLib/generateVirtualSequenceLib_api.h"

enum GENERATEVIRTUALSEQUENCELIB_API vorboseType
{
    SHOW_INIT_CAM_PATH = 0x01,
    SHOW_BUILD_PROC_MOV_OBJ = 0x02,
    SHOW_MOV_OBJ_DISTANCES = 0x04,
    SHOW_MOV_OBJ_3D_PTS = 0x08,
    SHOW_MOV_OBJ_CORRS_GEN = 0x10,
    SHOW_BUILD_PROC_STATIC_OBJ = 0x20,
    SHOW_STATIC_OBJ_DISTANCES = 0x40,
    SHOW_STATIC_OBJ_CORRS_GEN = 0x80,
    SHOW_STATIC_OBJ_3D_PTS = 0x100,
    SHOW_MOV_OBJ_MOVEMENT = 0x200,
    SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ = 0x400,
    SHOW_BACKPROJECT_OCCLUSIONS_STAT_OBJ = 0x800,
    SHOW_BACKPROJECT_MOV_OBJ_CORRS = 0x1000,
    SHOW_STEREO_INTERSECTION = 0x2000,
    SHOW_COMBINED_CORRESPONDENCES = 0x4000,
    PRINT_WARNING_MESSAGES = 0x8000,
    SHOW_IMGS_AT_ERROR = 0x10000,
    SHOW_PLANES_FOR_HOMOGRAPHY = 0x20000,
    SHOW_WARPED_PATCHES = 0x40000,
    SHOW_PATCHES_WITH_NOISE = 0x80000,
    SHOW_GTM_KEYPOINTS = 0x100000,
    SHOW_GTM_GEN_PROCESS = 0x200000,
    SHOW_GTM_INTERPOL_FLOW = 0x400000
};