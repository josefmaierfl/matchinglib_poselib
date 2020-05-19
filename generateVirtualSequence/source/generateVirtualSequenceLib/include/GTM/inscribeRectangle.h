//Auszug von Oliver Zendel's SequenceCalib
#pragma once

#include "glob_includes.h"
#include "opencv2/highgui/highgui.hpp"


void intersectPolys(const std::vector<cv::Point2d> &pntsPoly1, const std::vector<cv::Point2d> &pntsPoly2, std::vector<cv::Point2d> &pntsRes);
bool maxInscribedRect(std::vector<cv::Point2d> &polygon, cv::Point2d &retLT, cv::Point2d &retRB);