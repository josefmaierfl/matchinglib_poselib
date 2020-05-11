//Auszug von Oliver Zendel's SequenceCalib

#include "GTM/inscribeRectangle.h"
#include "polygon_helper.h"


using namespace cv;
using namespace std;


bool seqDetExpandRectAxis(std::vector<cv::Point2d> &pts_rect, const cv::Point2d& centralPnt, double lenInters, cv::Point2d &lt, cv::Point2d &rb);
//getPolylineIntersection return values:
// -1: if there is no intersection of poly with the line segment p0-p1 (lies completely inside or outside of poly)
//  0: if p0 was outside, p1 inside; p0 is replaced by the intersection of the line segment and the polygon
//  1: if p0 was inside, p1 outside; p1 is replaced by the intersection of the line segment and the polygon
//  2: if p0 and p1 are outside but line is intersecting with the polygon, p0 and p1 are replaced by the intersections of the line segment and the polygon
static int getPolylineIntersection(std::vector<cv::Point2d>  &poly, cv::Point2d &p0, cv::Point2d &p1);
bool seqDetExpandRectDiagonals(std::vector<cv::Point2d> &pts_rect, cv::Point2d &lt, cv::Point2d &rb, double lenInters);
bool seqDetExpandSides(std::vector<cv::Point2d> &pts_rect, cv::Point2d &lt, cv::Point2d &rb, double dStepRed);
bool seqDetCorrectCollisions(cv::Point2d &LT, cv::Point2d &RB, std::vector<cv::Point2d> &pts_rect, bool bTryFix=true);
double seqDetCalcIntersectionStrict(const Point2d& o1, const Point2d& p1, const Point2d& o2, const Point2d& p2, Point2d &r);
double seqDetMySimpleExtrEuclidianSquared(const cv::Point2d& pt);
double seqDetArea(cv::Point2d &lt, cv::Point2d &rb);


void intersectPolys(const std::vector<cv::Point2d> &pntsPoly1, const std::vector<cv::Point2d> &pntsPoly2, std::vector<cv::Point2d> &pntsRes)
{
  polygon_contour c1={0, nullptr, nullptr}, c2={0, nullptr, nullptr}, res={0, nullptr, nullptr};
  add_contour_from_array((double*)&pntsPoly1[0].x, (int)pntsPoly1.size(), &c1);
  add_contour_from_array((double*)&pntsPoly2[0].x, (int)pntsPoly2.size(), &c2);
  calc_polygon_clipping(POLYGON_OP_INTERSECT, &c1, &c2, &res);

  if(res.num_contours > 0)
  {
    double areaRes = 0;
    for(int c = 0; c < res.num_contours; ++c)
    {
      for(int v = 0; v < res.contour[c].num_vertices; ++v)
        pntsRes.emplace_back(res.contour[c].vertex[v].x,res.contour[c].vertex[v].y);
        
      /*double areaRes1 = cv::contourArea(pntsRes);
      areaRes += areaRes1;*/
    }
  }

  free_polygon(&c1);
  free_polygon(&c2);
  free_polygon(&res);
}

bool maxInscribedRect(std::vector<cv::Point2d> &pts_rect, cv::Point2d &retLT, cv::Point2d &retRB)
{
  const int maxItters     = 8;
  const int maxNumPntsTry = 6;

  if(pts_rect.size()<4)
    return false;
  //robustly find center and extent of polyline
  std::vector<double> sortx,sorty;
  double meanLen=0, meanX = 0, meanY = 0;
  int lastIdx = 0;
  std::vector<double> lenPrev;
  std::vector<cv::Point2d> diffPrev;

  double minX = DBL_MAX, maxX = -DBL_MAX, minY = DBL_MAX, maxY = -DBL_MAX;
  for(size_t j = 0; j < pts_rect.size(); j++)
  {
    minX = min(minX,pts_rect[j].x);
    maxX = max(maxX,pts_rect[j].x);
    minY = min(minY,pts_rect[j].y);
    maxY = max(maxY,pts_rect[j].y);

    cv::Point2d dist1;
    if(j==0)
      dist1 = pts_rect[j]-pts_rect[pts_rect.size()-1];
    else
      dist1 = pts_rect[j]-pts_rect[j-1];

    double dist1Len = sqrt(dist1.x*dist1.x+dist1.y*dist1.y);
    lenPrev.push_back(dist1Len);
    diffPrev.push_back(dist1);

    meanX += pts_rect[j].x*dist1Len;
    meanY += pts_rect[j].y*dist1Len;
    meanLen += dist1Len;
    cv::Point2d dist2 = pts_rect[j]-pts_rect[lastIdx];
    if((abs(dist2.x)+abs(dist2.y))>1)
    {
      sortx.push_back(pts_rect[j].x);
      sorty.push_back(pts_rect[j].y);
      lastIdx = j;
    }
  }
  if(meanLen <= 0)
    return false;
  
  double polyW = maxX-minX, polyH = maxY-minY;
  double lenInters = 2.0*sqrt(polyW*polyW+polyH*polyH);

  std::vector<cv::Point2d> tryPnts;
  //determine medianPnt for this polygon
  cv::Point2d meanPnt(meanX/meanLen,meanY/meanLen),medianPnt;
  if((sortx.size()>1)
    && (sorty.size()>1))
  {
    std::sort(sortx.begin(),sortx.end());
    std::sort(sorty.begin(),sorty.end());
    medianPnt = cv::Point2d(sortx[sortx.size()/2],sorty[sorty.size()/2]);
    tryPnts.push_back(medianPnt);
  }
  tryPnts.push_back(meanPnt);
  tryPnts.emplace_back(minX + polyW/2, minY + polyH/2);

  cv::Point2d bestLT(0,0), bestRB(0,0);
  for(size_t i=0; i<tryPnts.size();++i)
  {
    cv::Point2d lt,rb;
    bool bUpdatedRc = false;
    bool bRefinement = seqDetExpandRectAxis(pts_rect, tryPnts[i], lenInters, lt, rb);
    int j = 0;
    while(bRefinement
      && (j < maxItters)
      && (seqDetArea(bestLT,bestRB) < seqDetArea(lt,rb))
      )
    {
      bestLT = lt;
      bestRB = rb;
      bUpdatedRc = true;
      bRefinement = seqDetExpandRectDiagonals(pts_rect, lt, rb, lenInters);
    }

    lt = tryPnts[i];
    rb = cv::Point2d(tryPnts[i].x+3, tryPnts[i].y+3);
    j = 0;
    bRefinement = seqDetExpandRectDiagonals(pts_rect, lt, rb, lenInters);
    while(bRefinement
      && (j < maxItters)
      && (seqDetArea(bestLT,bestRB) < seqDetArea(lt,rb))
      )
    {
      bestLT = lt;
      bestRB = rb;
      bUpdatedRc = true;
      bRefinement = seqDetExpandRectDiagonals(pts_rect, lt, rb, lenInters);
    }

    if(bUpdatedRc &&(tryPnts.size() < maxNumPntsTry))
    {
      double bestW = bestRB.x-bestLT.x, bestH = bestRB.y-bestLT.y;
      tryPnts.emplace_back(bestLT.x+bestW/4,bestLT.y+bestH/4);
      tryPnts.emplace_back(bestLT.x+bestW*3/4,bestLT.y+bestH/4);
      tryPnts.emplace_back(bestLT.x+bestW/4,bestLT.y+bestH*3/4);
      tryPnts.emplace_back(bestLT.x+bestW*3/4,bestLT.y+bestH*3/4);
    }
  }
  
  double bestW = bestRB.x-bestLT.x, bestH = bestRB.y-bestLT.y;
  if(bestW<= 0 || bestH <= 0)
    return false;

  retLT = bestLT;
  retRB = bestRB;
  return true;
}

bool seqDetExpandRectAxis(std::vector<cv::Point2d> &pts_rect, const cv::Point2d& centralPnt, double lenInters, cv::Point2d &lt, cv::Point2d &rb)
{
  cv::Point2d offsPnts[4]={cv::Point2d(0,-lenInters),
                          cv::Point2d(0,lenInters),
                          cv::Point2d(-lenInters,0),
                          cv::Point2d(lenInters,0)};
  std::vector<cv::Point2d> resPnts;
  for(auto & offsPnt : offsPnts)
  {
    cv::Point2d checkPnt = centralPnt+offsPnt, cntrPnt = centralPnt;
    int retIdx = getPolylineIntersection(pts_rect, cntrPnt, checkPnt);
    if(retIdx == 0)
      resPnts.push_back(cntrPnt);
    else if(retIdx == 1)
      resPnts.push_back(checkPnt);
    else
      return false; // invalid setup
  }

  lt = cv::Point2d(resPnts[2].x, resPnts[0].y);
  rb = cv::Point2d(resPnts[3].x, resPnts[1].y);
  return seqDetCorrectCollisions(lt, rb, pts_rect);
}

int getPolylineIntersection(std::vector<cv::Point2d>  &poly, cv::Point2d &p0, cv::Point2d &p1)
{
  //PERFTODO: unroll call to seqDetCalcIntersection
  int numElems = (int)poly.size();
  int i= 0;
  int iPrev = numElems-1;
  auto dIntersect0 = DBL_MAX, dIntersect1 = DBL_MAX;
  cv::Point2d pRepl0 = p0, pRepl1 = p1;
  do
  {
    cv::Point2d inters;
    double tLen = seqDetCalcIntersectionStrict(p0, p1, poly[iPrev], poly[i], inters);
    if((tLen >= 0.0)
      && (tLen <= 1.0))
    {
      double checkDist0 = seqDetMySimpleExtrEuclidianSquared(p0-inters);
      double checkDist1 = seqDetMySimpleExtrEuclidianSquared(p1-inters);
      if(checkDist0 < checkDist1)
      {
        if(checkDist0 < dIntersect0)
        {
          pRepl0 = inters;
          dIntersect0 = checkDist0;
        }
      }
      else
      {
        if(checkDist1 < dIntersect1)
        {
          pRepl1 = inters;
          dIntersect1 = checkDist1;
        }
      }
    }
    iPrev = i++;
  }
  while(i<numElems);

  double diffPPnts0 = seqDetMySimpleExtrEuclidianSquared(p0 - pRepl0), diffPPnts1 = seqDetMySimpleExtrEuclidianSquared(p1 - pRepl1);
  const double dMinEpsilon = 0.0;
  bool bIntersect0 = (diffPPnts0>dMinEpsilon), bIntersect1 = (diffPPnts1>dMinEpsilon);
  if(bIntersect0)
    p0 = pRepl0;
  if(bIntersect1)
    p1 = pRepl1;

  if(bIntersect0 && bIntersect1)
    return 2;
  else if(bIntersect0)
    return 0;
  else if(bIntersect1)
    return 1;

  return -1;
}

bool seqDetExpandRectDiagonals(std::vector<cv::Point2d> &pts_rect, cv::Point2d &lt, cv::Point2d &rb, double lenInters)
{
  double cW = rb.x-lt.x, cH = rb.y-lt.y;
  if(cW <= 0 || cH <= 0)
    return false;
  cv::Point2d centralPnt((lt.x+rb.x)/2,(lt.y+rb.y)/2);
  
  double lenIntersX=lenInters*cW/cH,lenIntersY=lenInters*cH/cW;
  cv::Point2d offsPnts[4]={cv::Point2d(lenIntersX,-lenIntersY),
                          cv::Point2d(-lenIntersX,-lenIntersY),
                          cv::Point2d(-lenIntersX,lenIntersY),
                          cv::Point2d(lenIntersX,lenIntersY)};
  std::vector<cv::Point2d> resPnts;
  for(auto & offsPnt : offsPnts)
  {
    cv::Point2d checkPnt = centralPnt+offsPnt, cntrPnt = centralPnt;
    int retIdx = getPolylineIntersection(pts_rect, cntrPnt, checkPnt);
    if(retIdx == 0)
      resPnts.push_back(cntrPnt);
    else if(retIdx == 1)
      resPnts.push_back(checkPnt);
    else
      return false; // invalid setup
  }
  
  lt = cv::Point2d((min(resPnts[1].x,resPnts[2].x)),(min(resPnts[1].y,resPnts[0].y)));
  rb = cv::Point2d((max(resPnts[0].x,resPnts[3].x)),(max(resPnts[2].y,resPnts[3].y)));

  if(seqDetCorrectCollisions(lt, rb, pts_rect))
  {
    const double dlenInters2 = max((rb.x-lt.x),(rb.y-lt.y))/1024;
    cv::Point2d  lt1 = lt, rb1 = rb;
    if(seqDetExpandSides(pts_rect,lt1,rb1,dlenInters2))
    {
      lt=lt1;
      rb=rb1;
    }
    return true;
  }
  else
    return false;
}

bool seqDetExpandSides(std::vector<cv::Point2d> &pts_rect, cv::Point2d &lt, cv::Point2d &rb, double dStepRed)
{
  const int max_itters = 32;
  const double advDistFactor = 1.0/4.0;
  double cW = rb.x-lt.x, cH = rb.y-lt.y, dEpsilonLen = dStepRed/256;
  if(cW <= 0 || cH <= 0)
    return false;
  std::vector<cv::Point2d> cornerPnts; //ordered by Quadrant -> RT,LT,LB,RB
  cornerPnts.emplace_back(rb.x,lt.y);
  cornerPnts.push_back(lt); 
  cornerPnts.emplace_back(lt.x,rb.y);
  cornerPnts.push_back(rb);

  for(int i = 0; i < max_itters; ++i)
  {
    bool bAllPntsInside = true;
    for(int j=0; j<4; ++j)
    {
      int jNxt = (j+1)%4;
      cv::Point2d p0 = cornerPnts[j], p1 = cornerPnts[jNxt];
      cv::Point2d p0Op = cornerPnts[(j+2)%4], p1Op = cornerPnts[(j+3)%4];
      bool bChangeX = (i%2 == 1), bChangeNeg = ((i==0)||(i==3));
      double dChangeVal = (bChangeNeg?-dStepRed:dStepRed);
      double dAdvVal = ((i<2)?-advDistFactor:advDistFactor)*(bChangeX?cW:cH);
      if(!bChangeX)
      {
        p0.x += dChangeVal;
        p1.x -= dChangeVal;
        p0Op.x += dChangeVal;
        p1Op.x -= dChangeVal;

        p0.y += dAdvVal;
        p1.y += dAdvVal;
      }
      else
      {
        p0.y += dChangeVal;
        p1.y -= dChangeVal;
        p0Op.y += dChangeVal;
        p1Op.y -= dChangeVal;

        p0.x += dAdvVal;
        p1.x += dAdvVal;
      }

      int retIdx0 = getPolylineIntersection(pts_rect, p0Op, p0);
      int retIdx1 = getPolylineIntersection(pts_rect, p1Op, p1);
      if((retIdx0 >= 1) && (retIdx1 >= 1))
      {
        if(bChangeX)
        {
          if(bChangeNeg)
            cornerPnts[j].x = cornerPnts[jNxt].x = max(p0.x,p1.x)+dEpsilonLen;
          else
            cornerPnts[j].x = cornerPnts[jNxt].x = min(p0.x,p1.x)-dEpsilonLen;
        }
        else
        {
          if(bChangeNeg)
            cornerPnts[j].y = cornerPnts[jNxt].y = max(p0.y,p1.y)+dEpsilonLen;
          else
            cornerPnts[j].y = cornerPnts[jNxt].y = min(p0.y,p1.y)-dEpsilonLen;
        }
      }
    }
    if(bAllPntsInside)
      break;
  }

  cv::Point2d retLT = cornerPnts[1], retRB = cornerPnts[3];
  if(!seqDetCorrectCollisions(retLT, retRB, pts_rect))
    return false;
  if(seqDetArea(lt,rb) >= seqDetArea(retLT,retRB))
    return false;
  lt = retLT;
  rb = retRB;
  return true;
}

bool seqDetCorrectCollisions(cv::Point2d &LT, cv::Point2d &RB, std::vector<cv::Point2d> &pts_rect, bool bTryFix)
{
  const int maxItter = 128;
  const int maxPartCorrect = 3;
  std::vector<cv::Point2d> cornerPnts; //ordered by Quadrant -> RT,LT,LB,RB
  cornerPnts.emplace_back(RB.x,LT.y);
  cornerPnts.push_back(LT); 
  cornerPnts.emplace_back(LT.x,RB.y);
  cornerPnts.push_back(RB);

  cv::Point2d cntrPntOrig((LT.x+RB.x)/2,(LT.y+RB.y)/2);
  bool bAllPntsInside = false;
  double corrLen = min(RB.x-LT.x, RB.y-LT.y)/(maxItter*maxPartCorrect);
  double epsVal = corrLen/256;
  for(int itter = 0; itter < maxItter; ++itter)
  {
    bAllPntsInside = true;
    for(int i=0; i<4; ++i) //check if all 4 sides are inside of the polygon
    {
      int iNxt = (i+1)%4;
      cv::Point2d c0 = cornerPnts[i], c1 = cornerPnts[iNxt];
      int retIdx = getPolylineIntersection(pts_rect, c0, c1);

      if(retIdx != -1)
      {
        bAllPntsInside = false;
        //TODO: use slope at intersection to guide reduction...
        bool bChangeX = (i%2 == 1), bChangeNeg = (i>=2);
        if(bChangeX)
        {
            if(bChangeNeg)
            {
              cornerPnts[i].x-=corrLen;
              cornerPnts[iNxt].x-=corrLen;
            }
            else
            {
              cornerPnts[i].x+=corrLen;
              cornerPnts[iNxt].x+=corrLen;
            }
        }
        else
        {
          if(bChangeNeg)
          {
            cornerPnts[i].y-=corrLen;
            cornerPnts[iNxt].y-=corrLen;
          }
          else
          {
            cornerPnts[i].y+=corrLen;
            cornerPnts[iNxt].y+=corrLen;
          }
        }
      }
    }
    if(bAllPntsInside)
      break;
    else
    {
      cornerPnts[0].x-=epsVal;
      cornerPnts[0].y+=epsVal;
      cornerPnts[1].x+=epsVal;
      cornerPnts[1].y+=epsVal;
      cornerPnts[2].x+=epsVal;
      cornerPnts[2].y-=epsVal;
      cornerPnts[3].x-=epsVal;
      cornerPnts[3].y-=epsVal;
    }
  }

  if(bAllPntsInside)
  {
    LT = cornerPnts[1];
    RB = cornerPnts[3];

    int retDiag0 = getPolylineIntersection(pts_rect, cornerPnts[0], cornerPnts[2]);
    int retDiag1 = getPolylineIntersection(pts_rect, cornerPnts[1], cornerPnts[3]);

    if((retDiag0 != -1)
      ||(retDiag1 != -1))
    {
      if(!bTryFix)
        return false;
      //fixe errors where ROI coincides directly with the polyline
      LT.x++;
      LT.y++;
      RB.x--;
      RB.y--;
      return seqDetCorrectCollisions(LT, RB, pts_rect, false);
    }

    return true;
  }
  else
    return false; //invalid result
}

//adapted from http://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function/7448287#7448287
double seqDetCalcIntersectionStrict(const Point2d& o1, const Point2d& p1, const Point2d& o2, const Point2d& p2, Point2d &r)
{
  //const double eps_neg = 0.001;
  Point2d x = o2 - o1;
  Point2d d1 = p1 - o1;
  Point2d d2 = p2 - o2;

  double cross = d1.x*d2.y - d1.y*d2.x;
  if (std::abs(cross) < /*EPS*/1e-8)
    return -FLT_MAX;

  double t1 = (x.x * d2.y - x.y * d2.x)/cross;
  double t2 = (x.x * d1.y - x.y * d1.x)/cross;

  Point2d r2 = o2 + d2 * t2;
  r = o1 + d1 * t1;
  if((t2<0)|| (t2 > 1.0))
    return -FLT_MAX; //return invalid
  return t1;
}

double seqDetMySimpleExtrEuclidianSquared(const cv::Point2d& pt)
{
	return pt.x*pt.x+pt.y*pt.y;
}

double seqDetArea(cv::Point2d &lt, cv::Point2d &rb)
{
  return (rb.x-lt.x)*(rb.y-lt.y);
}
