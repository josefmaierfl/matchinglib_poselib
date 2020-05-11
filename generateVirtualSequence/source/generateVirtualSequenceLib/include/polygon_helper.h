/*
 based upon Generic Polygon Clipper by Alan Murta gpc@cs.man.ac.uk Advanced Interfaces Group, University of Manchester.
 modified by O.Zendel, Austrian Institute of Technology AIT
 Original Copyright applies :
===========================================================================
Project:   Generic Polygon Clipper
           A new algorithm for calculating the difference, intersection,
           exclusive-or or union of arbitrary polygon sets.
File:      gpc.c
Author:    Alan Murta (email: gpc@cs.man.ac.uk)
Version:   2.32
Date:      17th December 2004
Copyright: (C) Advanced Interfaces Group,
           University of Manchester.
           This software is free for non-commercial use. It may be copied,
           modified, and redistributed provided that this copyright notice
           is preserved on all copies. The intellectual property rights of
           the algorithms used reside with the University of Manchester
           Advanced Interfaces Group.
           You may not use this software, in whole or in part, in support
           of any commercial product without the express consent of the
           author.
           There is no warranty or other guarantee of fitness of this
           software for any purpose. It is provided solely "as is".
===========================================================================
*/


#ifndef SEQUENCE_DETECTION_POLYGON_HELPER
#define SEQUENCE_DETECTION_POLYGON_HELPER

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

// polygon operation
typedef enum
{
  POLYGON_OP_DIFFERENCE,
  POLYGON_OP_INTERSECT,
  POLYGON_OP_XOR, 
  POLYGON_OP_UNION
} polygon_op;

// single vertex
typedef struct
{
  double              x;
  double              y;
} polygon_vertex;

//vertex list
typedef struct
{
  int                 num_vertices; 
  polygon_vertex      *vertex;
} polygon_vertex_list;

//full contour
typedef struct
{
  int                     num_contours; 
  int                     *hole;
  polygon_vertex_list     *contour;
} polygon_contour;

void add_contour_from_array(const double * pStart, int numVerts, polygon_contour *pRet);
void free_polygon(polygon_contour *p);
void calc_polygon_clipping(polygon_op op, polygon_contour *poly_a, polygon_contour *poly_b, polygon_contour *result);

#ifdef __cplusplus
}
#endif

#endif // SEQUENCE_DETECTION_POLYGON_HELPER