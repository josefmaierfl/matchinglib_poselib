// based upon Generic Polygon Clipper by Alan Murta gpc@cs.man.ac.uk Advanced Interfaces Group, University of Manchester.
// reimplementation by O.Zendel, Austrian Institute of Technology AIT

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

void add_contour_from_array(double * pStart, int numVerts, polygon_contour *pRet);
void free_polygon(polygon_contour *p);
void calc_polygon_clipping(polygon_op op, polygon_contour *poly_a, polygon_contour *poly_b, polygon_contour *result);

#ifdef __cplusplus
}
#endif

#endif // SEQUENCE_DETECTION_POLYGON_HELPER