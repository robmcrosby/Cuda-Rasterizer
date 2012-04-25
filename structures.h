//
//  structures.h
//  CudaRasterizer
//
//  Created by Robert Crosby on 4/21/12.
//  Copyright (c) 2012 In-Con. All rights reserved.
//

#ifndef CudaRasterizer_structures_h
#define CudaRasterizer_structures_h

#include <stdint.h>
#include "linear_math.h"

#define VERTEX_SIZE = 9
#define NORMAL_OFFSET = 3
#define COLOR_OFFSET = 6

typedef struct color_st {
   uint8_t red;
   uint8_t green;
   uint8_t blue;
} color_t;

typedef struct bitmap_st {
   color_t *pixels;
   size_t width;
   size_t height;
} bitmap_t;


typedef struct drawbuffer_st {
   color_t *colorBuffer;
   color_t *d_colorBuffer;
   float *zBuffer;
   float *d_zBuffer;
   int *d_locks;
   int width;
   int height;
} drawbuffer_t;

typedef struct vertex_st {
   vec3_t location;
   vec3_t normal;
   vec3_t color;
} vertex_t;

/*
 struct for a polygon.
 */
typedef struct {
   ivec2_t loc1;
   ivec2_t loc2;
   ivec2_t loc3;
   vec3_t zValues;
   vec3_t color1;
   vec3_t color2;
   vec3_t color3;
   float det;
   ivec2_t high;
   ivec2_t low;
} polygon_t;

typedef struct mesh_st {
   vertex_t *vertices;
   vertex_t *d_vertices;
   int vertexCount;
   ivec3_t *triangles;
   ivec3_t *d_triangles;
   int triangleCount;
   polygon_t *d_polygons;
   int polygonCount;
   vec3_t high;
   vec3_t low;
} mesh_t;

color_t* bitmap_pixel_at(bitmap_t * bitmap, size_t x, size_t y);
color_t vec3_to_color(const vec3_t *vec);

color_t* drawbuffer_get_color_at(drawbuffer_t *buffer, int x, int y);
float* drawbuffer_get_zvalue_at(drawbuffer_t *buffer, int x, int y);

void mesh_set_normals(mesh_t *mesh);
void mesh_set_normals_cuda(mesh_t *mesh);

void mesh_translate_locations(mesh_t *mesh, mat4_t *mtx);
void mesh_translate_locations_cuda(mesh_t *mesh, mat4_t *mtx);

void mesh_light_directional(mesh_t *mesh, vec3_t *lightDir, vec3_t *lightColor);
void mesh_light_directional_cuda(mesh_t *mesh, vec3_t *lightDir, vec3_t *lightColor);

#endif
