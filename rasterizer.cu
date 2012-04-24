//
//  rasterizer.c
//  rasterizer
//
//  Created by Robert Crosby on 4/15/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "rasterizer.h"

/*
 check_barycentric
 ----------------------------------------------------------------------------
 Checks if the barycentric coordinates are inside.
 
 Takes a vec3_t with barycentric coordinates.
 Returns 1 if none of the coordinates are negative or 0 otherwise.
 */
static int check_barycentric(vec3_t *coords) {
   return coords->x >= 0 && coords->y >= 0 && coords->z >= 0;
}

/*
 get_barycentric
 ----------------------------------------------------------------------------
 Finds the barycentric coordinates of a point on a polygon.
 
 Takes a polygon_t and an x and y intenter for the location on the polygon.
 Returns a vec3_t with the three barycentric coordinates.
 */
static vec3_t get_barycentric(const polygon_t *poly, int x, int y) {
   vec3_t coords;
   
   coords.x = x * poly->loc2.y;
   coords.x -= x * poly->loc3.y;
   coords.x -= poly->loc2.x * y;
   coords.x += poly->loc2.x * poly->loc3.y;
   coords.x += poly->loc3.x * y;
   coords.x -= poly->loc3.x * poly->loc2.y;
   coords.x /= poly->det;
   
   coords.y = poly->loc1.x * y;
   coords.y -= poly->loc1.x * poly->loc3.y;
   coords.y -= x * poly->loc1.y;
   coords.y += x * poly->loc3.y;
   coords.y += poly->loc3.x * poly->loc1.y;
   coords.y -= poly->loc3.x * y;
   coords.y /= poly->det;
   
   coords.z = poly->loc1.x * poly->loc2.y;
   coords.z -= poly->loc1.x * y;
   coords.z -= poly->loc2.x * poly->loc1.y;
   coords.z += poly->loc2.x * y;
   coords.z += x * poly->loc1.y;
   coords.z -= x * poly->loc2.y;
   coords.z /= poly->det;
   
   return coords;
}

/*
 polygon_set_det
 ----------------------------------------------------------------------------
 Sets the determinate for the polygon for use latter with get_barycentric.
 
 Takes a polygon_t
 */
static void polygon_set_det(polygon_t *polygon) {
   float det = polygon->loc1.x * polygon->loc2.y;
   det -= polygon->loc1.x * polygon->loc3.y;
   det -= polygon->loc2.x * polygon->loc1.y;
   det += polygon->loc2.x * polygon->loc3.y;
   det += polygon->loc3.x * polygon->loc1.y;
   det -= polygon->loc3.x * polygon->loc2.y;
   polygon->det = det;
}

/*
 polygon_draw_pixel
 ----------------------------------------------------------------------------
 Draws to a given pixel with the data of a given polygon.
 
 Takes a color_t and float pointer for the pixel, a polygon_t, and an x and y for the pixel location.
 */
static void polygon_draw_pixel(color_t *pixel, float *zbuffer, polygon_t *polygon, int x, int y) {
   vec3_t bary;
   vec3_t colorf;
   float zvalue;
   
   // get and check the bary centric coordinates
   bary = get_barycentric(polygon, x, y);
   if (!check_barycentric(&bary))
      return;
   
   // check the zbuffer
   zvalue = bary.x * polygon->zValues.x + bary.y * polygon->zValues.y + bary.z * polygon->zValues.z;
   if (zvalue < *zbuffer)
      return;
   *zbuffer = zvalue;
   
   colorf.x = bary.x * polygon->color1.x + bary.y * polygon->color2.x + bary.z * polygon->color3.x;
   colorf.y = bary.x * polygon->color1.y + bary.y * polygon->color2.y + bary.z * polygon->color3.y;
   colorf.z = bary.x * polygon->color1.z + bary.y * polygon->color2.z + bary.z * polygon->color3.z;
   
   // draw the pixel
   *pixel = vec3_to_color(&colorf);
}

/*
 polygon_draw
 ----------------------------------------------------------------------------
 Iterates over the area defined in the polygon calling polygon_draw_pixel.
 
 Takes a drawbuffer_t to draw to and a polygon_t.
 */
static void polygon_draw(drawbuffer_t *drawbuffer, polygon_t *polygon) {
   for (int i = polygon->low.x; i < polygon->high.x; ++i) {
      for (int j = polygon->low.y; j < polygon->high.y; ++j) {
         color_t *pixel = drawbuffer_get_color_at(drawbuffer, i, j);
         float *zvalue = drawbuffer_get_zvalue_at(drawbuffer, i, j);
         polygon_draw_pixel(pixel, zvalue, polygon, i, j);
      }
   }
}

/*
 polygon_create
 ----------------------------------------------------------------------------
 Creates a polygon_t from a given triangle_t.
 
 Takes a triangle_t pointer, a vertex_t pointer to an array, and width and height of the drawable area.
 Returns a pointer to a polygon_t or NULL if the polygon is back facing.
 */
static polygon_t* polygon_create(ivec3_t *triangle, vertex_t *vertices, int width, int height) {
   polygon_t *polygon;
   vertex_t *v1, *v2, *v3;
   //vec3_t screenNorm = {0.0, 0.0, 1.0};
   
   /*
   // cull out any back facing faces.
   if (vec3_dot(&triangle->normal, &screenNorm) < 0)
      return NULL;*/
   
   // get the vertices of the triangle.
   v1 = vertices + triangle->x;
   v2 = vertices + triangle->y;
   v3 = vertices + triangle->z;
   
   // create the polygon and add the coordinates from the triangle vertices.
   polygon = (polygon_t *)malloc(sizeof(polygon_t));
   polygon->loc1 = vec3_to_ivec2(&v1->location);
   polygon->loc2 = vec3_to_ivec2(&v2->location);
   polygon->loc3 = vec3_to_ivec2(&v3->location);
   
   // find the high screen bounds.
   polygon->high.x = polygon->loc1.x > polygon->loc2.x ? polygon->loc1.x : polygon->loc2.x;
   polygon->high.x = polygon->high.x > polygon->loc3.x ? polygon->high.x : polygon->loc3.x;
   polygon->high.x = polygon->high.x >= width ? width - 1 : polygon->high.x;
   polygon->high.y = polygon->loc1.y > polygon->loc2.y ? polygon->loc1.y : polygon->loc2.y;
   polygon->high.y = polygon->high.y > polygon->loc3.y ? polygon->high.y : polygon->loc3.y;
   polygon->high.y = polygon->high.y >= height ? height - 1 : polygon->high.y;
   
   // find the low screen bounds.
   polygon->low.x = polygon->loc1.x < polygon->loc2.x ? polygon->loc1.x : polygon->loc2.x;
   polygon->low.x = polygon->low.x < polygon->loc3.x ? polygon->low.x : polygon->loc3.x;
   polygon->low.x = polygon->low.x < 0 ? 0 : polygon->low.x;
   polygon->low.y = polygon->loc1.y < polygon->loc2.y ? polygon->loc1.y : polygon->loc2.y;
   polygon->low.y = polygon->low.y < polygon->loc3.y ? polygon->low.y : polygon->loc3.y;
   polygon->low.y = polygon->low.y < 0 ? 0 : polygon->low.y;
   
   // get the color and z values from the triangle.
   polygon->zValues.x = v1->location.z;
   polygon->color1 = v1->color;
   polygon->zValues.y = v2->location.z;
   polygon->color2 = v2->color;
   polygon->zValues.z = v3->location.z;
   polygon->color3 = v3->color;
   
   polygon_set_det(polygon);
   
   return polygon;
}

/*
 check_barycentric
 ----------------------------------------------------------------------------
 Creates a polygon_t from a given triangle_t.
 
 Takes a triangle_t pointer, a vertex_t pointer to an array, and width and height of the drawable area.
 Returns a pointer to a polygon_t or NULL if the polygon is back facing.
 */
void rasterize_mesh(drawbuffer_t *buffers, mesh_t *mesh) {
   
   for (int i = 0; i < mesh->triangleCount; ++i) {
      ivec3_t *triangle = mesh->triangles + i;
      polygon_t *polygon = polygon_create(triangle, mesh->vertices, buffers->width, buffers->height);
      
      // draw the polygons.
      if (polygon != NULL) {
         polygon_draw(buffers, polygon);
      }
      
      free(polygon);
   }
}






/*
 ----------------------------------------------------------------------------
   CUDA Code
 ----------------------------------------------------------------------------
*/

__global__ void cuda_create_polygons(polygon_t *polygons, vertex_t *vertices, ivec3_t *triangles, int polyCount, int width, int height) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   vertex_t *v1, *v2, *v3;
   ivec3_t *tri;
   polygon_t polygon, *polyRef;
   
   if (i < polyCount) {
      tri = triangles + i;
      v1 = vertices + tri->x;
      v2 = vertices + tri->y;
      v3 = vertices + tri->z;
      polyRef = polygons + i;
      
      // locations
      polygon.loc1.x = v1->location.x;
      polygon.loc1.y = v1->location.y;
      polygon.loc2.x = v2->location.x;
      polygon.loc2.y = v2->location.y;
      polygon.loc3.x = v3->location.x;
      polygon.loc3.y = v3->location.y;
      
      // find the high screen bounds.
      polygon.high.x = polygon.loc1.x > polygon.loc2.x ? polygon.loc1.x : polygon.loc2.x;
      polygon.high.x = polygon.high.x > polygon.loc3.x ? polygon.high.x : polygon.loc3.x;
      polygon.high.x = polygon.high.x >= width ? width - 1 : polygon.high.x;
      polygon.high.y = polygon.loc1.y > polygon.loc2.y ? polygon.loc1.y : polygon.loc2.y;
      polygon.high.y = polygon.high.y > polygon.loc3.y ? polygon.high.y : polygon.loc3.y;
      polygon.high.y = polygon.high.y >= height ? height - 1 : polygon.high.y;
      
      // find the low screen bounds.
      polygon.low.x = polygon.loc1.x < polygon.loc2.x ? polygon.loc1.x : polygon.loc2.x;
      polygon.low.x = polygon.low.x < polygon.loc3.x ? polygon.low.x : polygon.loc3.x;
      polygon.low.x = polygon.low.x < 0 ? 0 : polygon.low.x;
      polygon.low.y = polygon.loc1.y < polygon.loc2.y ? polygon.loc1.y : polygon.loc2.y;
      polygon.low.y = polygon.low.y < polygon.loc3.y ? polygon.low.y : polygon.loc3.y;
      polygon.low.y = polygon.low.y < 0 ? 0 : polygon.low.y;
      
      // get the z values
      polygon.zValues.x = v1->location.z;
      polygon.zValues.y = v2->location.z;
      polygon.zValues.z = v3->location.z;
      
      // get the colors
      polygon.color1 = v1->color;
      polygon.color2 = v2->color;
      polygon.color3 = v3->color;
      
      // pre caluclate the lower part of barycentric.
      polygon.det = polygon.loc1.x * polygon.loc2.y;
      polygon.det -= polygon.loc1.x * polygon.loc3.y;
      polygon.det -= polygon.loc2.x * polygon.loc1.y;
      polygon.det += polygon.loc2.x * polygon.loc3.y;
      polygon.det += polygon.loc3.x * polygon.loc1.y;
      polygon.det -= polygon.loc3.x * polygon.loc2.y;
      
      // save to global memory
      *polyRef = polygon;
   }
}

__global__ void cuda_clear_buffers(color_t *colorBuffer, float *zBuffer, int numPixels) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   color_t *color, clearColor = {0, 0, 0};
   float *zVal, zClear = FLT_MIN;
   
   if (i < numPixels) {
      color = colorBuffer + i;
      *color = clearColor;
      zVal = zBuffer + i;
      *zVal = zClear;
   }
}

__device__ vec3_t cuda_get_barycentric(polygon_t *poly, int x, int y) {
   vec3_t coords;
   
   coords.x = x * poly->loc2.y;
   coords.x -= x * poly->loc3.y;
   coords.x -= poly->loc2.x * y;
   coords.x += poly->loc2.x * poly->loc3.y;
   coords.x += poly->loc3.x * y;
   coords.x -= poly->loc3.x * poly->loc2.y;
   coords.x /= poly->det;
   
   coords.y = poly->loc1.x * y;
   coords.y -= poly->loc1.x * poly->loc3.y;
   coords.y -= x * poly->loc1.y;
   coords.y += x * poly->loc3.y;
   coords.y += poly->loc3.x * poly->loc1.y;
   coords.y -= poly->loc3.x * y;
   coords.y /= poly->det;
   
   coords.z = poly->loc1.x * poly->loc2.y;
   coords.z -= poly->loc1.x * y;
   coords.z -= poly->loc2.x * poly->loc1.y;
   coords.z += poly->loc2.x * y;
   coords.z += x * poly->loc1.y;
   coords.z -= x * poly->loc2.y;
   coords.z /= poly->det;
   
   return coords;
}

__device__ void cuda_attomic_set_color(color_t *pixel, color_t *newColor, float *zBuffVal, float new_zVal) {
   float new_old, ret;
   
   if (new_zVal == 0.0f) {
      new_zVal += 0.0001f;
   }
   
   ret = atomicExch(zBuffVal, 0.0f);
   if (ret != 0.0f && ret < new_zVal) {
      new_old = new_zVal;
      pixel->red = newColor->red;
      pixel->green = newColor->green;
      pixel->blue = newColor->blue;
   }
   else
      new_old = ret;
   
   while (atomicExch(zBuffVal, new_old) != 0.0f) {
      new_old = atomicExch(zBuffVal, 0.0f);
   }
}

__device__ void cuda_draw_polygon_pixel(color_t *pixel, float *zBuffVal, polygon_t *polygon, int x, int y) {
   vec3_t bary;
   float zVal;
   color_t newColor;
   
   bary = cuda_get_barycentric(polygon, x, y);
   if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
      zVal = bary.x * polygon->zValues.x + bary.y * polygon->zValues.y + bary.z * polygon->zValues.z;
      newColor.red = (bary.x * polygon->color1.x + bary.y * polygon->color2.x + bary.z * polygon->color3.x) * 255;
      newColor.green = (bary.x * polygon->color1.y + bary.y * polygon->color2.y + bary.z * polygon->color3.y) * 255;
      newColor.blue = (bary.x * polygon->color1.z + bary.y * polygon->color2.z + bary.z * polygon->color3.z) * 255;
      cuda_attomic_set_color(pixel, &newColor, zBuffVal, zVal);
   }
}

__device__ color_t* cuda_get_color_at(color_t *colorBuffer, int width, int x, int y) {
   return colorBuffer + width*y + x;
}

__device__ float* cuda_get_zvalue_at(float *zBuffer, int width, int x, int y) {
   return zBuffer + width*y + x;
}

__global__ void cuda_draw_polygons(polygon_t *polygons, int polyCount, color_t *colorBuffer, float *zBuffer, int width, int height) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j, k;
   polygon_t *polygon;
   color_t *pixel;
   float *zVal;
   
   if (i < polyCount) {
      polygon = polygons + i;
      
      for (j = polygon->low.x; j < polygon->high.x; ++j) {
         for (k = polygon->low.y; k < polygon->high.y; ++k) {
            pixel = cuda_get_color_at(colorBuffer, width, j, k);
            zVal = cuda_get_zvalue_at(zBuffer, width, j, k);
            cuda_draw_polygon_pixel(pixel, zVal, polygon, j, k);
         }
      }
   }
}

void create_polygons_cuda(mesh_t *mesh, int width, int height) {
   int block_size = 16;
   int num_blocks = mesh->triangleCount / block_size + (mesh->triangleCount % block_size == 0 ? 0 : 1);
   
   cuda_create_polygons <<< num_blocks, block_size >>> (mesh->d_polygons, mesh->d_vertices, mesh->d_triangles, mesh->triangleCount, width, height);
   
   mesh->polygonCount = mesh->triangleCount;
}

void clear_buffers_cuda(drawbuffer_t *buffers) {
   int block_size = 32;
   int num_blocks = (buffers->width * buffers->height) / block_size + ((buffers->width * buffers->height) % block_size == 0 ? 0 : 1);
   
   cuda_clear_buffers <<< num_blocks, block_size >>> (buffers->d_colorBuffer, buffers->d_zBuffer, buffers->width * buffers->height);
}

void rasterize_polygons_cuda(drawbuffer_t *buffers, mesh_t *mesh) {
   int block_size = 16;
   int num_blocks = mesh->polygonCount / block_size + (mesh->polygonCount % block_size == 0 ? 0 : 1);
   
   cuda_draw_polygons <<< num_blocks, block_size >>> (mesh->d_polygons, mesh->polygonCount, buffers->d_colorBuffer,
                                                      buffers->d_zBuffer, buffers->width, buffers->height);
}






