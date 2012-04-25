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
static polygon_t* polygon_create(ivec3_t *triangle, vertex_t *vertices, int width, int height, mat4_t *mtx) {
   polygon_t *polygon;
   vertex_t *v1, *v2, *v3;
   vec3_t loc1, loc2, loc3;
   
   
   // get the vertices of the triangle.
   v1 = vertices + triangle->x;
   v2 = vertices + triangle->y;
   v3 = vertices + triangle->z;
   
   // translate the locations
   loc1 = mat4_translate_point(mtx, &v1->location);
   loc2 = mat4_translate_point(mtx, &v2->location);
   loc3 = mat4_translate_point(mtx, &v3->location);
   
   // create the polygon and add the coordinates from the triangle vertices.
   polygon = (polygon_t *)malloc(sizeof(polygon_t));
   polygon->loc1 = vec3_to_ivec2(&loc1);
   polygon->loc2 = vec3_to_ivec2(&loc2);
   polygon->loc3 = vec3_to_ivec2(&loc3);
   
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
   polygon->zValues.x = loc1.z;
   polygon->color1 = v1->color;
   polygon->zValues.y = loc2.z;
   polygon->color2 = v2->color;
   polygon->zValues.z = loc3.z;
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
void rasterize_mesh(drawbuffer_t *buffers, mesh_t *mesh, int duplicates) {
   mat4_t mtxScale, mtx;
   float scale;
   int subDivs = ceil(sqrtf(duplicates));
   
   scale = buffers->width < buffers->height ? buffers->width : buffers->height;
   scale /= subDivs;
   mtxScale = mat4_scaled(scale, scale, 1.0f);
   
   for (int i = 0; i < duplicates; ++i) {
      mtx = mtxScale;
      mat4_translate3f(&mtx, scale * (i/subDivs), scale * (i%subDivs), 0.0f);
      
      for (int j = 0; j < mesh->triangleCount; ++j) {
         ivec3_t *triangle = mesh->triangles + j;
         polygon_t *polygon = polygon_create(triangle, mesh->vertices, buffers->width, buffers->height, &mtx);
         
         // draw the polygons.
         if (polygon != NULL) {
            polygon_draw(buffers, polygon);
         }
         free(polygon);
      }
   }
}






/*
 ----------------------------------------------------------------------------
   CUDA Code
 ----------------------------------------------------------------------------
*/

__device__ void cuda_mat4_translate_point_r(mat4_t *m, vec3_t *pt) {
   vec3_t newpt;
   float w;
   newpt.x = pt->x * m->x.x + pt->y * m->y.x + pt->z * m->z.x + m->w.x;
   newpt.y = pt->x * m->x.y + pt->y * m->y.y + pt->z * m->z.y + m->w.y;
   newpt.z = pt->x * m->x.z + pt->y * m->y.z + pt->z * m->z.z + m->w.z;
   w = pt->x * m->x.w + pt->y * m->y.w + pt->z * m->z.w + m->w.w;
   
   pt->x = newpt.x / w;
   pt->y = newpt.y / w;
   pt->z = newpt.z / w;
}

__global__ void cuda_create_polygons(polygon_t *polygons, vertex_t *vertices, ivec3_t *triangles, int polyCount, int width, int height, mat4_t *mtx) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   vertex_t *v1, *v2, *v3;
   ivec3_t *tri;
   vec3_t loc1, loc2, loc3;
   polygon_t polygon, *polyRef;
   
   if (i < polyCount) {
      tri = triangles + i;
      v1 = vertices + tri->x;
      v2 = vertices + tri->y;
      v3 = vertices + tri->z;
      polyRef = polygons + i;
      
      loc1 = v1->location;
      loc2 = v2->location;
      loc3 = v3->location;
      
      cuda_mat4_translate_point_r(mtx, &loc1);
      cuda_mat4_translate_point_r(mtx, &loc2);
      cuda_mat4_translate_point_r(mtx, &loc3);
      
      // locations
      polygon.loc1.x = loc1.x;
      polygon.loc1.y = loc1.y;
      polygon.loc2.x = loc2.x;
      polygon.loc2.y = loc2.y;
      polygon.loc3.x = loc3.x;
      polygon.loc3.y = loc3.y;
      
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
      polygon.zValues.x = loc1.z;
      polygon.zValues.y = loc2.z;
      polygon.zValues.z = loc3.z;
      
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

__global__ void cuda_clear_buffers(color_t *colorBuffer, float *zBuffer, int *locks, int numPixels) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   color_t *color, clearColor = {0, 0, 0};
   float *zVal, zClear = FLT_MIN;
   int *lock;
   
   if (i < numPixels) {
      color = colorBuffer + i;
      *color = clearColor;
      zVal = zBuffer + i;
      *zVal = zClear;
      lock = locks + i;
      *lock = 0;
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

__device__ void cuda_attomic_set_color(color_t *pixel, color_t *newColor, float *zBuffVal, float new_zVal, int *lock) {
   int lockVal = 0;
   
   while (lockVal = atomicExch(lock, lockVal));
   
   if (*zBuffVal < new_zVal) {
      *zBuffVal = new_zVal;
      pixel->red = newColor->red;
      pixel->green = newColor->green;
      pixel->blue = newColor->blue;
   }
   
   atomicExch(lock, lockVal);
}

__device__ void cuda_draw_polygon_pixel(color_t *pixel, float *zBuffVal, int *lock, polygon_t *polygon, int x, int y) {
   vec3_t bary;
   float zVal;
   color_t newColor;
   
   bary = cuda_get_barycentric(polygon, x, y);
   if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
      zVal = bary.x * polygon->zValues.x + bary.y * polygon->zValues.y + bary.z * polygon->zValues.z;
      newColor.red = (bary.x * polygon->color1.x + bary.y * polygon->color2.x + bary.z * polygon->color3.x) * 255;
      newColor.green = (bary.x * polygon->color1.y + bary.y * polygon->color2.y + bary.z * polygon->color3.y) * 255;
      newColor.blue = (bary.x * polygon->color1.z + bary.y * polygon->color2.z + bary.z * polygon->color3.z) * 255;
      cuda_attomic_set_color(pixel, &newColor, zBuffVal, zVal, lock);
   }
}

__device__ color_t* cuda_get_color_at(color_t *colorBuffer, int width, int x, int y) {
   return colorBuffer + width*y + x;
}

__device__ float* cuda_get_zvalue_at(float *zBuffer, int width, int x, int y) {
   return zBuffer + width*y + x;
}

__device__ int* cuda_get_lock_at(int *locks, int width, int x, int y) {
   return locks + width*y + x;
}

__global__ void cuda_draw_polygons(polygon_t *polygons, int polyCount, color_t *colorBuffer, float *zBuffer, int *locks, int width, int height) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j, k;
   polygon_t *polygon;
   color_t *pixel;
   float *zVal;
   int *lock;
   
   if (i < polyCount) {
      polygon = polygons + i;
      
      for (j = polygon->low.x; j < polygon->high.x; ++j) {
         for (k = polygon->low.y; k < polygon->high.y; ++k) {
            pixel = cuda_get_color_at(colorBuffer, width, j, k);
            zVal = cuda_get_zvalue_at(zBuffer, width, j, k);
            lock = cuda_get_lock_at(locks, width, j, k);
            cuda_draw_polygon_pixel(pixel, zVal, lock, polygon, j, k);
         }
      }
   }
}

void create_polygons_cuda(mesh_t *mesh, int width, int height, mat4_t *mtx) {
   int block_size = 16;
   int num_blocks = mesh->triangleCount / block_size + (mesh->triangleCount % block_size == 0 ? 0 : 1);
   mat4_t *d_mtx;
   
   cudaMalloc((void **) &d_mtx, sizeof(mat4_t));
   cudaMemcpy(d_mtx, mtx, sizeof(mat4_t), cudaMemcpyHostToDevice);
   
   cuda_create_polygons <<< num_blocks, block_size >>> (mesh->d_polygons, mesh->d_vertices, mesh->d_triangles, mesh->polygonCount, width, height, d_mtx);
   
   cudaFree(d_mtx);
}

void clear_buffers_cuda(drawbuffer_t *buffers) {
   int block_size = 64;
   int num_blocks = (buffers->width * buffers->height) / block_size + ((buffers->width * buffers->height) % block_size == 0 ? 0 : 1);
   
   cuda_clear_buffers <<< num_blocks, block_size >>> (buffers->d_colorBuffer, buffers->d_zBuffer, buffers->d_locks, buffers->width * buffers->height);
}

void rasterize_mesh_cuda(drawbuffer_t *buffers, mesh_t *mesh, int duplicates) {
   mat4_t mtxScale, mtx;
   float scale;
   int subDivs = ceil(sqrtf(duplicates));
   int block_size = 16;
   int num_blocks = mesh->polygonCount / block_size + (mesh->polygonCount % block_size == 0 ? 0 : 1);
   
   scale = buffers->width < buffers->height ? buffers->width : buffers->height;
   scale /= subDivs;
   mtxScale = mat4_scaled(scale, scale, 1.0f);
   
   for (int i = 0; i < duplicates; ++i) {
      mtx = mtxScale;
      mat4_translate3f(&mtx, scale * (i/subDivs), scale * (i%subDivs), 0.0f);
      
      create_polygons_cuda(mesh, buffers->width, buffers->height, &mtx);
      
      cuda_draw_polygons <<< num_blocks, block_size >>> (mesh->d_polygons, mesh->polygonCount, buffers->d_colorBuffer,
                                                         buffers->d_zBuffer, buffers->d_locks, buffers->width, buffers->height);
   }
}






