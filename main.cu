//
//  main.c
//  CudaRasterizer
//
//  Created by Robert Crosby on 4/21/12.
//  Copyright (c) 2012 In-Con. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#ifdef __APPLE__
#include <CUDA/CUDA.h>
#endif
#ifdef __unix__
#include <cuda.h>
#endif
#include "structures.h"
#include "mesh_loader.h"
#include "rasterizer.h"
#include "blur_filter.h"
#include "png_loader.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "hrt.h"

#define STR_SIZE 24

#define SETUP_INDEX 0
#define NORMALS_INDEX 1
#define TRANSLATE_INDEX 2
#define LIGHT_INDEX 3
#define BUFFER_INDEX 4
#define RASTERIZE_INDEX 5
#define BLUR_INDEX 6
#define CLEANUP_INDEX 7
#define TOTAL_INDEX 8
#define TIME_COUNT 9

#define SCALE_TO_SCREEN 0.8
#define DEF_SIZE 1000

static const char *DEF_MESH = "monkey_high.m";
static const char *DEF_IMAGE = "test.png";

static char *timeNames[TIME_COUNT] = {"setup", "calculate normals",
   "translations", "light vertices", "create buffers", "rasterize mesh",
   "blur image", "cleanup", "total"};
static uint64_t times[TIME_COUNT];

void printTimes() {
   int ndx;
   char str[STR_SIZE];
   register uint64_t total = 0;

   for (ndx = 0; ndx < TIME_COUNT-1; ++ndx) {
      snprintf(str, STR_SIZE, "%" PRIu64 "ns", times[ndx]);
      printf("%s: %s\n", timeNames[ndx], str);
      total += times[ndx];
   }
   times[TOTAL_INDEX] = total;

   snprintf(str, STR_SIZE, "%" PRIu64 "ns", times[ndx]);
   printf("%s: %s\n", timeNames[ndx], str);
}

int render_mesh(const char *imageFile, const char *meshFile, int width, int height, int duplicates, int blur_iter) {
   mesh_t mesh = {NULL, NULL, 0, NULL, NULL, 0, NULL, 0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
   vec3_t center;
   mat4_t modelMtx;
   float scale;
   vec3_t lightDir = {-1.0, 1.0, 1.0};
   vec3_t lightColor = {0.7, 0.7, 0.7};
   drawbuffer_t buffers;
   bitmap_t bitmap;

   // load the mesh
   load_m_mesh(&mesh, meshFile);
   if (mesh.triangleCount == 0)
      return 1;

   //printf("triangiles: %d\n", mesh.triangleCount);

   times[SETUP_INDEX] = 0;

   hrt_start();

   mesh_set_normals(&mesh);

   hrt_stop();
   times[NORMALS_INDEX] = hrt_result();

   // create the transforms and apply to mesh
   center = vec3_add(&mesh.high, &mesh.low);
   center = vec3_scale(&center, -0.5);
   scale = mesh.high.x - mesh.low.x;
   scale = 1 / scale;

   modelMtx = mat4_translation(center.x, center.y, center.z);
   mat4_scale3f(&modelMtx, scale, -scale, scale);
   mat4_translate3f(&modelMtx, 0.5f, 0.5f, 1.0f);

   hrt_start();

   mesh_translate_locations(&mesh, &modelMtx);

   hrt_stop();
   times[TRANSLATE_INDEX] = hrt_result();

   hrt_start();

   // light the vertices
   mesh_light_directional(&mesh, &lightDir, &lightColor);

   hrt_stop();
   times[LIGHT_INDEX] = hrt_result();

   hrt_start();

   // create the color and z buffers
   buffers.width = width;
   buffers.height = height;
   buffers.colorBuffer = (color_t *) malloc(width * height * sizeof(color_t));
   buffers.zBuffer = (float *) malloc(width * height * sizeof(float));

   hrt_stop();
   times[BUFFER_INDEX] = hrt_result();

   hrt_start();

   // draw the mesh
   rasterize_mesh(&buffers, &mesh, duplicates);

   hrt_stop();
   times[RASTERIZE_INDEX] = hrt_result();

   free(mesh.vertices);
   free(mesh.triangles);
   free(buffers.zBuffer);

   hrt_start();

   // create a bit map
   bitmap.width = buffers.width;
   bitmap.height = buffers.height;
   bitmap.pixels = buffers.colorBuffer;

   // blur the bit map
   blur_bitmap(&bitmap, blur_iter);

   hrt_stop();
   times[BLUR_INDEX] = hrt_result();

   times[CLEANUP_INDEX] = 0;

   // write to file
   save_png_to_file(&bitmap, imageFile);

   free(buffers.colorBuffer);

   return 0;
}

int render_mesh_cuda(const char *imageFile, const char *meshFile, int width, int height, int duplicates, int blur_iter) {
   mesh_t mesh = {NULL, NULL, 0, NULL, NULL, 0, NULL, 0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
   size_t size;
   vec3_t center;
   mat4_t modelMtx;
   float scale;
   //float depth;
   vec3_t lightDir = {-1.0, 1.0, 1.0};
   vec3_t lightColor = {0.7, 0.7, 0.7};
   drawbuffer_t buffers;
   bitmap_t bitmap;

   // load the mesh
   load_m_mesh(&mesh, meshFile);
   if (mesh.triangleCount == 0)
      return 1;

   hrt_start();

   size = mesh.vertexCount * sizeof(vertex_t);
   if (cudaMalloc((void **) &mesh.d_vertices, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for vertices\n");
   cudaMemcpy(mesh.d_vertices, mesh.vertices, size, cudaMemcpyHostToDevice);
   free(mesh.vertices);

   size = mesh.triangleCount * sizeof(ivec3_t);
   if (cudaMalloc((void **) &mesh.d_triangles, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for triangles\n");
   cudaMemcpy(mesh.d_triangles, mesh.triangles, size, cudaMemcpyHostToDevice);
   free(mesh.triangles);

   // allocate the polygons
   mesh.polygonCount = mesh.triangleCount;
   size = mesh.polygonCount * sizeof(polygon_t);
   if (cudaMalloc((void **) &mesh.d_polygons, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for polygons\n");

   hrt_stop();
   times[SETUP_INDEX] = hrt_result();

   hrt_start();

   // set the normals of the vertices
   mesh_set_normals_cuda(&mesh);

   // create the transforms and apply to mesh
   center = vec3_add(&mesh.high, &mesh.low);
   center = vec3_scale(&center, -0.5);
   scale = mesh.high.x - mesh.low.x;
   scale = 1 / scale;

   hrt_stop();
   times[NORMALS_INDEX] = hrt_result();

   hrt_start();

   modelMtx = mat4_translation(center.x, center.y, center.z);
   mat4_scale3f(&modelMtx, scale, -scale, scale);
   mat4_translate3f(&modelMtx, 0.5f, 0.5f, 1.0);
   mesh_translate_locations_cuda(&mesh, &modelMtx);

   hrt_stop();
   times[TRANSLATE_INDEX] = hrt_result();

   hrt_start();

   // light the vertices
   mesh_light_directional_cuda(&mesh, &lightDir, &lightColor);

   hrt_stop();
   times[LIGHT_INDEX] = hrt_result();

   hrt_start();

   // create the buffers
   buffers.width = width;
   buffers.height = height;

   // create a color buffer on the device
   size = width * height * sizeof(int);
   //printf("size: %d\nwidth * height: %d\nsize of color_t: %d\n", size, width * height, sizeof(color_t));
   if (cudaMalloc((void **) &buffers.d_colorBuffer, size) == cudaErrorMemoryAllocation)
      printf("error creating color buffer\n");

   // create a depth buffer on the device
   size = width * height * sizeof(float);
   if (cudaMalloc((void **) &buffers.d_zBuffer, size) == cudaErrorMemoryAllocation)
      printf("error creating depth buffer\n");

   // create a lock buffer on the device
   size = width * height * sizeof(int);
   if (cudaMalloc((void **) &buffers.d_locks, size) == cudaErrorMemoryAllocation)
      printf("error creating lock buffer\n");

   // clear the buffers
   clear_buffers_cuda(&buffers);

   hrt_stop();
   times[BUFFER_INDEX] = hrt_result();

   hrt_start();

   // rasterize the polygons
   rasterize_mesh_cuda(&buffers, &mesh, duplicates);

   hrt_stop();
   times[RASTERIZE_INDEX] = hrt_result();

   hrt_start();

   // copy the color buffer to host
   size = width * height * sizeof(int);
   buffers.colorBuffer = (color_t *) malloc(size);
   cudaMemcpy(buffers.colorBuffer, buffers.d_colorBuffer, size, cudaMemcpyDeviceToHost);

   // free the buffers on the device
   cudaFree(buffers.d_colorBuffer);
   cudaFree(buffers.d_zBuffer);

   // free the polygons, vertices, and triangles on the device
   cudaFree(mesh.d_polygons);
   cudaFree(mesh.d_vertices);
   cudaFree(mesh.d_triangles);

   hrt_stop();
   times[CLEANUP_INDEX] = hrt_result();

   hrt_start();

   // put together the bit map
   bitmap.width = buffers.width;
   bitmap.height = buffers.height;
   bitmap.pixels = buffers.colorBuffer;

   // blur the bit map
   blur_bitmap_cuda(&bitmap, blur_iter);

   hrt_stop();
   times[BLUR_INDEX] = hrt_result();

   // write the bitmap to a file
   save_png_to_file(&bitmap, imageFile);

   // free the host color buffer
   free(buffers.colorBuffer);

   return 0;
}

int main(int argc, const char * argv[])
{
   const char *meshFile = DEF_MESH;
   const char *imageFile = DEF_IMAGE;
   int i, width, height, useCuda = 0, duplicates = 1, blur = 1, profile = 0;

   width = height = DEF_SIZE;

   for (i = 0; i < argc; ++i) {
      if (strstr(argv[i], "-i") != NULL && ++i < argc)
         meshFile = argv[i];
      else if (strstr(argv[i], "-o") != NULL && ++i < argc)
         imageFile = argv[i];
      else if (strstr(argv[i], "-w") != NULL && ++i < argc)
         sscanf(argv[i], "%d", &width);
      else if (strstr(argv[i], "-h") != NULL && ++i < argc)
         sscanf(argv[i], "%d", &height);
      else if (strstr(argv[i], "-cuda") != NULL)
         useCuda = 1;
      else if (strstr(argv[i], "-n") != NULL && ++i < argc)
         sscanf(argv[i], "%d", &duplicates);
      else if (strstr(argv[i], "-blur") != NULL && ++i < argc)
         sscanf(argv[i], "%d", &blur);
      else if (strstr(argv[i], "-t") != NULL)
         profile = 1;
   }

   if (useCuda)
      render_mesh_cuda(imageFile, meshFile, width, height, duplicates, blur);
   else
      render_mesh(imageFile, meshFile, width, height, duplicates, blur);

   if (profile) printTimes();

   return EXIT_SUCCESS;
}

