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

#define SCALE_TO_SCREEN 0.8
#define DEF_SIZE 1000

static const char *DEF_MESH = "monkey_high.m";
static const char *DEF_IMAGE = "test.png";

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
   
   mesh_set_normals(&mesh);
   
   // create the transforms and apply to mesh
   center = vec3_add(&mesh.high, &mesh.low);
   center = vec3_scale(&center, -0.5);
   scale = mesh.high.x - mesh.low.x;
   scale = 1 / scale;
   //scale = width / scale;
   //scale *= SCALE_TO_SCREEN;
   //depth = scale;
   
   modelMtx = mat4_translation(center.x, center.y, center.z);
   mat4_scale3f(&modelMtx, scale, -scale, scale);
   mat4_translate3f(&modelMtx, 0.5f, 0.5f, 1.0f);
   //mat4_translate3f(&modelMtx, width/2.0, height/2.0, depth);
   
   mesh_translate_locations(&mesh, &modelMtx);
   
   // light the vertices
   mesh_light_directional(&mesh, &lightDir, &lightColor);
   
   // create the color and z buffers
   buffers.width = width;
   buffers.height = height;
   buffers.colorBuffer = (color_t *) malloc(width * height * sizeof(color_t));
   buffers.zBuffer = (float *) malloc(width * height * sizeof(float));
   
   // draw the mesh
   rasterize_mesh(&buffers, &mesh, duplicates);
   
   free(mesh.vertices);
   free(mesh.triangles);
   free(buffers.zBuffer);
   
   // create a bit map
   bitmap.width = buffers.width;
   bitmap.height = buffers.height;
   bitmap.pixels = buffers.colorBuffer;
   
   // blur the bit map
   blur_bitmap(&bitmap, blur_iter);
   
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
   
   // set the normals of the vertices
   mesh_set_normals_cuda(&mesh);
   
   // create the transforms and apply to mesh
   center = vec3_add(&mesh.high, &mesh.low);
   center = vec3_scale(&center, -0.5);
   scale = mesh.high.x - mesh.low.x;
   scale = 1 / scale;
   //scale *= SCALE_TO_SCREEN;
   //depth = scale;
   
   modelMtx = mat4_translation(center.x, center.y, center.z);
   mat4_scale3f(&modelMtx, scale, -scale, scale);
   mat4_translate3f(&modelMtx, 0.5f, 0.5f, 1.0);
   mesh_translate_locations_cuda(&mesh, &modelMtx);
   
   // light the vertices
   mesh_light_directional_cuda(&mesh, &lightDir, &lightColor);
   
   // allocate the polygons
   mesh.polygonCount = mesh.triangleCount;
   size = mesh.polygonCount * sizeof(polygon_t);
   if (cudaMalloc((void **) &mesh.d_polygons, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for polygons\n");
   
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
   
   // rasterize the polygons
   rasterize_mesh_cuda(&buffers, &mesh, duplicates);
   
   // copy the color buffer to host
   size = width * height * sizeof(int);
   buffers.colorBuffer = (color_t *) malloc(size);
   cudaMemcpy(buffers.colorBuffer, buffers.d_colorBuffer, size, cudaMemcpyDeviceToHost);
   
   // write to file
   bitmap.width = buffers.width;
   bitmap.height = buffers.height;
   bitmap.pixels = buffers.colorBuffer;
   save_png_to_file(&bitmap, imageFile);
   
   // free the host color buffer
   free(buffers.colorBuffer);
   
   // free the color buffer on the device
   cudaFree(buffers.d_colorBuffer);
   
   // free the polygons and z buffer
   cudaFree(mesh.d_polygons);
   cudaFree(buffers.d_zBuffer);
   
   // free the vertices and triangles
   cudaFree(mesh.d_vertices);
   cudaFree(mesh.d_triangles);
   
   return 0;
}

int main(int argc, const char * argv[])
{
   const char *meshFile = DEF_MESH;
   const char *imageFile = DEF_IMAGE;
   int i, width, height, useCuda = 0, duplicates = 1, blur = 1;
   
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
   }
   
   if (useCuda) {
      return render_mesh_cuda(imageFile, meshFile, width, height, duplicates, blur);
   }
   return render_mesh(imageFile, meshFile, width, height, duplicates, blur);
}

