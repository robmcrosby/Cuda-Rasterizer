//
//  main.c
//  CudaRasterizer
//
//  Created by Robert Crosby on 4/21/12.
//  Copyright (c) 2012 In-Con. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <CUDA/CUDA.h>
#include "structures.h"
#include "mesh_loader.h"
#include "rasterizer.h"
#include "png_loader.h"

#define SCALE_TO_SCREEN 0.8
#define DEF_SIZE 1000

static const char *DEF_MESH = "dragon10k.m";
static const char *DEF_IMAGE = "test.png";

int render_mesh(const char *imageFile, const char *meshFile, int width, int height) {
   mesh_t mesh = {NULL, NULL, 0, NULL, NULL, 0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
   vec3_t center;
   mat4_t modelMtx;
   float scale;
   float depth;
   vec3_t lightDir = {-1.0, 1.0, 1.0};
   vec3_t lightColor = {0.7, 0.7, 0.7};
   drawbuffer_t buffers;
   bitmap_t bitmap;
   
   // load the mesh
   load_m_mesh(&mesh, meshFile);
   if (mesh.triangleCount == 0)
      return 1;
   
   mesh_set_normals(&mesh);
   
   // create the transforms and apply to mesh
   center = vec3_add(&mesh.high, &mesh.low);
   center = vec3_scale(&center, -0.5);
   scale = mesh.high.x - mesh.low.x;
   scale = width / scale;
   scale *= SCALE_TO_SCREEN;
   depth = scale;
   
   modelMtx = mat4_translation(center.x, center.y, center.z);
   mat4_scale3f(&modelMtx, scale, -scale, scale);
   mat4_translate3f(&modelMtx, width/2.0, height/2.0, depth);
   mesh_translate_locations(&mesh, &modelMtx);
   
   // light the vertices
   mesh_light_directional(&mesh, &lightDir, &lightColor);
   
   // create the color and z buffers
   buffers = drawbuffer_create(width, height);
   
   // draw the mesh
   rasterize_mesh(&buffers, &mesh);
   
   //printf("first pos: (%f, %f, %f)\n", mesh.vertices->color.x, mesh.vertices->color.y, mesh.vertices->color.z);
   
   free(mesh.vertices);
   free(mesh.triangles);
   
   // write to file
   bitmap.width = buffers.width;
   bitmap.height = buffers.height;
   bitmap.pixels = buffers.colorBuffer;
   save_png_to_file(&bitmap, imageFile);
   
   free(buffers.colorBuffer);
   free(buffers.zBuffer);
   
   return 0;
}

int render_mesh_cuda(const char *imageFile, const char *meshFile, int width, int height) {
   mesh_t mesh = {NULL, NULL, 0, NULL, NULL, 0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
   size_t size;
   vec3_t center;
   mat4_t modelMtx;
   float scale;
   float depth;
   vec3_t lightDir = {-1.0, 1.0, 1.0};
   vec3_t lightColor = {0.7, 0.7, 0.7};
   float num;
   
   // load the mesh
   load_m_mesh(&mesh, meshFile);
   if (mesh.triangleCount == 0)
      return 1;
   
   size = mesh.vertexCount * sizeof(vertex_t);
   if (cudaMalloc((void **) &mesh.d_vertices, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for vertices\n");
   cudaMemcpy(mesh.d_vertices, mesh.vertices, size, cudaMemcpyHostToDevice);
   
   size = mesh.triangleCount * sizeof(ivec3_t);
   if (cudaMalloc((void **) &mesh.d_triangles, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for triangles\n");
   cudaMemcpy(mesh.d_triangles, mesh.triangles, size, cudaMemcpyHostToDevice);
    
   mesh_set_normals_cuda(&mesh);
   
   // create the transforms and apply to mesh
   center = vec3_add(&mesh.high, &mesh.low);
   center = vec3_scale(&center, -0.5);
   scale = mesh.high.x - mesh.low.x;
   scale = width / scale;
   scale *= SCALE_TO_SCREEN;
   depth = scale;
   
   modelMtx = mat4_translation(center.x, center.y, center.z);
   mat4_scale3f(&modelMtx, scale, -scale, scale);
   mat4_translate3f(&modelMtx, width/2.0, height/2.0, depth);
   mesh_translate_locations_cuda(&mesh, &modelMtx);
   
   mesh_light_directional_cuda(&mesh, &lightDir, &lightColor);
   
   size = mesh.vertexCount * sizeof(vertex_t);
   cudaMemcpy(mesh.vertices, mesh.d_vertices, size, cudaMemcpyDeviceToHost);
   
   printf("first pos: (%f, %f, %f)\n", mesh.vertices->color.x, mesh.vertices->color.y, mesh.vertices->color.z);
   
   free(mesh.vertices);
   cudaFree(mesh.d_vertices);
   
   free(mesh.triangles);
   cudaFree(mesh.d_triangles);
   
   return 0;
}

int main(int argc, const char * argv[])
{
   const char *meshFile = DEF_MESH;
   const char *imageFile = DEF_IMAGE;
   int i, width, height, useCuda = 0;
   
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
   }
   
   if (useCuda) {
      return render_mesh_cuda(imageFile, meshFile, width, height);
   }
   return render_mesh(imageFile, meshFile, width, height);
}

