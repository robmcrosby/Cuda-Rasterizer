
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "structures.h"
#include "device_functions.h"
#include <cuda_runtime.h>

void mesh_set_normals(mesh_t *mesh) {
   int i;
   
   for (i = 0; i < mesh->triangleCount; ++i) {
      vec3_t vector1, vector2, normal;
      ivec3_t *tri = mesh->triangles + i;
      
      vector1 = vec3_subtract(&mesh->vertices[tri->y].location,
                              &mesh->vertices[tri->x].location);
      vector2 = vec3_subtract(&mesh->vertices[tri->z].location,
                              &mesh->vertices[tri->x].location);
      normal = vec3_cross(&vector1, &vector2);
      vec3_normalize(&normal);
      
      mesh->vertices[tri->x].normal = vec3_add(&mesh->vertices[tri->x].normal, &normal);
      mesh->vertices[tri->y].normal = vec3_add(&mesh->vertices[tri->y].normal, &normal);
      mesh->vertices[tri->z].normal = vec3_add(&mesh->vertices[tri->z].normal, &normal);
   }
   
   for (i = 0; i < mesh->vertexCount; ++i) {
      vertex_t *vert = mesh->vertices + i;
      vec3_normalize(&vert->normal);
   }
}


void mesh_translate_locations(mesh_t *mesh, mat4_t *mtx) {
   vertex_t *vertex;
   
   vertex = mesh->vertices;
   for (int i = 0; i < mesh->vertexCount; ++i, ++vertex)
      vertex->location = mat4_translate_point(mtx, &vertex->location);
}


void mesh_light_directional(mesh_t *mesh, vec3_t *lightDir, vec3_t *lightColor) {
   vertex_t *vertex;
   vec3_t diffColor;
   *lightDir = vec3_normalized(lightDir);
   
   vertex = mesh->vertices;
   for (int i = 0; i < mesh->vertexCount; ++i, ++vertex) {
      float df = vec3_dot(&vertex->normal, lightDir);
      df = df < 0 ? 0 : df;
      
      diffColor = vec3_scale(lightColor, df);
      vertex->color = vec3_add(&diffColor, &vertex->color);
   }
}


__device__ void atomicAddf(float* address, float value) {
   float old = value;  
   while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}

__device__ void cuda_vec3_add_atomic(vec3_t *v1, vec3_t *v2) {
   atomicAddf(&v1->x, v2->x);
   atomicAddf(&v1->y, v2->y);
   atomicAddf(&v1->z, v2->z);
}

__device__ vec3_t cuda_vec3_scale(vec3_t *v, float s) {
   vec3_t retV;
   retV.x = v->x * s;
   retV.y = v->y * s;
   retV.z = v->z * s;
   return retV;
}

__device__ vec3_t cuda_vec3_add(vec3_t *v1, vec3_t *v2) {
   vec3_t retV;
   retV.x = v1->x + v2->x;
   retV.y = v1->y + v2->y;
   retV.z = v1->z + v2->z;
   return retV;
}

__device__ vec3_t cuda_vec3_subtract(vec3_t *v1, vec3_t *v2) {
   vec3_t retV;
   retV.x = v1->x - v2->x;
   retV.y = v1->y - v2->y;
   retV.z = v1->z - v2->z;
   return retV;
}

__device__ vec3_t cuda_vec3_cross(vec3_t *v1, vec3_t *v2) {
   vec3_t retV;
   retV.x = v1->y * v2->z - v1->z * v2->y;
   retV.y = v1->z * v2->x - v1->x * v2->z;
   retV.z = v1->x * v2->y - v1->y * v2->x;
   return retV;
}

__device__ float cuda_vec3_length(vec3_t *v) {
   return sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
}

__device__ float cuda_vec3_dot(vec3_t *v1, vec3_t *v2) {
   return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

__device__ void cuda_vec3_normalize(vec3_t *v) {
   float l = cuda_vec3_length(v);
   v->x /= l;
   v->y /= l;
   v->z /= l;
}

__device__ void cuda_mat4_translate_point(mat4_t *m, vec3_t *pt) {
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

__global__ void cuda_add_normals(vertex_t *vertices, ivec3_t *triangles, int triCount) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   vec3_t vector1, vector2, normal;
   ivec3_t *tri;
   
   if (i < triCount) {
      tri = &triangles[i];
      
      vector1 = cuda_vec3_subtract(&vertices[tri->y].location,
                                   &vertices[tri->x].location);
      vector2 = cuda_vec3_subtract(&vertices[tri->z].location,
                                   &vertices[tri->x].location);
      normal = cuda_vec3_cross(&vector1, &vector2);
      cuda_vec3_normalize(&normal);

      cuda_vec3_add_atomic(&vertices[tri->x].normal, &normal);
      cuda_vec3_add_atomic(&vertices[tri->y].normal, &normal);
      cuda_vec3_add_atomic(&vertices[tri->z].normal, &normal);
   }
}

__global__ void cuda_normalize_normals(vertex_t *vertices, int vertCount) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   vertex_t *vert;
   
   if (i < vertCount) {
      vert = vertices + i;
      cuda_vec3_normalize(&vert->normal);
   }
}

__global__ void cuda_transform_vertices(vertex_t *vertices, mat4_t *mtx, int vertCount) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (i < vertCount) {
      cuda_mat4_translate_point(mtx, &vertices[i].location);
   }
}

void mesh_set_normals_cuda(mesh_t *mesh) {
   int block_size = 16;
   int num_blocks = mesh->triangleCount / block_size + (mesh->triangleCount % block_size == 0 ? 0 : 1);
   
   cuda_add_normals <<< num_blocks, block_size >>> (mesh->d_vertices, mesh->d_triangles, mesh->triangleCount);
   
   num_blocks = mesh->vertexCount / block_size + (mesh->vertexCount % block_size == 0 ? 0 : 1);
   cuda_normalize_normals <<< num_blocks, block_size >>> (mesh->d_vertices, mesh->vertexCount);
}

void mesh_translate_locations_cuda(mesh_t *mesh, mat4_t *mtx) {
   mat4_t *d_mtx;
   int block_size = 16;
   int num_blocks = mesh->vertexCount / block_size + (mesh->vertexCount % block_size == 0 ? 0 : 1);
   
   cudaMalloc((void **) &d_mtx, sizeof(mat4_t));
   cudaMemcpy(d_mtx, mtx, sizeof(mat4_t), cudaMemcpyHostToDevice);
   
   cuda_transform_vertices <<< num_blocks, block_size >>> (mesh->d_vertices, d_mtx, mesh->vertexCount);

   cudaFree(d_mtx);
}


__global__ void cuda_light_vertices(vertex_t *vertices, vec3_t *lightDir, vec3_t *lightColor, int vertCount) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   float df;
   vec3_t diffColor;
   
   if (i < vertCount) {
      df = cuda_vec3_dot(&vertices[i].normal, lightDir);
      df = df < 0 ? 0 : df;
      
      diffColor = cuda_vec3_scale(lightColor, df);
      vertices[i].color = cuda_vec3_add(&diffColor, &vertices[i].color);
   }
}

void mesh_light_directional_cuda(mesh_t *mesh, vec3_t *lightDir, vec3_t *lightColor) {
   int block_size = 16;
   int num_blocks = mesh->vertexCount / block_size + (mesh->vertexCount % block_size == 0 ? 0 : 1);
   vec3_t *d_lightDir;
    vec3_t *d_lightColor;
   
   *lightDir = vec3_normalized(lightDir);
   cudaMalloc((void **) &d_lightDir, sizeof(vec3_t));
   cudaMemcpy(d_lightDir, lightDir, sizeof(vec3_t), cudaMemcpyHostToDevice);
   
   cudaMalloc((void **) &d_lightColor, sizeof(vec3_t));
   cudaMemcpy(d_lightColor, lightColor, sizeof(vec3_t), cudaMemcpyHostToDevice);
   
   cuda_light_vertices <<< num_blocks, block_size >>> (mesh->d_vertices, d_lightDir, d_lightColor, mesh->vertexCount);
   
   cudaFree(d_lightDir);
   cudaFree(d_lightColor);
}
