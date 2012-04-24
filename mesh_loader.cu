#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "mesh_loader.h"

#define BUFFER_SIZE 128

void check_high_low(mesh_t *mesh, vec3_t *loc) {
   if (mesh->vertexCount == 1) {
      mesh->high = *loc;
      mesh->low = *loc;
      return;
   }
   
   mesh->high.x = mesh->high.x < loc->x ? loc->x : mesh->high.x;
   mesh->high.y = mesh->high.y < loc->y ? loc->y : mesh->high.y;
   mesh->high.z = mesh->high.z < loc->z ? loc->z : mesh->high.z;
   
   mesh->low.x = mesh->low.x > loc->x ? loc->x : mesh->low.x;
   mesh->low.y = mesh->low.y > loc->y ? loc->y : mesh->low.y;
   mesh->low.z = mesh->low.z > loc->z ? loc->z : mesh->low.z;
}

void load_m_mesh(mesh_t *mesh, const char *meshFile) {
   FILE *fp;
   char buffer[BUFFER_SIZE];
   int intBuf;
   int verticesSize = BUFFER_SIZE;
   int trianglesSize = BUFFER_SIZE;
   vertex_t blankVertex = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.2, 0.2, 0.2}};
   ivec3_t blankTriangle = {0, 0, 0};
   
   fp = fopen(meshFile, "r");
   if (!fp) {
      printf("error reading mesh file: %s\n", meshFile);
      mesh->triangleCount = 0;
      return;
   }
   
   mesh->vertices = (vertex_t *) malloc(BUFFER_SIZE * sizeof(vertex_t));
   mesh->triangles = (ivec3_t *) malloc(BUFFER_SIZE * 3 * sizeof(int));
   
   while (fgets(buffer, BUFFER_SIZE, fp) != NULL) {
      if (strstr(buffer, "Vertex") != NULL) {
         vertex_t *vertex;
         vec3_t loc;
         
         if (mesh->vertexCount >= verticesSize) {
            verticesSize *= BUFFER_SIZE;
            mesh->vertices = (vertex_t *) realloc(mesh->vertices, verticesSize * sizeof(vertex_t));
         }
         
         vertex = mesh->vertices + mesh->vertexCount++;
         *vertex = blankVertex;
         
         sscanf(buffer, "Vertex %d %f %f %f", &intBuf, &loc.x, &loc.y, &loc.z);
         vertex->location = loc;
         check_high_low(mesh, &loc);
      }
      else if (strstr(buffer, "Face") != NULL) {
         ivec3_t *triangle;
         
         if (mesh->triangleCount >= trianglesSize) {
            trianglesSize *= BUFFER_SIZE;
            mesh->triangles = (ivec3_t *) realloc(mesh->triangles, trianglesSize * 3 * sizeof(int));
         }
         
         triangle = mesh->triangles + mesh->triangleCount++;
         *triangle = blankTriangle;
         
         sscanf(buffer, "Face %d %d %d %d", &intBuf, &triangle->x, &triangle->y, &triangle->z);
         --triangle->x;
         --triangle->y;
         --triangle->z;
      }
   }
   
   fclose(fp);
}