#include <stdio.h>
#include <string.h>
#include "blur_filter.h"

#define BLUR_BLOCK_SIZE 16

vec3_t* hdrbuffer_get_color_at(hdrbuffer_t *buffer, int x, int y) {
   return buffer->pixels + buffer->width * y + x;
}

void blur_9X9_horizontal(hdrbuffer_t *dest, hdrbuffer_t *source) {
   int x, y, i;
   int offset[] = {0, 1, 2, 3, 4};
   float weight[] = {0.273438f, 0.21875f, 0.109375f, 0.03125f, 0.003906f};
   
   for (x = 0; x < source->width; ++x) {
      for (y = 0; y < source->height; ++y) {
         vec3_t source_color;
         vec3_t *dest_color;
         
         dest_color = hdrbuffer_get_color_at(dest, x, y);
         
         source_color = vec3_scale(hdrbuffer_get_color_at(source, x, y), weight[0]);
         *dest_color = source_color;
         
         for (i = 1; i < 5; ++i) {
            if (x+offset[i] >= source->width)
               source_color = vec3_scale(hdrbuffer_get_color_at(source, source->width-1, y), weight[i]);
            else
               source_color = vec3_scale(hdrbuffer_get_color_at(source, x+offset[i], y), weight[i]);
            *dest_color = vec3_add(dest_color, &source_color);
            
            if (x-offset[i] < 0)
               source_color = vec3_scale(hdrbuffer_get_color_at(source, 0, y), weight[i]);
            else
               source_color = vec3_scale(hdrbuffer_get_color_at(source, x-offset[i], y), weight[i]);
            *dest_color = vec3_add(dest_color, &source_color);
         }
      }
   }
}

void blur_9X9_vertical(hdrbuffer_t *dest, hdrbuffer_t *source) {
   int x, y, i;
   int offset[] = {0, 1, 2, 3, 4};
   float weight[] = {0.273438f, 0.21875f, 0.109375f, 0.03125f, 0.003906f};
   
   for (x = 0; x < source->width; ++x) {
      for (y = 0; y < source->height; ++y) {
         vec3_t source_color;
         vec3_t *dest_color;
         
         dest_color = hdrbuffer_get_color_at(dest, x, y);
         
         source_color = vec3_scale(hdrbuffer_get_color_at(source, x, y), weight[0]);
         *dest_color = source_color;
         
         for (i = 1; i < 5; ++i) {
            if (y+offset[i] >= source->height)
               source_color = vec3_scale(hdrbuffer_get_color_at(source, x, source->height-1), weight[i]);
            else
               source_color = vec3_scale(hdrbuffer_get_color_at(source, x, y+offset[i]), weight[i]);
            *dest_color = vec3_add(dest_color, &source_color);
            
            if (y-offset[i] < 0)
               source_color = vec3_scale(hdrbuffer_get_color_at(source, x, 0), weight[i]);
            else
               source_color = vec3_scale(hdrbuffer_get_color_at(source, x, y-offset[i]), weight[i]);
            *dest_color = vec3_add(dest_color, &source_color);
         }
      }
   }
}


int blur_bitmap(bitmap_t *bitmap, int itertions) {
   hdrbuffer_t buffer1, buffer2;
   color_t *color;
   vec3_t *pixel;
   int i;
   
   // create the hdr buffers.
   buffer1.pixels = (vec3_t *) malloc(bitmap->width * bitmap->height * sizeof(vec3_t));
   buffer2.pixels = (vec3_t *) malloc(bitmap->width * bitmap->height * sizeof(vec3_t));
   buffer1.width = buffer2.width = bitmap->width;
   buffer1.height = buffer2.height = bitmap->height;
   for (i = 0, color= bitmap->pixels, pixel = buffer1.pixels; i < bitmap->width * bitmap->height; ++i, ++color, ++pixel)
      *pixel = color_to_vec3(color);
   
   // apply the blur
   for (i = 0; i < itertions; ++i) {
      blur_9X9_horizontal(&buffer2, &buffer1);
      blur_9X9_vertical(&buffer1, &buffer2);
   }
   
   // put the colors back in the bitmap
   for (i = 0, color= bitmap->pixels, pixel = buffer1.pixels; i < bitmap->width * bitmap->height; ++i, ++color, ++pixel)
      *color = vec3_to_color(pixel);
   
   // free the buffers
   free(buffer1.pixels);
   free(buffer2.pixels);
   
   return 0;
}

__device__ vec3_t cuda_vec3_scale2(vec3_t *v, float s) {
   vec3_t retV;
   retV.x = v->x * s;
   retV.y = v->y * s;
   retV.z = v->z * s;
   return retV;
}

__device__ void cuda_vec3_add2(vec3_t *v1, vec3_t *v2) {
   v1->x += v2->x;
   v1->y += v2->y;
   v1->z += v2->z;
}

__device__ vec3_t* cuda_vec3_at(vec3_t *buffer, int x, int y, int width) {
   return buffer + width * y + x;
}

__constant__ int offset[] = {0, 1, 2, 3, 4};
__constant__ float weight[] = {0.273438f, 0.21875f, 0.109375f, 0.03125f, 0.003906f};

__global__ void cuda_blur_9X9_horizontal(vec3_t *buffer1, vec3_t *buffer2, int width, int height) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int i;
   vec3_t sum, value;
   
   if (x < width && y < height) {
      sum = cuda_vec3_scale2(cuda_vec3_at(buffer2, x, y, width), weight[0]);
      
      for (i = 1; i < 5; ++i) {
         if (x+offset[i] >= width)
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, width-1, y, width), weight[i]);
         else
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, x+offset[i], y, width), weight[i]);
         cuda_vec3_add2(&sum, &value);
         
         if (x-offset[i] < 0)
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, 0, y, width), weight[i]);
         else
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, x-offset[i], y, width), weight[i]);
         cuda_vec3_add2(&sum, &value);
      }
      
      buffer1[x + y*width] = sum;
   }
}

__global__ void cuda_blur_9X9_vertical(vec3_t *buffer1, vec3_t *buffer2, int width, int height) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int i;
   vec3_t sum, value;
   
   if (x < width && y < height) {
      sum = cuda_vec3_scale2(cuda_vec3_at(buffer2, x, y, width), weight[0]);
      
      for (i = 1; i < 5; ++i) {
         if (y+offset[i] >= height)
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, x, height-1, width), weight[i]);
         else
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, x, y+offset[i], width), weight[i]);
         cuda_vec3_add2(&sum, &value);
         
         if (y-offset[i] < 0)
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, x, 0, width), weight[i]);
         else
            value = cuda_vec3_scale2(cuda_vec3_at(buffer2, x, y-offset[i], width), weight[i]);
         cuda_vec3_add2(&sum, &value);
      }
      
      buffer1[x + y*width] = sum;
   }
}

void blur_9X9_horizontal_cuda(vec3_t *buffer1, vec3_t *buffer2, int width, int height) {
   dim3 block_size;
   dim3 num_blocks;
   
   block_size.x = block_size.y = BLUR_BLOCK_SIZE;
   num_blocks.x = width / block_size.x + (width % block_size.x == 0 ? 0 : 1);
   num_blocks.y = height / block_size.y + (height % block_size.y == 0 ? 0 : 1);
   
   cuda_blur_9X9_horizontal <<< num_blocks, block_size >>> (buffer1, buffer2, width, height);
}

void blur_9X9_vertical_cuda(vec3_t *buffer1, vec3_t *buffer2, int width, int height) {
   dim3 block_size;
   dim3 num_blocks;
   
   block_size.x = block_size.y = 16;
   num_blocks.x = width / block_size.x + (width % block_size.x == 0 ? 0 : 1);
   num_blocks.y = height / block_size.y + (height % block_size.y == 0 ? 0 : 1);
   
   cuda_blur_9X9_vertical <<< num_blocks, block_size >>> (buffer1, buffer2, width, height);
}

int blur_bitmap_cuda(bitmap_t *bitmap, int itertions) {
   vec3_t *buffer_h, *buffer1_d, *buffer2_d;
   color_t *color;
   vec3_t *pixel;
   size_t size;
   int i;
   
   // create the hdr buffers.
   size = bitmap->width * bitmap->height * sizeof(vec3_t);
   buffer_h = (vec3_t *) malloc(size);
   if (cudaMalloc((void **) &buffer1_d, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for blur buffer1\n");
   if (cudaMalloc((void **) &buffer2_d, size) == cudaErrorMemoryAllocation)
      printf("error creating memory for blur buffer2\n");
   
   // convert colors to vec3_t and move to device
   for (i = 0, color= bitmap->pixels, pixel = buffer_h; i < bitmap->width * bitmap->height; ++i, ++color, ++pixel)
      *pixel = color_to_vec3(color);
   cudaMemcpy(buffer1_d, buffer_h, size, cudaMemcpyHostToDevice);
   
   // blur the image
   for (i = 0; i < itertions; ++i) {
      blur_9X9_horizontal_cuda(buffer2_d, buffer1_d, bitmap->width, bitmap->height);
      blur_9X9_vertical_cuda(buffer1_d, buffer2_d, bitmap->width, bitmap->height);
   }
   
   // put the colors back in the bitmap
   cudaMemcpy(buffer_h, buffer1_d, size, cudaMemcpyDeviceToHost);
   for (i = 0, color= bitmap->pixels, pixel = buffer_h; i < bitmap->width * bitmap->height; ++i, ++color, ++pixel)
      *color = vec3_to_color(pixel);
   
   // free the buffers.
   free(buffer_h);
   cudaFree(buffer1_d);
   cudaFree(buffer2_d);
   
   return 0;
}