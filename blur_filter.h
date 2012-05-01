//
//  blur_filter.h
//  CudaRasterizer
//
//  Created by Robert Crosby on 4/27/12.
//  Copyright (c) 2012 In-Con. All rights reserved.
//

#ifndef CudaRasterizer_blur_filter_h
#define CudaRasterizer_blur_filter_h

#include "structures.h"

typedef struct {
   vec3_t *pixels;
   size_t width;
   size_t height;
} hdrbuffer_t;

int blur_bitmap(bitmap_t *bitmap, int itertions);
int blur_bitmap_cuda(bitmap_t *bitmap, int itertions);

#endif
