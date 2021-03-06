//
//  rasterizer.h
//  CudaRasterizer
//
//  Created by Robert Crosby on 4/21/12.
//  Copyright (c) 2012 In-Con. All rights reserved.
//

#ifndef CudaRasterizer_rasterizer_h
#define CudaRasterizer_rasterizer_h

#include "structures.h"

void rasterize_mesh(drawbuffer_t *buffers, mesh_t *mesh, int duplicates);

void clear_buffers_cuda(drawbuffer_t *buffers);
void rasterize_mesh_cuda(drawbuffer_t *buffers, mesh_t *mesh, int duplicates);

#endif
