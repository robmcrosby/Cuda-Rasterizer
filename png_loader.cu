//
//  png_loader.c
//  rasterizer
//
//  Created by Robert Crosby on 4/15/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include "png_loader.h"

int save_png_to_file(bitmap_t *bitmap, const char *path) {
   FILE *fp;
   int x, y;
   png_structp png_ptr = NULL;
   png_infop info_ptr;
   png_byte ** row_pointers = NULL;
   int status = -1;
   int pixel_size = 3;
   int depth = 8;
   
   fp = fopen(path, "wb");
   if (!fp)
      return status;
   
   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (png_ptr == NULL) {
      fclose(fp);
      return status;
   }
   
   info_ptr = png_create_info_struct(png_ptr);
   if (info_ptr == NULL) {
      png_destroy_write_struct(&png_ptr, &info_ptr);
      fclose(fp);
      return status;
   }
   
   if (setjmp(png_jmpbuf(png_ptr))) {
      png_destroy_write_struct(&png_ptr, &info_ptr);
      fclose(fp);
      return status;
   }
   
   // set image attributes.
   png_set_IHDR(png_ptr,
                info_ptr,
                bitmap->width,
                bitmap->height,
                depth,
                PNG_COLOR_TYPE_RGB,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT);
   
   // Initialize rows
   row_pointers = (png_byte **) png_malloc(png_ptr, bitmap->height * sizeof(png_byte*));
   for (y = 0; y < bitmap->height; ++y) {
      png_byte *row = (png_byte *) png_malloc(png_ptr, sizeof(uint8_t) * bitmap->width * pixel_size);
      row_pointers[y] = row;
      for (x = 0; x < bitmap->width; ++x) {
         color_t *pixel = bitmap_pixel_at(bitmap, x, y);
         *row++ = pixel->red;
         *row++ = pixel->green;
         *row++ = pixel->blue;
      }
   }
   
   // Write the image data
   png_init_io(png_ptr, fp);
   png_set_rows(png_ptr, info_ptr, row_pointers);
   png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
   status = 0;
   
   for (y = 0; y < bitmap->height; ++y)
      png_free(png_ptr, row_pointers[y]);
   png_free(png_ptr, row_pointers);
   
   png_destroy_write_struct(&png_ptr, &info_ptr);
   fclose(fp);
   return status;
}