//
//  linear_math.h
//  CudaRasterizer
//
//  Created by Robert Crosby on 4/21/12.
//  Copyright (c) 2012 In-Con. All rights reserved.
//

#ifndef CudaRasterizer_linear_math_h
#define CudaRasterizer_linear_math_h

typedef struct ivec2_st {
   int x;
   int y;
} ivec2_t;

typedef struct ivec3_st {
   int x;
   int y;
   int z;
} ivec3_t;

typedef struct vec2_st {
   float x;
   float y;
} vec2_t;

typedef struct vec3_st {
   float x;
   float y;
   float z;
} vec3_t;

typedef struct vec4_st {
   float x;
   float y;
   float z;
   float w;
} vec4_t;

typedef struct mat4_st {
   vec4_t x;
   vec4_t y;
   vec4_t z;
   vec4_t w;
} mat4_t;

void ivec2_max(ivec2_t *reslt, const ivec2_t *p1, const ivec2_t *p2);
void ivec2_min(ivec2_t *reslt, const ivec2_t *p1, const ivec2_t *p2);

float vec2_dot(const vec2_t *v1, const vec2_t *v2);

float vec3_length(const vec3_t *v);
vec3_t vec3_normalized(const vec3_t *v);
void vec3_normalize(vec3_t *v);
vec3_t vec3_add(const vec3_t *v1, const vec3_t *v2);
vec3_t vec3_subtract(const vec3_t *v1, const vec3_t *v2);
vec3_t vec3_scale(const vec3_t *v, float s);
vec3_t vec3_mult(const vec3_t *v1, const vec3_t *v2);
vec3_t vec3_cross(const vec3_t *v1, const vec3_t *v2);
float vec3_dot(const vec3_t *v1, const vec3_t *v2);
ivec2_t vec3_to_ivec2(const vec3_t *v);

mat4_t mat4_identity();
mat4_t mat4_translation(float x, float y, float z);
void mat4_translate3f(mat4_t *m, float x, float y, float z);
mat4_t mat4_scaled(float x, float y, float z);
void mat4_scale1f(mat4_t *m, float s);
void mat4_scale3f(mat4_t *m, float x, float y, float z);
mat4_t mat4_rotation(float degrees, float axisX, float axisY, float axisZ);
void mat4_rotate4f(mat4_t *m, float degrees, float axisX, float axisY, float axisZ);
void mat4_copy_to(mat4_t *mA, const mat4_t *mB);
void mat4_mult(mat4_t *mA, const mat4_t *mB);
vec3_t mat4_translate_point(const mat4_t *m, const vec3_t *pt);

#endif
