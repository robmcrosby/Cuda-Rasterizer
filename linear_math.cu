#include <stdio.h>
#include <math.h>
#include "linear_math.h"

void ivec2_max(ivec2_t *reslt, const ivec2_t *p1, const ivec2_t *p2) {
   reslt->x = p1->x > p2->x ? p1->x : p2->x;
   reslt->y = p1->y > p2->y ? p1->y : p2->y;
}

void ivec2_min(ivec2_t *reslt, const ivec2_t *p1, const ivec2_t *p2) {
   reslt->x = p1->x < p2->x ? p1->x : p2->x;
   reslt->y = p1->y < p2->y ? p1->y : p2->y;
}

float vec2_dot(const vec2_t *v1, const vec2_t *v2) {
   return v1->x * v2->x + v1->y * v2->y;
}

float vec3_length(const vec3_t *v) {
   return sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
}

vec3_t vec3_normalized(const vec3_t *v) {
   vec3_t nv;
   float l = vec3_length(v);
   nv.x = v->x / l;
   nv.y = v->y / l;
   nv.z = v->z / l;
   return nv;
}

void vec3_normalize(vec3_t *v) {
   float l = vec3_length(v);
   v->x /= l;
   v->y /= l;
   v->z /= l;
}

vec3_t vec3_add(const vec3_t *v1, const vec3_t *v2) {
   vec3_t retV;
   retV.x = v1->x + v2->x;
   retV.y = v1->y + v2->y;
   retV.z = v1->z + v2->z;
   return retV;
}

vec3_t vec3_subtract(const vec3_t *v1, const vec3_t *v2) {
   vec3_t retV;
   retV.x = v1->x - v2->x;
   retV.y = v1->y - v2->y;
   retV.z = v1->z - v2->z;
   return retV;
}

vec3_t vec3_scale(const vec3_t *v, float s) {
   vec3_t retV;
   retV.x = v->x * s;
   retV.y = v->y * s;
   retV.z = v->z * s;
   return retV;
}

vec3_t vec3_mult(const vec3_t *v1, const vec3_t *v2) {
   vec3_t retV;
   retV.x = v1->x * v2->x;
   retV.y = v1->y * v2->y;
   retV.z = v1->z * v2->z;
   return retV;
}

vec3_t vec3_cross(const vec3_t *v1, const vec3_t *v2) {
   vec3_t vector;
   vector.x = v1->y * v2->z - v1->z * v2->y;
   vector.y = v1->z * v2->x - v1->x * v2->z;
   vector.z = v1->x * v2->y - v1->y * v2->x;
   return vector;
}

float vec3_dot(const vec3_t *v1, const vec3_t *v2) {
   return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

ivec2_t vec3_to_ivec2(const vec3_t *v) {
   ivec2_t retV;
   retV.x = floorf(v->x);
   retV.y = floorf(v->y);
   return retV;
}

mat4_t mat4_identity() {
   mat4_t m;
   m.x.x = 1; m.x.y = 0; m.x.z = 0; m.x.w = 0;
   m.y.x = 0; m.y.y = 1; m.y.z = 0; m.y.w = 0;
   m.z.x = 0; m.z.y = 0; m.z.z = 1; m.z.w = 0;
   m.w.x = 0; m.w.y = 0; m.w.z = 0; m.w.w = 1;
   return m;
}

mat4_t mat4_translation(float x, float y, float z) {
   mat4_t m;
   m.x.x = 1; m.x.y = 0; m.x.z = 0; m.x.w = 0;
   m.y.x = 0; m.y.y = 1; m.y.z = 0; m.y.w = 0;
   m.z.x = 0; m.z.y = 0; m.z.z = 1; m.z.w = 0;
   m.w.x = x; m.w.y = y; m.w.z = z; m.w.w = 1;
   return m;
}

void mat4_translate3f(mat4_t *m, float x, float y, float z) {
   mat4_t translation = mat4_translation(x, y, z);
   mat4_mult(m, &translation);
}

mat4_t mat4_scaled(float x, float y, float z) {
   mat4_t m;
   m.x.x = x; m.x.y = 0; m.x.z = 0; m.x.w = 0;
   m.y.x = 0; m.y.y = y; m.y.z = 0; m.y.w = 0;
   m.z.x = 0; m.z.y = 0; m.z.z = z; m.z.w = 0;
   m.w.x = 0; m.w.y = 0; m.w.z = 0; m.w.w = 1;
   return m;
}

void mat4_scale1f(mat4_t *m, float s) {
   mat4_t scaled = mat4_scaled(s, s, s);
   mat4_mult(m, &scaled);
}

void mat4_scale3f(mat4_t *m, float x, float y, float z) {
   mat4_t scaled = mat4_scaled(x, y, z);
   mat4_mult(m, &scaled);
}

mat4_t mat4_rotation(float degrees, float axisX, float axisY, float axisZ) {
   float radians = degrees * 3.14159f / 180.0f;
   float s = sinf(radians);
   float c = cosf(radians);
   
   mat4_t m = mat4_identity();
   m.x.x = c + (1 - c) * axisX * axisX;
   m.x.y = (1 - c) * axisX * axisY - axisZ * s;
   m.x.z = (1 - c) * axisX * axisZ + axisY * s;
   m.y.x = (1 - c) * axisX * axisY + axisZ * s;
   m.y.y = c + (1 - c) * axisY * axisY;
   m.y.z = (1 - c) * axisY * axisZ - axisX * s;
   m.z.x = (1 - c) * axisX * axisZ - axisY * s;
   m.z.y = (1 - c) * axisY * axisZ + axisX * s;
   m.z.z = c + (1 - c) * axisZ * axisZ;
   return m;
}

void mat4_rotate4f(mat4_t *m, float degrees, float axisX, float axisY, float axisZ) {
   mat4_t rotation = mat4_rotation(degrees, axisX, axisY, axisZ);
   mat4_mult(m, &rotation);
}

void mat4_copy_to(mat4_t *mA, const mat4_t *mB) {
   mA->x.x = mB->x.x; mA->x.y = mB->x.y; mA->x.z = mB->x.z; mA->x.w = mB->x.w;
   mA->y.x = mB->y.x; mA->y.y = mB->y.y; mA->y.z = mB->y.z; mA->y.w = mB->y.w;
   mA->z.x = mB->z.x; mA->z.y = mB->z.y; mA->z.z = mB->z.z; mA->z.w = mB->z.w;
   mA->w.x = mB->w.x; mA->w.y = mB->w.y; mA->w.z = mB->w.z; mA->w.w = mB->w.w;
}

void mat4_mult(mat4_t *mA, const mat4_t *mB) {
   mat4_t m;
   m.x.x = mA->x.x * mB->x.x + mA->x.y * mB->y.x + mA->x.z * mB->z.x + mA->x.w * mB->w.x;
   m.x.y = mA->x.x * mB->x.y + mA->x.y * mB->y.y + mA->x.z * mB->z.y + mA->x.w * mB->w.y;
   m.x.z = mA->x.x * mB->x.z + mA->x.y * mB->y.z + mA->x.z * mB->z.z + mA->x.w * mB->w.z;
   m.x.w = mA->x.x * mB->x.w + mA->x.y * mB->y.w + mA->x.z * mB->z.w + mA->x.w * mB->w.w;
   m.y.x = mA->y.x * mB->x.x + mA->y.y * mB->y.x + mA->y.z * mB->z.x + mA->y.w * mB->w.x;
   m.y.y = mA->y.x * mB->x.y + mA->y.y * mB->y.y + mA->y.z * mB->z.y + mA->y.w * mB->w.y;
   m.y.z = mA->y.x * mB->x.z + mA->y.y * mB->y.z + mA->y.z * mB->z.z + mA->y.w * mB->w.z;
   m.y.w = mA->y.x * mB->x.w + mA->y.y * mB->y.w + mA->y.z * mB->z.w + mA->y.w * mB->w.w;
   m.z.x = mA->z.x * mB->x.x + mA->z.y * mB->y.x + mA->z.z * mB->z.x + mA->z.w * mB->w.x;
   m.z.y = mA->z.x * mB->x.y + mA->z.y * mB->y.y + mA->z.z * mB->z.y + mA->z.w * mB->w.y;
   m.z.z = mA->z.x * mB->x.z + mA->z.y * mB->y.z + mA->z.z * mB->z.z + mA->z.w * mB->w.z;
   m.z.w = mA->z.x * mB->x.w + mA->z.y * mB->y.w + mA->z.z * mB->z.w + mA->z.w * mB->w.w;
   m.w.x = mA->w.x * mB->x.x + mA->w.y * mB->y.x + mA->w.z * mB->z.x + mA->w.w * mB->w.x;
   m.w.y = mA->w.x * mB->x.y + mA->w.y * mB->y.y + mA->w.z * mB->z.y + mA->w.w * mB->w.y;
   m.w.z = mA->w.x * mB->x.z + mA->w.y * mB->y.z + mA->w.z * mB->z.z + mA->w.w * mB->w.z;
   m.w.w = mA->w.x * mB->x.w + mA->w.y * mB->y.w + mA->w.z * mB->z.w + mA->w.w * mB->w.w;
   mat4_copy_to(mA, &m);
}

vec3_t mat4_translate_point(const mat4_t *m, const vec3_t *pt) {
   vec3_t newpt;
   float w;
   newpt.x = pt->x * m->x.x + pt->y * m->y.x + pt->z * m->z.x + m->w.x;
   newpt.y = pt->x * m->x.y + pt->y * m->y.y + pt->z * m->z.y + m->w.y;
   newpt.z = pt->x * m->x.z + pt->y * m->y.z + pt->z * m->z.z + m->w.z;
   w = pt->x * m->x.w + pt->y * m->y.w + pt->z * m->z.w + m->w.w;
   
   newpt.x /= w;
   newpt.y /= w;
   newpt.z /= w;
   return newpt;
}

