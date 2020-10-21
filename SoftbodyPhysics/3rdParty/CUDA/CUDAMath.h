/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#pragma once

#include <QVector3D>
#include <Eigen/Core>

#include "cuda_runtime.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
    return make_float2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

////////////////////////////////////////////////////////////////////////////////
// matrix math
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float3 abs(float3 v)
{
	return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

struct float3x3
{
	float3 column0;
	float3 column1;
	float3 column2;
};

inline __device__ __host__ float3x3 transpose(float3x3 m)
{
	float3x3 m_transpose;
	m_transpose.column0.x = m.column0.x;
	m_transpose.column0.y = m.column1.x;
	m_transpose.column0.z = m.column2.x;

	m_transpose.column1.x = m.column0.y;
	m_transpose.column1.y = m.column1.y;
	m_transpose.column1.z = m.column2.y;

	m_transpose.column2.x = m.column0.z;
	m_transpose.column2.y = m.column1.z;
	m_transpose.column2.z = m.column2.z;
	return m_transpose;
}

inline __device__ __host__ float3x3 make_float3x3(float angle, float3 axis)
{
	float3x3 m;
	float sinAngle = sin(angle);
	float cosAngle = cos(angle);
	float oneMinusCosAngle = 1.f - cosAngle;
	float axisXY = axis.x * axis.y;
	float axisYZ = axis.y * axis.z;
	float axisZX = axis.z * axis.x;
	m.column0.x = cosAngle + axis.x * axis.x * oneMinusCosAngle;
	m.column0.y = axisXY * oneMinusCosAngle + axis.z * sinAngle;
	m.column0.z = axisZX * oneMinusCosAngle - axis.y * sinAngle;
	m.column1.x = axisXY * oneMinusCosAngle - axis.z * sinAngle;
	m.column1.y = cosAngle + axis.y * axis.y * oneMinusCosAngle;
	m.column1.z = axisYZ * oneMinusCosAngle + axis.x * sinAngle;
	m.column2.x = axisZX * oneMinusCosAngle + axis.y * sinAngle;
	m.column2.y = axisYZ * oneMinusCosAngle - axis.x * sinAngle;
	m.column2.z = cosAngle + axis.z * axis.z * oneMinusCosAngle;
	return m;
}

inline __device__ __host__ float3x3 make_float3x3(float s)
{
	float3x3 m;
	m.column0.x = s;
	m.column0.y = s;
	m.column0.z = s;
	m.column1.x = s;
	m.column1.y = s;
	m.column1.z = s;
	m.column2.x = s;
	m.column2.y = s;
	m.column2.z = s;
	return m;
}

inline __device__ __host__ float det(float3x3 mat)
{
	return mat.column0.x * (mat.column1.y * mat.column2.z - mat.column2.y * mat.column1.z) - 
		   mat.column1.x * (mat.column0.y * mat.column2.z - mat.column2.y * mat.column0.z) + 
		   mat.column2.x * (mat.column0.y * mat.column1.z - mat.column1.y * mat.column0.z);
}

// matrix * vector
inline __device__ __host__ float3 operator*(float3x3 m, float3 v)
{
	return make_float3(m.column0.x * v.x + m.column1.x * v.y + m.column2.x * v.z,
					   m.column0.y * v.x + m.column1.y * v.y + m.column2.y * v.z,
					   m.column0.z * v.x + m.column1.z * v.y + m.column2.z * v.z);
}

// vector x vector
inline __device__ __host__ float3 crossProduct(float3 v1, float3 v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y,
					   v1.z * v2.x - v1.x * v2.z,
					   v1.x * v2.y - v1.y * v2.x);
}

// vector * vector^T
inline __device__ __host__ float3x3 outerProduct(float3 v1, float3 v2)
{
	float3x3 m;
	m.column0.x = v1.x * v2.x;
	m.column0.y = v1.y * v2.x;
	m.column0.z = v1.z * v2.x;
	m.column1.x = v1.x * v2.y;
	m.column1.y = v1.y * v2.y;
	m.column1.z = v1.z * v2.y;
	m.column2.x = v1.x * v2.z;
	m.column2.y = v1.y * v2.z;
	m.column2.z = v1.z * v2.z;
	return m;
}

// scalar * matrix
inline __device__ __host__ float3x3 operator*(float s, float3x3 m1)
{
	float3x3 m2;
	m2.column0.x = s * m1.column0.x;
	m2.column0.y = s * m1.column0.y;
	m2.column0.z = s * m1.column0.z;
	m2.column1.x = s * m1.column1.x;
	m2.column1.y = s * m1.column1.y;
	m2.column1.z = s * m1.column1.z;
	m2.column2.x = s * m1.column2.x;
	m2.column2.y = s * m1.column2.y;
	m2.column2.z = s * m1.column2.z;
	return m2;
}

// matrix * matrix
inline __device__ __host__ float3x3 operator*(float3x3 m1, float3x3 m2)
{
	float3x3 m3;
	m3.column0.x = m1.column0.x * m2.column0.x + m1.column1.x * m2.column0.y + m1.column2.x * m2.column0.z;
	m3.column0.y = m1.column0.y * m2.column0.x + m1.column1.y * m2.column0.y + m1.column2.y * m2.column0.z;
	m3.column0.z = m1.column0.z * m2.column0.x + m1.column1.z * m2.column0.y + m1.column2.z * m2.column0.z;
	m3.column1.x = m1.column0.x * m2.column1.x + m1.column1.x * m2.column1.y + m1.column2.x * m2.column1.z;
	m3.column1.y = m1.column0.y * m2.column1.x + m1.column1.y * m2.column1.y + m1.column2.y * m2.column1.z;
	m3.column1.z = m1.column0.z * m2.column1.x + m1.column1.z * m2.column1.y + m1.column2.z * m2.column1.z;
	m3.column2.x = m1.column0.x * m2.column2.x + m1.column1.x * m2.column2.y + m1.column2.x * m2.column2.z;
	m3.column2.y = m1.column0.y * m2.column2.x + m1.column1.y * m2.column2.y + m1.column2.y * m2.column2.z;
	m3.column2.z = m1.column0.z * m2.column2.x + m1.column1.z * m2.column2.y + m1.column2.z * m2.column2.z;
	return m3;
}

inline __host__ __device__ void operator+=(float3x3 &m1, float3x3 m2)
{
	m1.column0.x += m2.column0.x;
	m1.column0.y += m2.column0.y;
	m1.column0.z += m2.column0.z;
	m1.column1.x += m2.column1.x;
	m1.column1.y += m2.column1.y;
	m1.column1.z += m2.column1.z;
	m1.column2.x += m2.column2.x;
	m1.column2.y += m2.column2.y;
	m1.column2.z += m2.column2.z;
}

inline __host__ __device__ float3x3 operator+(float3x3 m1, float3x3 m2)
{
	float3x3 m3;
	m3.column0.x = m1.column0.x + m2.column0.x;
	m3.column0.y = m1.column0.y + m2.column0.y;
	m3.column0.z = m1.column0.z + m2.column0.z;
	m3.column1.x = m1.column1.x + m2.column1.x;
	m3.column1.y = m1.column1.y + m2.column1.y;
	m3.column1.z = m1.column1.z + m2.column1.z;
	m3.column2.x = m1.column2.x + m2.column2.x;
	m3.column2.y = m1.column2.y + m2.column2.y;
	m3.column2.z = m1.column2.z + m2.column2.z;
	return m3;
}

inline __host__ __device__ void operator-=(float3x3 &m1, float3x3 m2)
{
	m1.column0.x -= m2.column0.x;
	m1.column0.y -= m2.column0.y;
	m1.column0.z -= m2.column0.z;
	m1.column1.x -= m2.column1.x;
	m1.column1.y -= m2.column1.y;
	m1.column1.z -= m2.column1.z;
	m1.column2.x -= m2.column2.x;
	m1.column2.y -= m2.column2.y;
	m1.column2.z -= m2.column2.z;
}

inline __host__ __device__ float3x3 operator-(float3x3 m1, float3x3 m2)
{
	float3x3 m3;
	m3.column0.x = m1.column0.x - m2.column0.x;
	m3.column0.y = m1.column0.y - m2.column0.y;
	m3.column0.z = m1.column0.z - m2.column0.z;
	m3.column1.x = m1.column1.x - m2.column1.x;
	m3.column1.y = m1.column1.y - m2.column1.y;
	m3.column1.z = m1.column1.z - m2.column1.z;
	m3.column2.x = m1.column2.x - m2.column2.x;
	m3.column2.y = m1.column2.y - m2.column2.y;
	m3.column2.z = m1.column2.z - m2.column2.z;
	return m3;
}

struct float9
{
	float x0;
	float y0;
	float z0;
	float x1;
	float y1;
	float z1;
	float x2;
	float y2;
	float z2;
};

inline __device__ __host__ float9 make_float9(float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2)
{
	float9 v;
	v.x0 = x0;
	v.y0 = y0;
	v.z0 = z0;
	v.x1 = x1;
	v.y1 = y1;
	v.z1 = z1;
	v.x2 = x2;
	v.y2 = y2;
	v.z2 = z2;
	return v;
}

struct float3x9
{
	float3 column0;
	float3 column1;
	float3 column2;
	float3 column3;
	float3 column4;
	float3 column5;
	float3 column6;
	float3 column7;
	float3 column8;
};

inline __device__ __host__ float frobeniusNorm(float3x3 m)
{
	float frobeniusNorm = 0.f;
	frobeniusNorm += powf(fabsf(m.column0.x), 2) + powf(fabsf(m.column0.y), 2) + powf(fabsf(m.column0.z), 2);
	frobeniusNorm += powf(fabsf(m.column1.x), 2) + powf(fabsf(m.column1.y), 2) + powf(fabsf(m.column1.z), 2);
	frobeniusNorm += powf(fabsf(m.column2.x), 2) + powf(fabsf(m.column2.y), 2) + powf(fabsf(m.column2.z), 2);
	return frobeniusNorm;
}

inline __device__ __host__ float frobeniusNorm(float3x9 m)
{
	float frobeniusNorm = 0.f;
	frobeniusNorm += powf(fabsf(m.column0.x), 2) + powf(fabsf(m.column0.y), 2) + powf(fabsf(m.column0.z), 2);
	frobeniusNorm += powf(fabsf(m.column1.x), 2) + powf(fabsf(m.column1.y), 2) + powf(fabsf(m.column1.z), 2);
	frobeniusNorm += powf(fabsf(m.column2.x), 2) + powf(fabsf(m.column2.y), 2) + powf(fabsf(m.column2.z), 2);
	frobeniusNorm += powf(fabsf(m.column3.x), 2) + powf(fabsf(m.column3.y), 2) + powf(fabsf(m.column3.z), 2);
	frobeniusNorm += powf(fabsf(m.column4.x), 2) + powf(fabsf(m.column4.y), 2) + powf(fabsf(m.column4.z), 2);
	frobeniusNorm += powf(fabsf(m.column5.x), 2) + powf(fabsf(m.column5.y), 2) + powf(fabsf(m.column5.z), 2);
	frobeniusNorm += powf(fabsf(m.column6.x), 2) + powf(fabsf(m.column6.y), 2) + powf(fabsf(m.column6.z), 2);
	frobeniusNorm += powf(fabsf(m.column7.x), 2) + powf(fabsf(m.column7.y), 2) + powf(fabsf(m.column7.z), 2);
	frobeniusNorm += powf(fabsf(m.column8.x), 2) + powf(fabsf(m.column8.y), 2) + powf(fabsf(m.column8.z), 2);
	return frobeniusNorm;
}

inline __device__ __host__ float3x9 make_float3x9(float s)
{
	float3x9 m;
	m.column0 = make_float3(s);
	m.column1 = make_float3(s);
	m.column2 = make_float3(s);
	m.column3 = make_float3(s);
	m.column4 = make_float3(s);
	m.column5 = make_float3(s);
	m.column6 = make_float3(s);
	m.column7 = make_float3(s);
	m.column8 = make_float3(s);
	return m;
}

inline __device__ __host__ float3x9 make_float3x9(float3x3 m1)
{
	float3x9 m2;
	m2.column0 = m1.column0;
	m2.column1 = m1.column1;
	m2.column2 = m1.column2;
	m2.column3 = make_float3(0.0f);
	m2.column4 = make_float3(0.0f);
	m2.column5 = make_float3(0.0f);
	m2.column6 = make_float3(0.0f);
	m2.column7 = make_float3(0.0f);
	m2.column8 = make_float3(0.0f);
	return m2;
}

// 3d vector * 9d vector^T
inline __device__ __host__ float3x9 outerProduct(float3 v1, float9 v2)
{
	float3x9 m;
	m.column0.x = v1.x * v2.x0;
	m.column0.y = v1.y * v2.x0;
	m.column0.z = v1.z * v2.x0;
	m.column1.x = v1.x * v2.y0;
	m.column1.y = v1.y * v2.y0;
	m.column1.z = v1.z * v2.y0;
	m.column2.x = v1.x * v2.z0;
	m.column2.y = v1.y * v2.z0;
	m.column2.z = v1.z * v2.z0;

	m.column3.x = v1.x * v2.x1;
	m.column3.y = v1.y * v2.x1;
	m.column3.z = v1.z * v2.x1;
	m.column4.x = v1.x * v2.y1;
	m.column4.y = v1.y * v2.y1;
	m.column4.z = v1.z * v2.y1;
	m.column5.x = v1.x * v2.z1;
	m.column5.y = v1.y * v2.z1;
	m.column5.z = v1.z * v2.z1;

	m.column6.x = v1.x * v2.x2;
	m.column6.y = v1.y * v2.x2;
	m.column6.z = v1.z * v2.x2;
	m.column7.x = v1.x * v2.y2;
	m.column7.y = v1.y * v2.y2;
	m.column7.z = v1.z * v2.y2;
	m.column8.x = v1.x * v2.z2;
	m.column8.y = v1.y * v2.z2;
	m.column8.z = v1.z * v2.z2;
	return m;
}

inline __host__ __device__ void operator+=(float3x9 &m1, float3x9 m2)
{
	m1.column0.x += m2.column0.x;
	m1.column0.y += m2.column0.y;
	m1.column0.z += m2.column0.z;
	m1.column1.x += m2.column1.x;
	m1.column1.y += m2.column1.y;
	m1.column1.z += m2.column1.z;
	m1.column2.x += m2.column2.x;
	m1.column2.y += m2.column2.y;
	m1.column2.z += m2.column2.z;

	m1.column3.x += m2.column3.x;
	m1.column3.y += m2.column3.y;
	m1.column3.z += m2.column3.z;
	m1.column4.x += m2.column4.x;
	m1.column4.y += m2.column4.y;
	m1.column4.z += m2.column4.z;
	m1.column5.x += m2.column5.x;
	m1.column5.y += m2.column5.y;
	m1.column5.z += m2.column5.z;

	m1.column6.x += m2.column6.x;
	m1.column6.y += m2.column6.y;
	m1.column6.z += m2.column6.z;
	m1.column7.x += m2.column7.x;
	m1.column7.y += m2.column7.y;
	m1.column7.z += m2.column7.z;
	m1.column8.x += m2.column8.x;
	m1.column8.y += m2.column8.y;
	m1.column8.z += m2.column8.z;
}


inline __host__ __device__ float3x9 operator+(float3x9 m1, float3x9 m2)
{
	float3x9 m3;
	m3.column0.x = m1.column0.x + m2.column0.x;
	m3.column0.y = m1.column0.y + m2.column0.y;
	m3.column0.z = m1.column0.z + m2.column0.z;
	m3.column1.x = m1.column1.x + m2.column1.x;
	m3.column1.y = m1.column1.y + m2.column1.y;
	m3.column1.z = m1.column1.z + m2.column1.z;
	m3.column2.x = m1.column2.x + m2.column2.x;
	m3.column2.y = m1.column2.y + m2.column2.y;
	m3.column2.z = m1.column2.z + m2.column2.z;

	m3.column3.x = m1.column3.x + m2.column3.x;
	m3.column3.y = m1.column3.y + m2.column3.y;
	m3.column3.z = m1.column3.z + m2.column3.z;
	m3.column4.x = m1.column4.x + m2.column4.x;
	m3.column4.y = m1.column4.y + m2.column4.y;
	m3.column4.z = m1.column4.z + m2.column4.z;
	m3.column5.x = m1.column5.x + m2.column5.x;
	m3.column5.y = m1.column5.y + m2.column5.y;
	m3.column5.z = m1.column5.z + m2.column5.z;

	m3.column6.x = m1.column6.x + m2.column6.x;
	m3.column6.y = m1.column6.y + m2.column6.y;
	m3.column6.z = m1.column6.z + m2.column6.z;
	m3.column7.x = m1.column7.x + m2.column7.x;
	m3.column7.y = m1.column7.y + m2.column7.y;
	m3.column7.z = m1.column7.z + m2.column7.z;
	m3.column8.x = m1.column8.x + m2.column8.x;
	m3.column8.y = m1.column8.y + m2.column8.y;
	m3.column8.z = m1.column8.z + m2.column8.z;
	return m3;
}

inline __device__ __host__ float3x9 operator*(float s, float3x9 m1)
{
	float3x9 m2;
	m2.column0.x = s * m1.column0.x;
	m2.column0.y = s * m1.column0.y;
	m2.column0.z = s * m1.column0.z;
	m2.column1.x = s * m1.column1.x;
	m2.column1.y = s * m1.column1.y;
	m2.column1.z = s * m1.column1.z;
	m2.column2.x = s * m1.column2.x;
	m2.column2.y = s * m1.column2.y;
	m2.column2.z = s * m1.column2.z;

	m2.column3.x = s * m1.column3.x;
	m2.column3.y = s * m1.column3.y;
	m2.column3.z = s * m1.column3.z;
	m2.column4.x = s * m1.column4.x;
	m2.column4.y = s * m1.column4.y;
	m2.column4.z = s * m1.column4.z;
	m2.column5.x = s * m1.column5.x;
	m2.column5.y = s * m1.column5.y;
	m2.column5.z = s * m1.column5.z;

	m2.column6.x = s * m1.column6.x;
	m2.column6.y = s * m1.column6.y;
	m2.column6.z = s * m1.column6.z;
	m2.column7.x = s * m1.column7.x;
	m2.column7.y = s * m1.column7.y;
	m2.column7.z = s * m1.column7.z;
	m2.column8.x = s * m1.column8.x;
	m2.column8.y = s * m1.column8.y;
	m2.column8.z = s * m1.column8.z;
	return m2;
}

struct float9x9
{
	float9 column0;
	float9 column1;
	float9 column2;
	float9 column3;
	float9 column4;
	float9 column5;
	float9 column6;
	float9 column7;
	float9 column8;
};

inline __device__ __host__ float3x9 operator*(float3x9 m1, float9x9 m2)
{
	float3x9 m3;
	m3.column0.x = m1.column0.x * m2.column0.x0 + m1.column1.x * m2.column0.y0 + m1.column2.x * m2.column0.z0 +
				   m1.column3.x * m2.column0.x1 + m1.column4.x * m2.column0.y1 + m1.column5.x * m2.column0.z1 +
				   m1.column6.x * m2.column0.x2 + m1.column7.x * m2.column0.y2 + m1.column8.x * m2.column0.z2;
	m3.column0.y = m1.column0.y * m2.column0.x0 + m1.column1.y * m2.column0.y0 + m1.column2.y * m2.column0.z0 +
				   m1.column3.y * m2.column0.x1 + m1.column4.y * m2.column0.y1 + m1.column5.y * m2.column0.z1 +
				   m1.column6.y * m2.column0.x2 + m1.column7.y * m2.column0.y2 + m1.column8.y * m2.column0.z2;
	m3.column0.z = m1.column0.z * m2.column0.x0 + m1.column1.z * m2.column0.y0 + m1.column2.z * m2.column0.z0 +
				   m1.column3.z * m2.column0.x1 + m1.column4.z * m2.column0.y1 + m1.column5.z * m2.column0.z1 +
				   m1.column6.z * m2.column0.x2 + m1.column7.z * m2.column0.y2 + m1.column8.z * m2.column0.z2;

	m3.column1.x = m1.column0.x * m2.column1.x0 + m1.column1.x * m2.column1.y0 + m1.column2.x * m2.column1.z0 +
				   m1.column3.x * m2.column1.x1 + m1.column4.x * m2.column1.y1 + m1.column5.x * m2.column1.z1 +
				   m1.column6.x * m2.column1.x2 + m1.column7.x * m2.column1.y2 + m1.column8.x * m2.column1.z2;
	m3.column1.y = m1.column0.y * m2.column1.x0 + m1.column1.y * m2.column1.y0 + m1.column2.y * m2.column1.z0 +
				   m1.column3.y * m2.column1.x1 + m1.column4.y * m2.column1.y1 + m1.column5.y * m2.column1.z1 +
				   m1.column6.y * m2.column1.x2 + m1.column7.y * m2.column1.y2 + m1.column8.y * m2.column1.z2;
	m3.column1.z = m1.column0.z * m2.column1.x0 + m1.column1.z * m2.column1.y0 + m1.column2.z * m2.column1.z0 +
				   m1.column3.z * m2.column1.x1 + m1.column4.z * m2.column1.y1 + m1.column5.z * m2.column1.z1 +
				   m1.column6.z * m2.column1.x2 + m1.column7.z * m2.column1.y2 + m1.column8.z * m2.column1.z2;

	m3.column2.x = m1.column0.x * m2.column2.x0 + m1.column1.x * m2.column2.y0 + m1.column2.x * m2.column2.z0 +
				   m1.column3.x * m2.column2.x1 + m1.column4.x * m2.column2.y1 + m1.column5.x * m2.column2.z1 +
				   m1.column6.x * m2.column2.x2 + m1.column7.x * m2.column2.y2 + m1.column8.x * m2.column2.z2;
	m3.column2.y = m1.column0.y * m2.column2.x0 + m1.column1.y * m2.column2.y0 + m1.column2.y * m2.column2.z0 +
				   m1.column3.y * m2.column2.x1 + m1.column4.y * m2.column2.y1 + m1.column5.y * m2.column2.z1 +
				   m1.column6.y * m2.column2.x2 + m1.column7.y * m2.column2.y2 + m1.column8.y * m2.column2.z2;
	m3.column2.z = m1.column0.z * m2.column2.x0 + m1.column1.z * m2.column2.y0 + m1.column2.z * m2.column2.z0 +
				   m1.column3.z * m2.column2.x1 + m1.column4.z * m2.column2.y1 + m1.column5.z * m2.column2.z1 +
				   m1.column6.z * m2.column2.x2 + m1.column7.z * m2.column2.y2 + m1.column8.z * m2.column2.z2;

	m3.column3.x = m1.column0.x * m2.column3.x0 + m1.column1.x * m2.column3.y0 + m1.column2.x * m2.column3.z0 +
				   m1.column3.x * m2.column3.x1 + m1.column4.x * m2.column3.y1 + m1.column5.x * m2.column3.z1 +
				   m1.column6.x * m2.column3.x2 + m1.column7.x * m2.column3.y2 + m1.column8.x * m2.column3.z2;
	m3.column3.y = m1.column0.y * m2.column3.x0 + m1.column1.y * m2.column3.y0 + m1.column2.y * m2.column3.z0 +
				   m1.column3.y * m2.column3.x1 + m1.column4.y * m2.column3.y1 + m1.column5.y * m2.column3.z1 +
				   m1.column6.y * m2.column3.x2 + m1.column7.y * m2.column3.y2 + m1.column8.y * m2.column3.z2;
	m3.column3.z = m1.column0.z * m2.column3.x0 + m1.column1.z * m2.column3.y0 + m1.column2.z * m2.column3.z0 +
				   m1.column3.z * m2.column3.x1 + m1.column4.z * m2.column3.y1 + m1.column5.z * m2.column3.z1 +
				   m1.column6.z * m2.column3.x2 + m1.column7.z * m2.column3.y2 + m1.column8.z * m2.column3.z2;

	m3.column4.x = m1.column0.x * m2.column4.x0 + m1.column1.x * m2.column4.y0 + m1.column2.x * m2.column4.z0 +
				   m1.column3.x * m2.column4.x1 + m1.column4.x * m2.column4.y1 + m1.column5.x * m2.column4.z1 +
				   m1.column6.x * m2.column4.x2 + m1.column7.x * m2.column4.y2 + m1.column8.x * m2.column4.z2;
	m3.column4.y = m1.column0.y * m2.column4.x0 + m1.column1.y * m2.column4.y0 + m1.column2.y * m2.column4.z0 +
				   m1.column3.y * m2.column4.x1 + m1.column4.y * m2.column4.y1 + m1.column5.y * m2.column4.z1 +
				   m1.column6.y * m2.column4.x2 + m1.column7.y * m2.column4.y2 + m1.column8.y * m2.column4.z2;
	m3.column4.z = m1.column0.z * m2.column4.x0 + m1.column1.z * m2.column4.y0 + m1.column2.z * m2.column4.z0 +
				   m1.column3.z * m2.column4.x1 + m1.column4.z * m2.column4.y1 + m1.column5.z * m2.column4.z1 +
				   m1.column6.z * m2.column4.x2 + m1.column7.z * m2.column4.y2 + m1.column8.z * m2.column4.z2;

	m3.column5.x = m1.column0.x * m2.column5.x0 + m1.column1.x * m2.column5.y0 + m1.column2.x * m2.column5.z0 +
				   m1.column3.x * m2.column5.x1 + m1.column4.x * m2.column5.y1 + m1.column5.x * m2.column5.z1 +
				   m1.column6.x * m2.column5.x2 + m1.column7.x * m2.column5.y2 + m1.column8.x * m2.column5.z2;
	m3.column5.y = m1.column0.y * m2.column5.x0 + m1.column1.y * m2.column5.y0 + m1.column2.y * m2.column5.z0 +
				   m1.column3.y * m2.column5.x1 + m1.column4.y * m2.column5.y1 + m1.column5.y * m2.column5.z1 +
				   m1.column6.y * m2.column5.x2 + m1.column7.y * m2.column5.y2 + m1.column8.y * m2.column5.z2;
	m3.column5.z = m1.column0.z * m2.column5.x0 + m1.column1.z * m2.column5.y0 + m1.column2.z * m2.column5.z0 +
				   m1.column3.z * m2.column5.x1 + m1.column4.z * m2.column5.y1 + m1.column5.z * m2.column5.z1 +
				   m1.column6.z * m2.column5.x2 + m1.column7.z * m2.column5.y2 + m1.column8.z * m2.column5.z2;

	m3.column6.x = m1.column0.x * m2.column6.x0 + m1.column1.x * m2.column6.y0 + m1.column2.x * m2.column6.z0 +
				   m1.column3.x * m2.column6.x1 + m1.column4.x * m2.column6.y1 + m1.column5.x * m2.column6.z1 +
				   m1.column6.x * m2.column6.x2 + m1.column7.x * m2.column6.y2 + m1.column8.x * m2.column6.z2;
	m3.column6.y = m1.column0.y * m2.column6.x0 + m1.column1.y * m2.column6.y0 + m1.column2.y * m2.column6.z0 +
				   m1.column3.y * m2.column6.x1 + m1.column4.y * m2.column6.y1 + m1.column5.y * m2.column6.z1 +
				   m1.column6.y * m2.column6.x2 + m1.column7.y * m2.column6.y2 + m1.column8.y * m2.column6.z2;
	m3.column6.z = m1.column0.z * m2.column6.x0 + m1.column1.z * m2.column6.y0 + m1.column2.z * m2.column6.z0 +
				   m1.column3.z * m2.column6.x1 + m1.column4.z * m2.column6.y1 + m1.column5.z * m2.column6.z1 +
				   m1.column6.z * m2.column6.x2 + m1.column7.z * m2.column6.y2 + m1.column8.z * m2.column6.z2;

	m3.column7.x = m1.column0.x * m2.column7.x0 + m1.column1.x * m2.column7.y0 + m1.column2.x * m2.column7.z0 +
				   m1.column3.x * m2.column7.x1 + m1.column4.x * m2.column7.y1 + m1.column5.x * m2.column7.z1 +
				   m1.column6.x * m2.column7.x2 + m1.column7.x * m2.column7.y2 + m1.column8.x * m2.column7.z2;
	m3.column7.y = m1.column0.y * m2.column7.x0 + m1.column1.y * m2.column7.y0 + m1.column2.y * m2.column7.z0 +
				   m1.column3.y * m2.column7.x1 + m1.column4.y * m2.column7.y1 + m1.column5.y * m2.column7.z1 +
				   m1.column6.y * m2.column7.x2 + m1.column7.y * m2.column7.y2 + m1.column8.y * m2.column7.z2;
	m3.column7.z = m1.column0.z * m2.column7.x0 + m1.column1.z * m2.column7.y0 + m1.column2.z * m2.column7.z0 +
				   m1.column3.z * m2.column7.x1 + m1.column4.z * m2.column7.y1 + m1.column5.z * m2.column7.z1 +
				   m1.column6.z * m2.column7.x2 + m1.column7.z * m2.column7.y2 + m1.column8.z * m2.column7.z2;

	m3.column8.x = m1.column0.x * m2.column8.x0 + m1.column1.x * m2.column8.y0 + m1.column2.x * m2.column8.z0 +
				   m1.column3.x * m2.column8.x1 + m1.column4.x * m2.column8.y1 + m1.column5.x * m2.column8.z1 +
				   m1.column6.x * m2.column8.x2 + m1.column7.x * m2.column8.y2 + m1.column8.x * m2.column8.z2;
	m3.column8.y = m1.column0.y * m2.column8.x0 + m1.column1.y * m2.column8.y0 + m1.column2.y * m2.column8.z0 +
				   m1.column3.y * m2.column8.x1 + m1.column4.y * m2.column8.y1 + m1.column5.y * m2.column8.z1 +
				   m1.column6.y * m2.column8.x2 + m1.column7.y * m2.column8.y2 + m1.column8.y * m2.column8.z2;
	m3.column8.z = m1.column0.z * m2.column8.x0 + m1.column1.z * m2.column8.y0 + m1.column2.z * m2.column8.z0 +
				   m1.column3.z * m2.column8.x1 + m1.column4.z * m2.column8.y1 + m1.column5.z * m2.column8.z1 +
				   m1.column6.z * m2.column8.x2 + m1.column7.z * m2.column8.y2 + m1.column8.z * m2.column8.z2;
	return m3;
}


inline __device__ __host__ float3 operator*(float3x9 m, float9 v)
{
	return make_float3(m.column0.x * v.x0 + m.column1.x * v.y0 + m.column2.x * v.z0 +
					   m.column3.x * v.x1 + m.column4.x * v.y1 + m.column5.x * v.z1 + 
					   m.column6.x * v.x2 + m.column7.x * v.y2 + m.column8.x * v.z2,
					   m.column0.y * v.x0 + m.column1.y * v.y0 + m.column2.y * v.z0 +
					   m.column3.y * v.x1 + m.column4.y * v.y1 + m.column5.y * v.z1 +
					   m.column6.y * v.x2 + m.column7.y * v.y2 + m.column8.y * v.z2,
					   m.column0.z * v.x0 + m.column1.z * v.y0 + m.column2.z * v.z0 +
					   m.column3.z * v.x1 + m.column4.z * v.y1 + m.column5.z * v.z1 +
					   m.column6.z * v.x2 + m.column7.z * v.y2 + m.column8.z * v.z2);
}

inline __host__ float3x3 convertToFloat3x3(const Eigen::Matrix3f& eigenMatrix)
{
	float3x3 matrix;
	matrix.column0.x = eigenMatrix.col(0).x();
	matrix.column0.y = eigenMatrix.col(0).y();
	matrix.column0.z = eigenMatrix.col(0).z();
	matrix.column1.x = eigenMatrix.col(1).x();
	matrix.column1.y = eigenMatrix.col(1).y();
	matrix.column1.z = eigenMatrix.col(1).z();
	matrix.column2.x = eigenMatrix.col(2).x();
	matrix.column2.y = eigenMatrix.col(2).y();
	matrix.column2.z = eigenMatrix.col(2).z();
	return matrix;
}

inline __host__ float9x9 convertToFloat9x9(const Eigen::MatrixXf& eigenMatrix)
{
	float9x9 matrix;	// row, column
	matrix.column0.x0 = eigenMatrix(0, 0);
	matrix.column0.y0 = eigenMatrix(1, 0);
	matrix.column0.z0 = eigenMatrix(2, 0);
	matrix.column0.x1 = eigenMatrix(3, 0);
	matrix.column0.y1 = eigenMatrix(4, 0);
	matrix.column0.z1 = eigenMatrix(5, 0);
	matrix.column0.x2 = eigenMatrix(6, 0);
	matrix.column0.y2 = eigenMatrix(7, 0);
	matrix.column0.z2 = eigenMatrix(8, 0);

	matrix.column1.x0 = eigenMatrix(0, 1);
	matrix.column1.y0 = eigenMatrix(1, 1);
	matrix.column1.z0 = eigenMatrix(2, 1);
	matrix.column1.x1 = eigenMatrix(3, 1);
	matrix.column1.y1 = eigenMatrix(4, 1);
	matrix.column1.z1 = eigenMatrix(5, 1);
	matrix.column1.x2 = eigenMatrix(6, 1);
	matrix.column1.y2 = eigenMatrix(7, 1);
	matrix.column1.z2 = eigenMatrix(8, 1);

	matrix.column2.x0 = eigenMatrix(0, 2);
	matrix.column2.y0 = eigenMatrix(1, 2);
	matrix.column2.z0 = eigenMatrix(2, 2);
	matrix.column2.x1 = eigenMatrix(3, 2);
	matrix.column2.y1 = eigenMatrix(4, 2);
	matrix.column2.z1 = eigenMatrix(5, 2);
	matrix.column2.x2 = eigenMatrix(6, 2);
	matrix.column2.y2 = eigenMatrix(7, 2);
	matrix.column2.z2 = eigenMatrix(8, 2);

	matrix.column3.x0 = eigenMatrix(0, 3);
	matrix.column3.y0 = eigenMatrix(1, 3);
	matrix.column3.z0 = eigenMatrix(2, 3);
	matrix.column3.x1 = eigenMatrix(3, 3);
	matrix.column3.y1 = eigenMatrix(4, 3);
	matrix.column3.z1 = eigenMatrix(5, 3);
	matrix.column3.x2 = eigenMatrix(6, 3);
	matrix.column3.y2 = eigenMatrix(7, 3);
	matrix.column3.z2 = eigenMatrix(8, 3);

	matrix.column4.x0 = eigenMatrix(0, 4);
	matrix.column4.y0 = eigenMatrix(1, 4);
	matrix.column4.z0 = eigenMatrix(2, 4);
	matrix.column4.x1 = eigenMatrix(3, 4);
	matrix.column4.y1 = eigenMatrix(4, 4);
	matrix.column4.z1 = eigenMatrix(5, 4);
	matrix.column4.x2 = eigenMatrix(6, 4);
	matrix.column4.y2 = eigenMatrix(7, 4);
	matrix.column4.z2 = eigenMatrix(8, 4);

	matrix.column5.x0 = eigenMatrix(0, 5);
	matrix.column5.y0 = eigenMatrix(1, 5);
	matrix.column5.z0 = eigenMatrix(2, 5);
	matrix.column5.x1 = eigenMatrix(3, 5);
	matrix.column5.y1 = eigenMatrix(4, 5);
	matrix.column5.z1 = eigenMatrix(5, 5);
	matrix.column5.x2 = eigenMatrix(6, 5);
	matrix.column5.y2 = eigenMatrix(7, 5);
	matrix.column5.z2 = eigenMatrix(8, 5);

	matrix.column6.x0 = eigenMatrix(0, 6);
	matrix.column6.y0 = eigenMatrix(1, 6);
	matrix.column6.z0 = eigenMatrix(2, 6);
	matrix.column6.x1 = eigenMatrix(3, 6);
	matrix.column6.y1 = eigenMatrix(4, 6);
	matrix.column6.z1 = eigenMatrix(5, 6);
	matrix.column6.x2 = eigenMatrix(6, 6);
	matrix.column6.y2 = eigenMatrix(7, 6);
	matrix.column6.z2 = eigenMatrix(8, 6);

	matrix.column7.x0 = eigenMatrix(0, 7);
	matrix.column7.y0 = eigenMatrix(1, 7);
	matrix.column7.z0 = eigenMatrix(2, 7);
	matrix.column7.x1 = eigenMatrix(3, 7);
	matrix.column7.y1 = eigenMatrix(4, 7);
	matrix.column7.z1 = eigenMatrix(5, 7);
	matrix.column7.x2 = eigenMatrix(6, 7);
	matrix.column7.y2 = eigenMatrix(7, 7);
	matrix.column7.z2 = eigenMatrix(8, 7);

	matrix.column8.x0 = eigenMatrix(0, 8);
	matrix.column8.y0 = eigenMatrix(1, 8);
	matrix.column8.z0 = eigenMatrix(2, 8);
	matrix.column8.x1 = eigenMatrix(3, 8);
	matrix.column8.y1 = eigenMatrix(4, 8);
	matrix.column8.z1 = eigenMatrix(5, 8);
	matrix.column8.x2 = eigenMatrix(6, 8);
	matrix.column8.y2 = eigenMatrix(7, 8);
	matrix.column8.z2 = eigenMatrix(8, 8);
	return matrix;
}

inline __host__ float3 convertToFloat3(const QVector3D& qVector)
{
	float3 vector;
	vector.x = qVector.x();
	vector.y = qVector.y();
	vector.z = qVector.z();
	return vector;
}