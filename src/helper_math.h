/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int uint;
typedef unsigned short ushort;
static_assert(sizeof(uint) == 4, "uint must be 32 bits.");
static_assert(sizeof(ushort) == 2, "ushort must be 16 bits.");

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

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
inline __host__ __device__ float2 make_float2(float4 a)
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

inline __host__ __device__ float3 make_float3(float s[3])
{
    return make_float3(s[0], s[1], s[2]);
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

inline __host__ __device__ uint2 make_uint2(float s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(float2 s)
{
    return make_uint2(s.x, s.y);
}

inline __host__ __device__ uint3 make_uint3(float s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(float3 s)
{
    return make_uint3(s.x, s.y, s.z);
}

inline __host__ __device__ uint4 make_uint4(float s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(float4 s)
{
    return make_uint4(s.x, s.y, s.z, s.w);
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

inline __host__ __device__ float2 operator-(float2& a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2& a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3& a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4& a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4& a)
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
inline __host__ __device__ void operator+=(float2& a, float2 b)
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
inline __host__ __device__ void operator+=(float2& a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2& a, int2 b)
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
inline __host__ __device__ void operator+=(int2& a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2& a, uint2 b)
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
inline __host__ __device__ void operator+=(uint2& a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(float3& a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3& a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3& a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3& a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3& a, uint b)
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
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4& a, float4 b)
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
inline __host__ __device__ void operator+=(float4& a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(int4& a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(int4& a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4& a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(uint4& a, uint b)
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
inline __host__ __device__ void operator-=(float2& a, float2 b)
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
inline __host__ __device__ void operator-=(float2& a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2& a, int2 b)
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
inline __host__ __device__ void operator-=(int2& a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2& a, uint2 b)
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
inline __host__ __device__ void operator-=(uint2& a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3& a, float3 b)
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
inline __host__ __device__ void operator-=(float3& a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3& a, int3 b)
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
inline __host__ __device__ void operator-=(int3& a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3& a, uint3 b)
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
inline __host__ __device__ void operator-=(uint3& a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4& a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(float4& a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(int4& a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4& a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4& a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4& a, uint b)
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
inline __host__ __device__ void operator*=(float2& a, float2 b)
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
inline __host__ __device__ void operator*=(float2& a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2& a, int2 b)
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
inline __host__ __device__ void operator*=(int2& a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2& a, uint2 b)
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
inline __host__ __device__ void operator*=(uint2& a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3& a, float3 b)
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
inline __host__ __device__ void operator*=(float3& a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3& a, int3 b)
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
inline __host__ __device__ void operator*=(int3& a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3& a, uint3 b)
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
inline __host__ __device__ void operator*=(uint3& a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float4& a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4& a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(int4& a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4& a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4& a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4& a, uint b)
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
inline __host__ __device__ void operator/=(float2& a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2& a, float b)
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
inline __host__ __device__ void operator/=(float3& a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3& a, float b)
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
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float4& a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(float4& a, float b)
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


inline __host__ __device__ uint2 operator/(uint2 a, uint2 b)
{
    return make_uint2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ uint3 operator/(uint3 a, uint3 b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ uint4 operator/(uint4 a, uint4 b)
{
    return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __host__ __device__ int2 operator/(int2 a, int2 b)
{
    return make_int2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ int4 operator/(int4 a, int4 b)
{
    return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}


inline __host__ __device__ float2 operator/(float2 a, uint2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float3 operator/(float3 a, uint3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float4 operator/(float4 a, uint4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __host__ __device__ float2 operator/(float2 a, int2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float3 operator/(float3 a, int3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float4 operator/(float4 a, int4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}


////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x, b.x), min(a.y, b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x, b.x), max(a.y, b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t * (b - a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t * (b - a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t * (b - a);
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
    return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
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
    return (y * y * (3.0f - (2.0f * y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float2(3.0f) - (make_float2(2.0f) * y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float3(3.0f) - (make_float3(2.0f) * y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float4(3.0f) - (make_float4(2.0f) * y)));
}

////////////////////////////////////////////////////////////////////////////////
// bitshift
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int2 operator<<(const int2 a, const int i)
{
    return make_int2(a.x << i, a.y << i);
}
inline __device__ __host__ int3 operator<<(const int3 a, const int i)
{
    return make_int3(a.x << i, a.y << i, a.z << i);
}
inline __device__ __host__ int4 operator<<(const int4 a, const int i)
{
    return make_int4(a.x << i, a.y << i, a.z << i, a.w << i);
}

inline __device__ __host__ uint2 operator<<(const uint2 a, const int i)
{
    return make_uint2(a.x << i, a.y << i);
}
inline __device__ __host__ uint3 operator<<(const uint3 a, const int i)
{
    return make_uint3(a.x << i, a.y << i, a.z << i);
}
inline __device__ __host__ uint4 operator<<(const uint4 a, const int i)
{
    return make_uint4(a.x << i, a.y << i, a.z << i, a.w << i);
}


inline __device__ __host__ int2 operator>>(const int2 a, const int i)
{
    return make_int2(a.x >> i, a.y >> i);
}
inline __device__ __host__ int3 operator>>(const int3 a, const int i)
{
    return make_int3(a.x >> i, a.y >> i, a.z >> i);
}
inline __device__ __host__ int4 operator>>(const int4 a, const int i)
{
    return make_int4(a.x >> i, a.y >> i, a.z >> i, a.w >> i);
}

inline __device__ __host__ uint2 operator>>(const uint2 a, const int i)
{
    return make_uint2(a.x >> i, a.y >> i);
}
inline __device__ __host__ uint3 operator>>(const uint3 a, const int i)
{
    return make_uint3(a.x >> i, a.y >> i, a.z >> i);
}
inline __device__ __host__ uint4 operator>>(const uint4 a, const int i)
{
    return make_uint4(a.x >> i, a.y >> i, a.z >> i, a.w >> i);
}

////////////////////////////////////////////////////////////////////////////////
// logical operations
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int2 operator^(const int2 a, const int2 b)
{
    return make_int2(a.x ^ b.x, a.y ^ b.y);
}
inline __device__ __host__ int3 operator^(const int3 a, const int3 b)
{
    return make_int3(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z);
}
inline __device__ __host__ int4 operator^(const int4 a, const int4 b)
{
    return make_int4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

inline __device__ __host__ uint2 operator^(const uint2 a, const uint2 b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}
inline __device__ __host__ uint3 operator^(const uint3 a, const uint3 b)
{
    return make_uint3(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z);
}
inline __device__ __host__ uint4 operator^(const uint4 a, const uint4 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

inline __device__ __host__ int2 operator&(const int2 a, const int2 b)
{
    return make_int2(a.x & b.x, a.y & b.y);
}
inline __device__ __host__ int3 operator&(const int3 a, const int3 b)
{
    return make_int3(a.x & b.x, a.y & b.y, a.z & b.z);
}
inline __device__ __host__ int4 operator&(const int4 a, const int4 b)
{
    return make_int4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}

inline __device__ __host__ uint2 operator&(const uint2 a, const uint2 b)
{
    return make_uint2(a.x & b.x, a.y & b.y);
}
inline __device__ __host__ uint3 operator&(const uint3 a, const uint3 b)
{
    return make_uint3(a.x & b.x, a.y & b.y, a.z & b.z);
}
inline __device__ __host__ uint4 operator&(const uint4 a, const uint4 b)
{
    return make_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}

inline __device__ __host__ int2 operator|(const int2 a, const int2 b)
{
    return make_int2(a.x | b.x, a.y | b.y);
}
inline __device__ __host__ int3 operator|(const int3 a, const int3 b)
{
    return make_int3(a.x | b.x, a.y | b.y, a.z | b.z);
}
inline __device__ __host__ int4 operator|(const int4 a, const int4 b)
{
    return make_int4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}

inline __device__ __host__ uint2 operator|(const uint2 a, const uint2 b)
{
    return make_uint2(a.x | b.x, a.y | b.y);
}
inline __device__ __host__ uint3 operator|(const uint3 a, const uint3 b)
{
    return make_uint3(a.x | b.x, a.y | b.y, a.z | b.z);
}
inline __device__ __host__ uint4 operator|(const uint4 a, const uint4 b)
{
    return make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}

////////////////////////////////////////////////////////////////////////////////
// pseudo-random number generation
////////////////////////////////////////////////////////////////////////////////

struct rng_state
{
    int4 state;
    __host__ __device__ rng_state(int a = -1, int b = -2, int c = -3, int d = -4) : state(make_int4(a, b, c, d)) { }
    __host__ __device__ void update_state(int s = 0, int iters = 3)
    {
        for (int i = 0; i < iters; i++)
        {
            int t = state.x ^ (s * 21481264 + i - 118947813) ^ (state.w << 3);
            t ^= t >> 5; t += state.y * (1481912741 ^ state.z);
            state.w = state.z; state.z = state.y; state.y = state.x;
            state.x = t ^ (t << 3);
        }
    }

    __host__ __device__ int gen_int(int s = 0)
    {
        update_state(s, 3);
        return state.x;
    }
    __host__ __device__ int2 gen_int2(int s = 0)
    {
        update_state(s, 3);
        int v1 = state.x;
        update_state(s, 2);
        int v2 = state.x;
        return make_int2(v1, v2);
    }
    __host__ __device__ int3 gen_int3(int s = 0)
    {
        update_state(s, 7);
        return make_int3(state.x, state.y, state.z);
    }
    __host__ __device__ int4 gen_int4(int s = 0)
    {
        update_state(s, 8);
        int v1 = state.x;
        int v2 = state.y;
        int v3 = state.z;
        update_state(s, 2);
        return make_int4(v1, v2, v3, state.x);
    }

    __host__ __device__ int gen_int(int min_incl, int max_excl, int s = 0)
    {
        update_state(s, 3);
        return (int)(((uint)state.x) % ((uint)(max_excl - min_incl))) + min_incl;
    }
    __host__ __device__ float gen_float(int s = 0)
    {
        return gen_int(s) / 2147483648.f;
    }
    __host__ __device__ float2 gen_float2(int s = 0)
    {
        return make_float2(gen_int2(s)) / 2147483648.f;
    }
    __host__ __device__ float3 gen_float3(int s = 0)
    {
        return make_float3(gen_int3(s)) / 2147483648.f;
    }
    __host__ __device__ float4 gen_float4(int s = 0)
    {
        return make_float4(gen_int4(s)) / 2147483648.f;
    }
};

////////////////////////////////////////////////////////////////////////////////
// swizzle
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ int2 yx(const int2 a)
{
    return make_int2(a.y, a.x);
}

inline __device__ __host__ int3 yxz(const int3 a)
{
    return make_int3(a.y, a.x, a.z);
}
inline __device__ __host__ int3 yzx(const int3 a)
{
    return make_int3(a.y, a.z, a.x);
}
inline __device__ __host__ int3 zyx(const int3 a)
{
    return make_int3(a.z, a.y, a.x);
}
inline __device__ __host__ int3 zxy(const int3 a)
{
    return make_int3(a.z, a.x, a.y);
}
inline __device__ __host__ int3 xzy(const int3 a)
{
    return make_int3(a.x, a.z, a.y);
}

inline __device__ __host__ int4 yxzw(const int4 a) { return make_int4(a.y, a.x, a.z, a.w); }
inline __device__ __host__ int4 yzxw(const int4 a) { return make_int4(a.y, a.z, a.x, a.w); }
inline __device__ __host__ int4 zyxw(const int4 a) { return make_int4(a.z, a.y, a.x, a.w); }
inline __device__ __host__ int4 zxyw(const int4 a) { return make_int4(a.z, a.x, a.y, a.w); }
inline __device__ __host__ int4 xzyw(const int4 a) { return make_int4(a.x, a.z, a.y, a.w); }
inline __device__ __host__ int4 xywz(const int4 a) { return make_int4(a.x, a.y, a.w, a.z); }
inline __device__ __host__ int4 yxwz(const int4 a) { return make_int4(a.y, a.x, a.w, a.z); }
inline __device__ __host__ int4 yzwx(const int4 a) { return make_int4(a.y, a.z, a.w, a.x); }
inline __device__ __host__ int4 zywx(const int4 a) { return make_int4(a.z, a.y, a.w, a.x); }
inline __device__ __host__ int4 zxwy(const int4 a) { return make_int4(a.z, a.x, a.w, a.y); }
inline __device__ __host__ int4 xzwy(const int4 a) { return make_int4(a.x, a.z, a.w, a.y); }
inline __device__ __host__ int4 xwyz(const int4 a) { return make_int4(a.x, a.w, a.y, a.z); }
inline __device__ __host__ int4 ywxz(const int4 a) { return make_int4(a.y, a.w, a.x, a.z); }
inline __device__ __host__ int4 ywzx(const int4 a) { return make_int4(a.y, a.w, a.z, a.x); }
inline __device__ __host__ int4 zwyx(const int4 a) { return make_int4(a.z, a.w, a.y, a.x); }
inline __device__ __host__ int4 zwxy(const int4 a) { return make_int4(a.z, a.w, a.x, a.y); }
inline __device__ __host__ int4 xwzy(const int4 a) { return make_int4(a.x, a.w, a.z, a.y); }
inline __device__ __host__ int4 wxyz(const int4 a) { return make_int4(a.w, a.x, a.y, a.z); }
inline __device__ __host__ int4 wyxz(const int4 a) { return make_int4(a.w, a.y, a.x, a.z); }
inline __device__ __host__ int4 wyzx(const int4 a) { return make_int4(a.w, a.y, a.z, a.x); }
inline __device__ __host__ int4 wzyx(const int4 a) { return make_int4(a.w, a.z, a.y, a.x); }
inline __device__ __host__ int4 wzxy(const int4 a) { return make_int4(a.w, a.z, a.x, a.y); }
inline __device__ __host__ int4 wxzy(const int4 a) { return make_int4(a.w, a.x, a.z, a.y); }


inline __device__ __host__ float2 yx(const float2 a)
{
    return make_float2(a.y, a.x);
}

inline __device__ __host__ float3 yxz(const float3 a)
{
    return make_float3(a.y, a.x, a.z);
}
inline __device__ __host__ float3 yzx(const float3 a)
{
    return make_float3(a.y, a.z, a.x);
}
inline __device__ __host__ float3 zyx(const float3 a)
{
    return make_float3(a.z, a.y, a.x);
}
inline __device__ __host__ float3 zxy(const float3 a)
{
    return make_float3(a.z, a.x, a.y);
}
inline __device__ __host__ float3 xzy(const float3 a)
{
    return make_float3(a.x, a.z, a.y);
}

inline __device__ __host__ float4 yxzw(const float4 a) { return make_float4(a.y, a.x, a.z, a.w); }
inline __device__ __host__ float4 yzxw(const float4 a) { return make_float4(a.y, a.z, a.x, a.w); }
inline __device__ __host__ float4 zyxw(const float4 a) { return make_float4(a.z, a.y, a.x, a.w); }
inline __device__ __host__ float4 zxyw(const float4 a) { return make_float4(a.z, a.x, a.y, a.w); }
inline __device__ __host__ float4 xzyw(const float4 a) { return make_float4(a.x, a.z, a.y, a.w); }
inline __device__ __host__ float4 xywz(const float4 a) { return make_float4(a.x, a.y, a.w, a.z); }
inline __device__ __host__ float4 yxwz(const float4 a) { return make_float4(a.y, a.x, a.w, a.z); }
inline __device__ __host__ float4 yzwx(const float4 a) { return make_float4(a.y, a.z, a.w, a.x); }
inline __device__ __host__ float4 zywx(const float4 a) { return make_float4(a.z, a.y, a.w, a.x); }
inline __device__ __host__ float4 zxwy(const float4 a) { return make_float4(a.z, a.x, a.w, a.y); }
inline __device__ __host__ float4 xzwy(const float4 a) { return make_float4(a.x, a.z, a.w, a.y); }
inline __device__ __host__ float4 xwyz(const float4 a) { return make_float4(a.x, a.w, a.y, a.z); }
inline __device__ __host__ float4 ywxz(const float4 a) { return make_float4(a.y, a.w, a.x, a.z); }
inline __device__ __host__ float4 ywzx(const float4 a) { return make_float4(a.y, a.w, a.z, a.x); }
inline __device__ __host__ float4 zwyx(const float4 a) { return make_float4(a.z, a.w, a.y, a.x); }
inline __device__ __host__ float4 zwxy(const float4 a) { return make_float4(a.z, a.w, a.x, a.y); }
inline __device__ __host__ float4 xwzy(const float4 a) { return make_float4(a.x, a.w, a.z, a.y); }
inline __device__ __host__ float4 wxyz(const float4 a) { return make_float4(a.w, a.x, a.y, a.z); }
inline __device__ __host__ float4 wyxz(const float4 a) { return make_float4(a.w, a.y, a.x, a.z); }
inline __device__ __host__ float4 wyzx(const float4 a) { return make_float4(a.w, a.y, a.z, a.x); }
inline __device__ __host__ float4 wzyx(const float4 a) { return make_float4(a.w, a.z, a.y, a.x); }
inline __device__ __host__ float4 wzxy(const float4 a) { return make_float4(a.w, a.z, a.x, a.y); }
inline __device__ __host__ float4 wxzy(const float4 a) { return make_float4(a.w, a.x, a.z, a.y); }


inline __device__ __host__ uint2 yx(const uint2 a)
{
    return make_uint2(a.y, a.x);
}

inline __device__ __host__ uint3 yxz(const uint3 a)
{
    return make_uint3(a.y, a.x, a.z);
}
inline __device__ __host__ uint3 yzx(const uint3 a)
{
    return make_uint3(a.y, a.z, a.x);
}
inline __device__ __host__ uint3 zyx(const uint3 a)
{
    return make_uint3(a.z, a.y, a.x);
}
inline __device__ __host__ uint3 zxy(const uint3 a)
{
    return make_uint3(a.z, a.x, a.y);
}
inline __device__ __host__ uint3 xzy(const uint3 a)
{
    return make_uint3(a.x, a.z, a.y);
}

inline __device__ __host__ uint4 yxzw(const uint4 a) { return make_uint4(a.y, a.x, a.z, a.w); }
inline __device__ __host__ uint4 yzxw(const uint4 a) { return make_uint4(a.y, a.z, a.x, a.w); }
inline __device__ __host__ uint4 zyxw(const uint4 a) { return make_uint4(a.z, a.y, a.x, a.w); }
inline __device__ __host__ uint4 zxyw(const uint4 a) { return make_uint4(a.z, a.x, a.y, a.w); }
inline __device__ __host__ uint4 xzyw(const uint4 a) { return make_uint4(a.x, a.z, a.y, a.w); }
inline __device__ __host__ uint4 xywz(const uint4 a) { return make_uint4(a.x, a.y, a.w, a.z); }
inline __device__ __host__ uint4 yxwz(const uint4 a) { return make_uint4(a.y, a.x, a.w, a.z); }
inline __device__ __host__ uint4 yzwx(const uint4 a) { return make_uint4(a.y, a.z, a.w, a.x); }
inline __device__ __host__ uint4 zywx(const uint4 a) { return make_uint4(a.z, a.y, a.w, a.x); }
inline __device__ __host__ uint4 zxwy(const uint4 a) { return make_uint4(a.z, a.x, a.w, a.y); }
inline __device__ __host__ uint4 xzwy(const uint4 a) { return make_uint4(a.x, a.z, a.w, a.y); }
inline __device__ __host__ uint4 xwyz(const uint4 a) { return make_uint4(a.x, a.w, a.y, a.z); }
inline __device__ __host__ uint4 ywxz(const uint4 a) { return make_uint4(a.y, a.w, a.x, a.z); }
inline __device__ __host__ uint4 ywzx(const uint4 a) { return make_uint4(a.y, a.w, a.z, a.x); }
inline __device__ __host__ uint4 zwyx(const uint4 a) { return make_uint4(a.z, a.w, a.y, a.x); }
inline __device__ __host__ uint4 zwxy(const uint4 a) { return make_uint4(a.z, a.w, a.x, a.y); }
inline __device__ __host__ uint4 xwzy(const uint4 a) { return make_uint4(a.x, a.w, a.z, a.y); }
inline __device__ __host__ uint4 wxyz(const uint4 a) { return make_uint4(a.w, a.x, a.y, a.z); }
inline __device__ __host__ uint4 wyxz(const uint4 a) { return make_uint4(a.w, a.y, a.x, a.z); }
inline __device__ __host__ uint4 wyzx(const uint4 a) { return make_uint4(a.w, a.y, a.z, a.x); }
inline __device__ __host__ uint4 wzyx(const uint4 a) { return make_uint4(a.w, a.z, a.y, a.x); }
inline __device__ __host__ uint4 wzxy(const uint4 a) { return make_uint4(a.w, a.z, a.x, a.y); }
inline __device__ __host__ uint4 wxzy(const uint4 a) { return make_uint4(a.w, a.x, a.z, a.y); }


inline __host__ __device__ bool operator==(int2 a, int2 b)
{
    return a.x == b.x && a.y == b.y;
}
inline __host__ __device__ bool operator==(int3 a, int3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ bool operator==(int4 a, int4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline __host__ __device__ bool operator==(float2 a, float2 b)
{
    return a.x == b.x && a.y == b.y;
}
inline __host__ __device__ bool operator==(float3 a, float3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ bool operator==(float4 a, float4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline __host__ __device__ bool operator==(uint2 a, uint2 b)
{
    return a.x == b.x && a.y == b.y;
}
inline __host__ __device__ bool operator==(uint3 a, uint3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ bool operator==(uint4 a, uint4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}


inline __host__ __device__ bool operator!=(int2 a, int2 b)
{
    return a.x != b.x || a.y != b.y;
}
inline __host__ __device__ bool operator!=(int3 a, int3 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
inline __host__ __device__ bool operator!=(int4 a, int4 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

inline __host__ __device__ bool operator!=(float2 a, float2 b)
{
    return a.x != b.x || a.y != b.y;
}
inline __host__ __device__ bool operator!=(float3 a, float3 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
inline __host__ __device__ bool operator!=(float4 a, float4 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

inline __host__ __device__ bool operator!=(uint2 a, uint2 b)
{
    return a.x != b.x || a.y != b.y;
}
inline __host__ __device__ bool operator!=(uint3 a, uint3 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
inline __host__ __device__ bool operator!=(uint4 a, uint4 b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}


#include <string>
inline std::string to_string(const int2& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y);
}
inline std::string to_string(const int3& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z);
}
inline std::string to_string(const int4& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w);
}

inline std::string to_string(const float2& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y);
}
inline std::string to_string(const float3& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z);
}
inline std::string to_string(const float4& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w);
}

inline std::string to_string(const uint2& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y);
}
inline std::string to_string(const uint3& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z);
}
inline std::string to_string(const uint4& a)
{
    return std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ", " + std::to_string(a.w);
}


inline __host__ __device__ int sign(int a)
{
    return (a > 0) - (a < 0);
}
inline __host__ __device__ int sign(float a)
{
    return (a > 0) - (a < 0);
}

inline __host__ __device__ int2 sign(int2 a)
{
    return make_int2(sign(a.x), sign(a.y));
}
inline __host__ __device__ int3 sign(int3 a)
{
    return make_int3(sign(a.x), sign(a.y), sign(a.z));
}
inline __host__ __device__ int4 sign(int4 a)
{
    return make_int4(sign(a.x), sign(a.y), sign(a.z), sign(a.w));
}

inline __host__ __device__ int2 sign(float2 a)
{
    return make_int2(sign(a.x), sign(a.y));
}
inline __host__ __device__ int3 sign(float3 a)
{
    return make_int3(sign(a.x), sign(a.y), sign(a.z));
}
inline __host__ __device__ int4 sign(float4 a)
{
    return make_int4(sign(a.x), sign(a.y), sign(a.z), sign(a.w));
}


inline __host__ __device__ float2 ceilf(float2 f)
{
    return make_float2(ceilf(f.x), ceilf(f.y));
}

inline __host__ __device__ float3 ceilf(float3 f)
{
    return make_float3(ceilf(f.x), ceilf(f.y), ceilf(f.z));
}

inline __host__ __device__ float4 ceilf(float4 f)
{
    return make_float4(ceilf(f.x), ceilf(f.y), ceilf(f.z), ceilf(f.w));
}


inline __host__ __device__ float2 roundf(float2 f)
{
    return make_float2(roundf(f.x), roundf(f.y));
}

inline __host__ __device__ float3 roundf(float3 f)
{
    return make_float3(roundf(f.x), roundf(f.y), roundf(f.z));
}

inline __host__ __device__ float4 roundf(float4 f)
{
    return make_float4(roundf(f.x), roundf(f.y), roundf(f.z), roundf(f.w));
}


inline __host__ __device__ int sign(int a, int low, int high)
{
    return (a > 0) * high + (a < 0) * low;
}
inline __host__ __device__ int sign(float a, int low, int high)
{
    return (a > 0) * high + (a < 0) * low;
}

inline __host__ __device__ int2 sign(int2 a, int low, int high)
{
    return make_int2(sign(a.x, low, high), sign(a.y, low, high));
}
inline __host__ __device__ int3 sign(int3 a, int low, int high)
{
    return make_int3(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high));
}
inline __host__ __device__ int4 sign(int4 a, int low, int high)
{
    return make_int4(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high), sign(a.w, low, high));
}

inline __host__ __device__ int2 sign(float2 a, int low, int high)
{
    return make_int2(sign(a.x, low, high), sign(a.y, low, high));
}
inline __host__ __device__ int3 sign(float3 a, int low, int high)
{
    return make_int3(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high));
}
inline __host__ __device__ int4 sign(float4 a, int low, int high)
{
    return make_int4(sign(a.x, low, high), sign(a.y, low, high), sign(a.z, low, high), sign(a.w, low, high));
}


inline __host__ __device__ int2 round_intf(float2 a)
{
    return make_int2(rintf(a.x), rintf(a.y));
}
inline __host__ __device__ int3 round_intf(float3 a)
{
    return make_int3(rintf(a.x), rintf(a.y), rintf(a.z));
}
inline __host__ __device__ int4 round_intf(float4 a)
{
    return make_int4(rintf(a.x), rintf(a.y), rintf(a.z), rintf(a.w));
}


inline __host__ __device__ int2 mod(int2 a, int2 b)
{
    return make_int2(a.x % b.x, a.y % b.y);
}

inline __host__ __device__ int3 mod(int3 a, int3 b)
{
    return make_int3(a.x % b.x, a.y % b.y, a.z % b.z);
}

inline __host__ __device__ int4 mod(int4 a, int4 b)
{
    return make_int4(a.x % b.x, a.y % b.y, b.z % b.z, b.w % b.w);
}

inline __host__ __device__ uint2 mod(uint2 a, uint2 b)
{
    return make_uint2(a.x % b.x, a.y % b.y);
}

inline __host__ __device__ uint3 mod(uint3 a, uint3 b)
{
    return make_uint3(a.x % b.x, a.y % b.y, a.z % b.z);
}

inline __host__ __device__ uint4 mod(uint4 a, uint4 b)
{
    return make_uint4(a.x % b.x, a.y % b.y, a.z % b.z, a.w % b.w);
}

inline __host__ __device__ uint2 mod(uint2 a, uint b)
{
    return make_uint2(a.x % b, a.y % b);
}

inline __host__ __device__ uint3 mod(uint3 a, uint b)
{
    return make_uint3(a.x % b, a.y % b, a.z % b);
}

inline __host__ __device__ uint4 mod(uint4 a, uint b)
{
    return make_uint4(a.x % b, a.y % b, a.z % b, a.w % b);
}

// M(a,b)
inline __host__ __device__ float arithmetic_geometric_mean(float a, float b)
{
    float ta = 0.f;
#pragma unroll
    for (int i = 0; i < 30; i++)
    {
        ta = sqrtf(a * b);
        b = (a + b) * .5f;
        a = ta;
    }
    return (a + b) * .5f;
}

// EllipticK[k]
inline __host__ __device__ float elliptic_integral_first_kind_mathematica(float k)
{
    return 1.57079632679f / arithmetic_geometric_mean(1.f, sqrtf(1.f - k));
}

// Integral from 0 to pi/2 of 1/Sqrt(1-k^2sin^2(x)) dx
inline __host__ __device__ float elliptic_integral_first_kind(float k)
{
    return elliptic_integral_first_kind_mathematica(k * k);
}

// erf(x)
inline __host__ __device__ float erf_lossy(float x)
{
    const int sgn = sign(x); x = fabsf(x); float temp = x * x;
    x = 1.f + 0.278393f * x + 0.230389f * temp + 0.000972f * temp * x + 0.078108f * temp * temp; x *= x; x *= x;
    return (1.f - 1.f / x) * sgn;
}

// W_f(x)
inline __host__ __device__ float lambert_W_fast(const float x)
{
    float guess = (x > 1.5f) ? logf(x) * .66666666666666f + .333333333333333f : x / (1.f + x);
    guess = (x * expf(-guess) + guess * guess) / (1.f + guess);
    guess = (x * expf(-guess) + guess * guess) / (1.f + guess);
    guess = (x * expf(-guess) + guess * guess) / (1.f + guess);
    return (x * expf(-guess) + guess * guess) / (1.f + guess);
}

// W(x)
inline __host__ __device__ float lambert_W(const float x)
{
    float guess = (x > 2.15f) ? logf(x) * .66666666666666f + .333333333333333f : (2.f * x * (3.f + 4.f * x)) / (6.f + 14.f * x + 5.f * x * x);
    guess = (x * expf(-guess) + guess * guess) / (1.f + guess);
    guess = (x * expf(-guess) + guess * guess) / (1.f + guess);
    guess = (x * expf(-guess) + guess * guess) / (1.f + guess);
    guess = (x * expf(-guess) + guess * guess) / (1.f + guess);
    return (x * expf(-guess) + guess * guess) / (1.f + guess);
}


inline __host__ __device__ int global_max(int2 a)
{
    return a.x > a.y ? a.x : a.y;
}

inline __host__ __device__ int global_min(int2 a)
{
    return a.x < a.y ? a.x : a.y;
}

inline __host__ __device__ uint global_max(uint2 a)
{
    return a.x > a.y ? a.x : a.y;
}

inline __host__ __device__ uint global_min(uint2 a)
{
    return a.x < a.y ? a.x : a.y;
}

inline __host__ __device__ float global_max(float2 a)
{
    return a.x > a.y ? a.x : a.y;
}

inline __host__ __device__ float global_min(float2 a)
{
    return a.x < a.y ? a.x : a.y;
}


inline __host__ __device__ float global_max(float3 a)
{
    return fmaxf(a.x, fmaxf(a.y, a.z));
}

inline __host__ __device__ float global_min(float3 a)
{
    return fminf(a.x, fminf(a.y, a.z));
}

inline __host__ __device__ float global_max(float4 a)
{
    return fmaxf(a.x, fmaxf(a.y, fmaxf(a.z, a.w)));
}

inline __host__ __device__ float global_min(float4 a)
{
    return fminf(a.x, fminf(a.y, fminf(a.z, a.w)));
}


inline __host__ __device__ int global_max(int3 a)
{
    return max(a.x, max(a.y, a.z));
}

inline __host__ __device__ int global_min(int3 a)
{
    return min(a.x, min(a.y, a.z));
}

inline __host__ __device__ int global_max(int4 a)
{
    return max(a.x, max(a.y, max(a.z, a.w)));
}

inline __host__ __device__ int global_min(int4 a)
{
    return min(a.x, min(a.y, min(a.z, a.w)));
}


inline __host__ __device__ uint global_max(uint3 a)
{
    return max(a.x, max(a.y, a.z));
}

inline __host__ __device__ uint global_min(uint3 a)
{
    return min(a.x, min(a.y, a.z));
}

inline __host__ __device__ uint global_max(uint4 a)
{
    return max(a.x, max(a.y, max(a.z, a.w)));
}

inline __host__ __device__ uint global_min(uint4 a)
{
    return min(a.x, min(a.y, min(a.z, a.w)));
}


inline __host__ __device__ float4 sqrtf(float4 a)
{
    return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
}
inline __host__ __device__ float3 sqrtf(float3 a)
{
    return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}
inline __host__ __device__ float2 sqrtf(float2 a)
{
    return make_float2(sqrtf(a.x), sqrtf(a.y));
}

inline __host__ __device__ float4 powf(float4 a, float p)
{
    return make_float4(powf(a.x, p), powf(a.y, p), powf(a.z, p), powf(a.w, p));
}
inline __host__ __device__ float3 powf(float3 a, float p)
{
    return make_float3(powf(a.x, p), powf(a.y, p), powf(a.z, p));
}
inline __host__ __device__ float2 powf(float2 a, float p)
{
    return make_float2(powf(a.x, p), powf(a.y, p));
}

inline __device__ __host__ float4 soft_normalize(const float4 v)
{
    return v * rsqrtf(dot(v, v) + 1E-15f);
}

inline __device__ __host__ float3 soft_normalize(const float3 v)
{
    return v * rsqrtf(dot(v, v) + 1E-15f);
}

inline __device__ __host__ float2 soft_normalize(const float2 v)
{
    return v * rsqrtf(dot(v, v) + 1E-15f);
}

inline __device__ __host__ float modf_corr(const float v, const float m)
{
    return v - floorf(v / m) * m;
}

inline __device__ __host__ float modf_mid_point(const float v, const float m)
{
    return v - floorf(v / m + 0.5f) * m;
}

inline __device__ __host__ float arccothf(const float x)
{
    return 0.5f * logf((x + 1.f) / (x - 1.f));
}
inline __device__ __host__ double arccoth(const double x)
{
    return 0.5 * log((x + 1.) / (x - 1.));
}

typedef long long ilong;
typedef unsigned long long ulong;

inline __device__ __host__ ulong gcdl(ulong a, ulong b)
{
    ulong temp;
    while (b != 0)
    {
        temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}
inline __device__ __host__ uint gcd(uint a, uint b)
{
    uint temp;
    while (b != 0)
    {
        temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

struct rational
{
    int numerator;
    uint denominator;

    __host__ __device__ rational(int num, uint den = 1)
    {
        uint g = gcd((ulong)abs(num), den);
        g = (g > 1) ? g : 1;
        numerator = num / (int)g;
        denominator = den / g;
    }
    __host__ __device__ rational(ilong num, ulong den, const bool _0)
    {
        while ((den > 4294967295u || num > 2147483647 || num < -2147483647) && num != 0)
        {
            num >>= 1;
            den >>= 1;
        }

        ulong g = gcdl((ulong)abs(num), den);
        g = (g > 1) ? g : 1; num /= (ilong)g; den /= g;

        numerator = (int)num;
        denominator = (uint)den;
    }

    __host__ __device__ rational operator/(const int& a) const {
        return rational(numerator * (2 * (ilong)(a > 0) - 1), denominator * (ulong)abs(a), false);
    }
    __host__ __device__ rational operator*(const int& a) const {
        return rational(numerator * (ilong)a, denominator, false);
    }
    __host__ __device__ rational operator+(const int& a) const {
        return rational(numerator + a * (ilong)denominator, denominator, false);
    }
    __host__ __device__ rational operator-(const int& a) const {
        return rational(numerator - a * (ilong)denominator, denominator, false);
    }

    __host__ __device__ rational operator/(const rational& a) const {
        return rational(numerator * (ilong)a.denominator * (2 * (ilong)(a.numerator > 0) - 1), denominator * (ulong)abs((ilong)a.numerator), false);
    }
    __host__ __device__ rational operator*(const rational& a) const {
        return rational(numerator * (ilong)a.numerator, (ulong)denominator * a.denominator, false);
    }
    __host__ __device__ rational operator+(const rational& a) const {
        return rational(numerator * (ilong)a.denominator + a.numerator * (ilong)denominator, (ulong)denominator * a.denominator, false);
    }
    __host__ __device__ rational operator-(const rational& a) const {
        return rational(numerator * (ilong)a.denominator - a.numerator * (ilong)denominator, (ulong)denominator * a.denominator, false);
    }

    __host__ __device__ float to_float() const {
        return ((float)numerator) / denominator;
    }
    __host__ __device__ double to_double() const {
        return ((double)numerator) / denominator;
    }
    __host__ __device__ int to_int() const {
        return numerator / denominator;
    }
};

inline std::string to_string(const rational& a)
{
    if (a.denominator == 0)
        return a.numerator == 0 ? "NaN" : "Unsigned Infinity";
    else if (a.denominator == 1)
        return std::to_string(a.numerator);

    return std::to_string(a.numerator) + " / " + std::to_string(a.denominator);
}


struct float3x3
{
    float mat[9];

    __host__ __device__ float3x3() : mat() {}
    __host__ __device__ float3x3(const float3 x, const float3 y, const float3 z)
    {
        mat[0] = x.x;
        mat[1] = x.y;
        mat[2] = x.z;

        mat[3] = y.x;
        mat[4] = y.y;
        mat[5] = y.z;

        mat[6] = z.x;
        mat[7] = z.y;
        mat[8] = z.z;
    }
    inline __host__ __device__ float3 diag() const
    {
        return make_float3(mat[0], mat[4], mat[8]);
    }
    inline __host__ __device__ float index(const int c, const int r) const
    {
        return mat[c * 3 + r];
    }
    inline __host__ __device__ float3x3 transpose() const
    {
        float3x3 result;
        result.mat[0] = mat[0];
        result.mat[1] = mat[3];
        result.mat[2] = mat[6];
        result.mat[3] = mat[1];
        result.mat[4] = mat[4];
        result.mat[5] = mat[7];
        result.mat[6] = mat[2];
        result.mat[7] = mat[5];
        result.mat[8] = mat[8];
        return result;
    }
    inline __host__ __device__ void mad_column(const float3 v, float m, const int c)
    {
        mat[c * 3] += v.x * m;
        mat[c * 3 + 1] += v.y * m;
        mat[c * 3 + 2] += v.z * m;
    }
    inline __host__ __device__ void mad_row(const float3 v, float m, const int r)
    {
        mat[r] += v.x * m;
        mat[r + 3] += v.y * m;
        mat[r + 6] += v.z * m;
    }
    inline __host__ __device__ void set_column(const float3 v, const int c)
    {
        mat[c * 3] = v.x;
        mat[c * 3 + 1] = v.y;
        mat[c * 3 + 2] = v.z;
    }
    inline __host__ __device__ void set_row(const float3 v, const int r)
    {
        mat[r] = v.x;
        mat[r + 3] = v.y;
        mat[r + 6] = v.z;
    }
    inline __host__ __device__ float3 column(const int c) const
    {
        return make_float3(mat[c * 3], mat[c * 3 + 1], mat[c * 3 + 2]);
    }
    inline __host__ __device__ float3 row(const int r) const
    {
        return make_float3(mat[r], mat[r + 3], mat[r + 6]);
    }
    inline __host__ __device__ float3x3 operator-() const
    {
        float3x3 result;
        result.mat[0] = -mat[0];
        result.mat[1] = -mat[1];
        result.mat[2] = -mat[2];
        result.mat[3] = -mat[3];
        result.mat[4] = -mat[4];
        result.mat[5] = -mat[5];
        result.mat[6] = -mat[6];
        result.mat[7] = -mat[7];
        result.mat[8] = -mat[8];
        return result;
    }
    inline __host__ __device__ float3x3 operator-(const float3x3 a) const
    {
        float3x3 result;
        result.mat[0] = mat[0] - a.mat[0];
        result.mat[1] = mat[1] - a.mat[1];
        result.mat[2] = mat[2] - a.mat[2];
        result.mat[3] = mat[3] - a.mat[3];
        result.mat[4] = mat[4] - a.mat[4];
        result.mat[5] = mat[5] - a.mat[5];
        result.mat[6] = mat[6] - a.mat[6];
        result.mat[7] = mat[7] - a.mat[7];
        result.mat[8] = mat[8] - a.mat[8];
        return result;
    }
    inline __host__ __device__ float3x3 operator+(const float3x3 a) const
    {
        float3x3 result;
        result.mat[0] = mat[0] + a.mat[0];
        result.mat[1] = mat[1] + a.mat[1];
        result.mat[2] = mat[2] + a.mat[2];
        result.mat[3] = mat[3] + a.mat[3];
        result.mat[4] = mat[4] + a.mat[4];
        result.mat[5] = mat[5] + a.mat[5];
        result.mat[6] = mat[6] + a.mat[6];
        result.mat[7] = mat[7] + a.mat[7];
        result.mat[8] = mat[8] + a.mat[8];
        return result;
    }
    inline __host__ __device__ float3x3 operator/(const float a) const
    {
        float3x3 result;
        result.mat[0] = mat[0] / a;
        result.mat[1] = mat[1] / a;
        result.mat[2] = mat[2] / a;
        result.mat[3] = mat[3] / a;
        result.mat[4] = mat[4] / a;
        result.mat[5] = mat[5] / a;
        result.mat[6] = mat[6] / a;
        result.mat[7] = mat[7] / a;
        result.mat[8] = mat[8] / a;
        return result;
    }
    inline __host__ __device__ void operator*=(const float a) {
        mat[0] *= a;
        mat[1] *= a;
        mat[2] *= a;
        mat[3] *= a;
        mat[4] *= a;
        mat[5] *= a;
        mat[6] *= a;
        mat[7] *= a;
        mat[8] *= a;
    }
    inline __host__ __device__ void operator+=(const float3x3 a) {
        mat[0] += a.mat[0];
        mat[1] += a.mat[1];
        mat[2] += a.mat[2];
        mat[3] += a.mat[3];
        mat[4] += a.mat[4];
        mat[5] += a.mat[5];
        mat[6] += a.mat[6];
        mat[7] += a.mat[7];
        mat[8] += a.mat[8];
    }
    inline __host__ __device__ void operator-=(const float3x3 a) {
        mat[0] -= a.mat[0];
        mat[1] -= a.mat[1];
        mat[2] -= a.mat[2];
        mat[3] -= a.mat[3];
        mat[4] -= a.mat[4];
        mat[5] -= a.mat[5];
        mat[6] -= a.mat[6];
        mat[7] -= a.mat[7];
        mat[8] -= a.mat[8];
    }
    inline __host__ __device__ void operator/=(const float a) {
        mat[0] /= a;
        mat[1] /= a;
        mat[2] /= a;
        mat[3] /= a;
        mat[4] /= a;
        mat[5] /= a;
        mat[6] /= a;
        mat[7] /= a;
        mat[8] /= a;
    }
    inline __host__ __device__ float3x3 operator*(const float a) const
    {
        float3x3 result;
        result.mat[0] = mat[0] * a;
        result.mat[1] = mat[1] * a;
        result.mat[2] = mat[2] * a;
        result.mat[3] = mat[3] * a;
        result.mat[4] = mat[4] * a;
        result.mat[5] = mat[5] * a;
        result.mat[6] = mat[6] * a;
        result.mat[7] = mat[7] * a;
        result.mat[8] = mat[8] * a;
        return result;
    }
    inline __host__ __device__ float3 operator*(const float3 a) const
    {
        return column(0) * a.x + column(1) * a.y + column(2) * a.z;
    }
    inline __host__ __device__ float3x3 operator*(const float3x3 a) const
    {
        float3x3 result;
        result.mat[0] = dot(row(0), a.column(0));
        result.mat[1] = dot(row(1), a.column(0));
        result.mat[2] = dot(row(2), a.column(0));
        result.mat[3] = dot(row(0), a.column(1));
        result.mat[4] = dot(row(1), a.column(1));
        result.mat[5] = dot(row(2), a.column(1));
        result.mat[6] = dot(row(0), a.column(2));
        result.mat[7] = dot(row(1), a.column(2));
        result.mat[8] = dot(row(2), a.column(2));
        return result;
    }
    inline __host__ __device__ float trace() const { return mat[0] + mat[4] + mat[8]; }
    inline __host__ __device__ float determinant() const {
        return mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) - mat[3] * (mat[1] * mat[8] - mat[2] * mat[7]) + mat[6] * (mat[1] * mat[5] - mat[2] * mat[4]);
    }
    inline __host__ __device__ float3x3 inverse() const {
        float3x3 result;
        result.mat[0] = -mat[5] * mat[7] + mat[4] * mat[8];
        result.mat[3] = mat[5] * mat[6] - mat[3] * mat[8];
        result.mat[6] = -mat[4] * mat[6] + mat[3] * mat[7];
        result.mat[1] = mat[2] * mat[7] - mat[1] * mat[8];
        result.mat[4] = -mat[2] * mat[6] + mat[0] * mat[8];
        result.mat[7] = mat[1] * mat[6] - mat[0] * mat[7];
        result.mat[2] = -mat[2] * mat[4] + mat[1] * mat[5];
        result.mat[5] = mat[2] * mat[3] - mat[0] * mat[5];
        result.mat[8] = -mat[1] * mat[3] + mat[0] * mat[4];
        return result / determinant();
    }
};


#endif