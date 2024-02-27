/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "math/matrix.h"

namespace vox {

CUDA_CALLABLE inline Vector3F closest_point_to_aabb(const Vector3F &p, const Vector3F &lower, const Vector3F &upper) {
    Vector3F c;

    {
        float v = p[0];
        if (v < lower[0]) v = lower[0];
        if (v > upper[0]) v = upper[0];
        c[0] = v;
    }

    {
        float v = p[1];
        if (v < lower[1]) v = lower[1];
        if (v > upper[1]) v = upper[1];
        c[1] = v;
    }

    {
        float v = p[2];
        if (v < lower[2]) v = lower[2];
        if (v > upper[2]) v = upper[2];
        c[2] = v;
    }

    return c;
}

CUDA_CALLABLE inline Vector2F closest_point_to_triangle(const Vector3F &a, const Vector3F &b, const Vector3F &c, const Vector3F &p) {
    Vector3F ab = b - a;
    Vector3F ac = c - a;
    Vector3F ap = p - a;

    float u, v, w;
    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        v = 0.0f;
        w = 0.0f;
        u = 1.0f - v - w;
        return Vector2F(u, v);
    }

    Vector3F bp = p - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        v = 1.0f;
        w = 0.0f;
        u = 1.0f - v - w;
        return Vector2F(u, v);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        v = d1 / (d1 - d3);
        w = 0.0f;
        u = 1.0f - v - w;
        return Vector2F(u, v);
    }

    Vector3F cp = p - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        v = 0.0f;
        w = 1.0f;
        u = 1.0f - v - w;
        return Vector2F(u, v);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        v = 0.0f;
        w = d2 / (d2 - d6);
        u = 1.0f - v - w;
        return Vector2F(u, v);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        v = 1.0f - w;
        u = 1.0f - v - w;
        return Vector2F(u, v);
    }

    float denom = 1.0f / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    u = 1.0f - v - w;
    return Vector2F(u, v);
}

CUDA_CALLABLE inline Vector2F furthest_point_to_triangle(const Vector3F &a, const Vector3F &b, const Vector3F &c, const Vector3F &p) {
    Vector3F pa = p - a;
    Vector3F pb = p - b;
    Vector3F pc = p - c;
    float dist_a = pa.dot(pa);
    float dist_b = pb.dot(pb);
    float dist_c = pc.dot(pc);

    if (dist_a > dist_b && dist_a > dist_c)
        return Vector2F(1.0f, 0.0f);// a is furthest
    if (dist_b > dist_c)
        return Vector2F(0.0f, 1.0f);// b is furthest
    return Vector2F(0.0f, 0.0f);    // c is furthest
}

CUDA_CALLABLE inline bool intersect_ray_aabb(const Vector3F &pos, const Vector3F &rcp_dir, const Vector3F &lower, const Vector3F &upper, float &t) {
    float l1, l2, lmin, lmax;

    l1 = (lower[0] - pos[0]) * rcp_dir[0];
    l2 = (upper[0] - pos[0]) * rcp_dir[0];
    lmin = ::min(l1, l2);
    lmax = ::max(l1, l2);

    l1 = (lower[1] - pos[1]) * rcp_dir[1];
    l2 = (upper[1] - pos[1]) * rcp_dir[1];
    lmin = ::max(::min(l1, l2), lmin);
    lmax = ::min(::max(l1, l2), lmax);

    l1 = (lower[2] - pos[2]) * rcp_dir[2];
    l2 = (upper[2] - pos[2]) * rcp_dir[2];
    lmin = ::max(::min(l1, l2), lmin);
    lmax = ::min(::max(l1, l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;

    return hit;
}

// Moller and Trumbore's method
CUDA_CALLABLE inline bool intersect_ray_tri_moller(const Vector3F &p, const Vector3F &dir, const Vector3F &a, const Vector3F &b, const Vector3F &c,
                                                   float &t, float &u, float &v, float &w, float &sign, Vector3F *normal) {
    Vector3F ab = b - a;
    Vector3F ac = c - a;
    Vector3F n = ab.cross(ac);

    float d = -dir.dot(n);
    float ood = 1.0f / d;// No need to check for division by zero here as infinity arithmetic will save us...
    Vector3F ap = p - a;

    t = ap.dot(n) * ood;
    if (t < 0.0f)
        return false;

    Vector3F e = -dir.cross(ap);
    v = ac.dot(e) * ood;
    if (v < 0.0f || v > 1.0f)// ...here...
        return false;
    w = -ab.dot(e) * ood;
    if (w < 0.0f || (v + w) > 1.0f)// ...and here
        return false;

    u = 1.0f - v - w;
    if (normal)
        *normal = n;

    sign = d;

    return true;
}

CUDA_CALLABLE inline bool intersect_ray_tri_rtcd(const Vector3F &p, const Vector3F &dir, const Vector3F &a, const Vector3F &b, const Vector3F &c,
                                                 float &t, float &u, float &v, float &w, float &sign, Vector3F *normal) {
    const Vector3F ab = b - a;
    const Vector3F ac = c - a;

    // calculate normal
    Vector3F n = ab.cross(ac);

    // need to solve a system of three equations to give t, u, v
    float d = -dir.dot(n);

    // if dir is parallel to triangle plane or points away from triangle
    if (d <= 0.0f)
        return false;

    Vector3F ap = p - a;
    t = ap.dot(n);

    // ignores tris behind
    if (t < 0.0f)
        return false;

    // compute barycentric coordinates
    Vector3F e = -dir.cross(ap);
    v = ac.dot(e);
    if (v < 0.0f || v > d) return false;

    w = -ab.dot(e);
    if (w < 0.0f || v + w > d) return false;

    float ood = 1.0f / d;
    t *= ood;
    v *= ood;
    w *= ood;
    u = 1.0f - v - w;

    // optionally write out normal (todo: this branch is a performance concern, should probably remove)
    if (normal)
        *normal = n;

    return true;
}

#ifndef __CUDA_ARCH__

// these are provided as built-ins by CUDA
inline float __int_as_float(int i) {
    return *(float *)(&i);
}

inline int __float_as_int(float f) {
    return *(int *)(&f);
}

#endif

CUDA_CALLABLE inline float xorf(float x, int y) {
    return __int_as_float(__float_as_int(x) ^ y);
}

CUDA_CALLABLE inline int sign_mask(float x) {
    return __float_as_int(x) & 0x80000000;
}

CUDA_CALLABLE inline int max_dim(Vector3F a) {
    float x = abs(a[0]);
    float y = abs(a[1]);
    float z = abs(a[2]);

    return Vector3F(x, y, z).dominantAxis();
}

// computes the difference of products a*b - c*d using
// FMA instructions for improved numerical precision
CUDA_CALLABLE inline float diff_product(float a, float b, float c, float d) {
    float cd = c * d;
    float diff = fmaf(a, b, -cd);
    float error = fmaf(-c, d, cd);

    return diff + error;
}

// http://jcgt.org/published/0002/01/05/
CUDA_CALLABLE inline bool intersect_ray_tri_woop(const Vector3F &p, const Vector3F &dir, const Vector3F &a, const Vector3F &b, const Vector3F &c, float &t, float &u, float &v, float &sign, Vector3F *normal) {
    // todo: precompute for ray

    int kz = max_dim(dir);
    int kx = kz + 1;
    if (kx == 3) kx = 0;
    int ky = kx + 1;
    if (ky == 3) ky = 0;

    if (dir[kz] < 0.0f) {
        float tmp = kx;
        kx = ky;
        ky = tmp;
    }

    float Sx = dir[kx] / dir[kz];
    float Sy = dir[ky] / dir[kz];
    float Sz = 1.0f / dir[kz];

    // todo: end precompute

    const Vector3F A = a - p;
    const Vector3F B = b - p;
    const Vector3F C = c - p;

    const float Ax = A[kx] - Sx * A[kz];
    const float Ay = A[ky] - Sy * A[kz];
    const float Bx = B[kx] - Sx * B[kz];
    const float By = B[ky] - Sy * B[kz];
    const float Cx = C[kx] - Sx * C[kz];
    const float Cy = C[ky] - Sy * C[kz];

    float U = diff_product(Cx, By, Cy, Bx);
    float V = diff_product(Ax, Cy, Ay, Cx);
    float W = diff_product(Bx, Ay, By, Ax);

    if (U == 0.0f || V == 0.0f || W == 0.0f) {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) {
        return false;
    }

    float det = U + V + W;

    if (det == 0.0f) {
        return false;
    }

    const float Az = Sz * A[kz];
    const float Bz = Sz * B[kz];
    const float Cz = Sz * C[kz];
    const float T = U * Az + V * Bz + W * Cz;

    int det_sign = sign_mask(det);
    if (xorf(T, det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
    {
        return false;
    }

    const float rcpDet = 1.0f / det;
    u = U * rcpDet;
    v = V * rcpDet;
    t = T * rcpDet;
    sign = det;

    // optionally write out normal (todo: this branch is a performance concern, should probably remove)
    if (normal) {
        const Vector3F ab = b - a;
        const Vector3F ac = c - a;

        // calculate normal
        *normal = ab.cross(ac);
    }

    return true;
}

// Möller's method
#include "intersect_tri.h"

CUDA_CALLABLE inline int intersect_tri_tri(
    Vector3F &v0, Vector3F &v1, Vector3F &v2,
    Vector3F &u0, Vector3F &u1, Vector3F &u2) {
    return NoDivTriTriIsect(&v0[0], &v1[0], &v2[0], &u0[0], &u1[0], &u2[0]);
}

}// namespace vox
