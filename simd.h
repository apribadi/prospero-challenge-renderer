#pragma once

#include <arm_neon.h>

// -------- MNEMONICS --------
//
// v - [v]ector
// x - 128-bit [x]mm
// y - 256-bit [y]mm
// z - 512-bit [z]mm
// b - 8-bit [b]yte
// h - 16-bit [h]alf
// s - 32-bit [s]ingle
// d - 64-bit [d]ouble
// u - [u]nsigned integer
// i - signed [i]nteger
// f - [f]loat
//
// e.g.
//
// vzsf - 512-bit vector of (16) 32-bit floats

typedef uint8x16_t    vxbu;
typedef float32x4_t   vxsf;
typedef uint32x4_t    vxsu;

typedef uint8x16x2_t  vybu;
typedef uint16x8x2_t  vyhu;

typedef float32x4x4_t vzsf;
typedef uint32x4x4_t  vzsu;

static inline vxbu vxbu_load(uint8_t p[16]) {
  return vld1q_u8(p);
}

static inline void vxbu_store(uint8_t p[16], vxbu x) {
  vst1q_u8(p, x);
}

static inline vxbu vxbu_dup(uint8_t x) {
  return vdupq_n_u8(x);
}

static inline vxbu vxbu_add(vxbu x, vxbu y) {
  return vaddq_u8(x, y);
}

static inline vxbu vxbu_sub(vxbu x, vxbu y) {
  return vsubq_u8(x, y);
}

static inline vxbu vxbu_test(vxbu x, vxbu y) {
  return vtstq_u8(x, y);
}

static inline vxbu vxbu_and(vxbu x, vxbu y) {
  return vandq_u8(x, y);
}

static inline vxbu vxbu_or(vxbu x, vxbu y) {
  return vorrq_u8(x, y);
}

static inline vyhu vxbu_vyhu_movemask(vxbu x) {
  return (uint16x8x2_t) {{
    vzip1q_u8(x, x),
    vzip2q_u8(x, x),
  }};
}

static inline vxsf vxsf_load(float p[4]) {
  return vld1q_f32(p);
}

static inline vxsf vxsf_dup(float x) {
  return vdupq_n_f32(x);
}

static inline float vxsf_get(vxsf x, size_t i) {
  return x[i];
}

static inline vxsf vxsf_add_n(vxsf x, float y) {
  return vaddq_f32(x, vdupq_n_f32(y));
}

static inline vxsf vxsf_mul_n(vxsf x, float y) {
  return vmulq_n_f32(x, y);
}

static inline vxsf vxsf_fma(vxsf x, vxsf y, vxsf z) {
  return vfmaq_f32(z, x, y);
}

static inline vxsf vxsf_fma_n(vxsf x, float y, vxsf z) {
  return vfmaq_n_f32(z, x, y);
}

static inline vxsf vxsf_select(vxsu p, vxsf x, vxsf y) {
  return vbslq_f32(p, x, y);
}

static inline vxsu vxsu_dup(uint32_t x) {
  return vdupq_n_u32(x);
}

static inline vybu vybu_load(uint8_t p[32]) {
  return vld1q_u8_x2(p);
}

static inline void vybu_store(uint8_t p[32], vybu x) {
  vst1q_u8_x2(p, x);
}

static inline vybu vybu_dup(uint8_t x) {
  return (uint8x16x2_t) {{ vdupq_n_u8(x), vdupq_n_u8(x) }};
}

static inline vybu vybu_and(vybu x, vybu y) {
  return (uint8x16x2_t) {{
    vandq_u8(x.val[0], y.val[0]),
    vandq_u8(x.val[1], y.val[1]),
  }};
}

static inline vybu vybu_or(vybu x, vybu y) {
  return (uint8x16x2_t) {{
    vorrq_u8(x.val[0], y.val[0]),
    vorrq_u8(x.val[1], y.val[1]),
  }};
}

static inline vyhu vyhu_load(uint16_t p[16]) {
  return vld1q_u16_x2(p);
}

static inline void vyhu_store(uint16_t p[16], vyhu x) {
  vst1q_u16_x2(p, x);
}

static inline vyhu vyhu_dup(uint16_t x) {
  return (uint16x8x2_t) {{
    vdupq_n_u16(x),
    vdupq_n_u16(x),
  }};
}

static inline vyhu vyhu_select(vyhu p, vyhu x, vyhu y) {
  return (uint16x8x2_t) {{
    vbslq_u16(p.val[0], x.val[0], y.val[0]),
    vbslq_u16(p.val[1], x.val[1], y.val[1]),
  }};
}

static inline vzsf vzsf_load(float p[16]) {
  return vld1q_f32_x4(p);
}

static inline vzsf vzsf_from_vxsf_x4(vxsf x0, vxsf x1, vxsf x2, vxsf x3) {
  return (float32x4x4_t) {{ x0, x1, x2, x3 }};
}

static inline vzsf vzsf_select(vzsu p, vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vbslq_f32(p.val[0], x.val[0], y.val[0]),
    vbslq_f32(p.val[1], x.val[1], y.val[1]),
    vbslq_f32(p.val[2], x.val[2], y.val[2]),
    vbslq_f32(p.val[3], x.val[3], y.val[3]),
  }};
}

static inline vzsf vzsf_dup(float x) {
  return (float32x4x4_t) {{ vdupq_n_f32(x), vdupq_n_f32(x), vdupq_n_f32(x), vdupq_n_f32(x) }};
}

static inline vzsf vzsf_and(vzsf x, vzsu y) {
  return (float32x4x4_t) {{
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[0]), y.val[0])),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[1]), y.val[1])),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[2]), y.val[2])),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[3]), y.val[3])),
  }};
}

static inline vzsf vzsf_or(vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3]))),
  }};
}

static inline vzsf vzsf_xor(vzsf x, vzsu y) {
  return (float32x4x4_t) {{
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x.val[0]), y.val[0])),
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x.val[1]), y.val[1])),
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x.val[2]), y.val[2])),
    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x.val[3]), y.val[3])),
  }};
}

static inline vzsf vzsf_neg(vzsf x) {
  return (float32x4x4_t) {{
    vnegq_f32(x.val[0]),
    vnegq_f32(x.val[1]),
    vnegq_f32(x.val[2]),
    vnegq_f32(x.val[3]),
  }};
}

static inline vzsf vzsf_add(vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vaddq_f32(x.val[0], y.val[0]),
    vaddq_f32(x.val[1], y.val[1]),
    vaddq_f32(x.val[2], y.val[2]),
    vaddq_f32(x.val[3], y.val[3]),
  }};
}

static inline vzsf vzsf_add_n(vzsf x, float y) {
  return (float32x4x4_t) {{
    vaddq_f32(x.val[0], vdupq_n_f32(y)),
    vaddq_f32(x.val[1], vdupq_n_f32(y)),
    vaddq_f32(x.val[2], vdupq_n_f32(y)),
    vaddq_f32(x.val[3], vdupq_n_f32(y)),
  }};
}

static inline vzsf vzsf_mul(vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vmulq_f32(x.val[0], y.val[0]),
    vmulq_f32(x.val[1], y.val[1]),
    vmulq_f32(x.val[2], y.val[2]),
    vmulq_f32(x.val[3], y.val[3]),
  }};
}

static inline vzsf vzsf_mul_n(vzsf x, float y) {
  return (float32x4x4_t) {{
    vmulq_n_f32(x.val[0], y),
    vmulq_n_f32(x.val[1], y),
    vmulq_n_f32(x.val[2], y),
    vmulq_n_f32(x.val[3], y),
  }};
}

static inline vzsf vzsf_fma(vzsf x, vzsf y, vzsf z) {
  return (float32x4x4_t) {{
    vfmaq_f32(z.val[0], x.val[0], y.val[0]),
    vfmaq_f32(z.val[1], x.val[1], y.val[1]),
    vfmaq_f32(z.val[2], x.val[2], y.val[2]),
    vfmaq_f32(z.val[3], x.val[3], y.val[3]),
  }};
}

static inline vzsf vzsf_fma_n(vzsf x, float y, vzsf z) {
  return (float32x4x4_t) {{
    vfmaq_n_f32(z.val[0], x.val[0], y),
    vfmaq_n_f32(z.val[1], x.val[1], y),
    vfmaq_n_f32(z.val[2], x.val[2], y),
    vfmaq_n_f32(z.val[3], x.val[3], y),
  }};
}

static inline vzsu vzsf_le(vzsf x, vzsf y) {
  return (uint32x4x4_t) {{
    vcleq_f32(x.val[0], y.val[0]),
    vcleq_f32(x.val[1], y.val[1]),
    vcleq_f32(x.val[2], y.val[2]),
    vcleq_f32(x.val[3], y.val[3]),
  }};
}

static inline vzsu vzsf_lt(vzsf x, vzsf y) {
  return (uint32x4x4_t) {{
    vcltq_f32(x.val[0], y.val[0]),
    vcltq_f32(x.val[1], y.val[1]),
    vcltq_f32(x.val[2], y.val[2]),
    vcltq_f32(x.val[3], y.val[3])
  }};
}

static inline vzsf vzsf_max(vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vmaxq_f32(x.val[0], y.val[0]),
    vmaxq_f32(x.val[1], y.val[1]),
    vmaxq_f32(x.val[2], y.val[2]),
    vmaxq_f32(x.val[3], y.val[3]),
  }};
}

static inline vzsu vzsu_dup(uint32_t x) {
  return (uint32x4x4_t) {{
    vdupq_n_u32(x),
    vdupq_n_u32(x),
    vdupq_n_u32(x),
    vdupq_n_u32(x),
  }};
}

static inline vxbu vzsu_vxbu_movemask(vzsu x) {
  uint16x8_t a = vuzp1q_u16(vreinterpretq_u16_u32(x.val[0]), vreinterpretq_u16_u32(x.val[1]));
  uint16x8_t b = vuzp1q_u16(vreinterpretq_u16_u32(x.val[2]), vreinterpretq_u16_u32(x.val[3]));
  return vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
}
