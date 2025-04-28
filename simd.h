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
typedef uint8x16x4_t  vzbu;
typedef uint16x8x2_t  vyhu;
typedef float32x4_t   vxsf;
typedef float32x4x4_t vzsf;
typedef uint32x4x4_t  vzsu;

static inline vxsf vxsf_load(float p[4]) {
  return vld1q_f32(p);
}

static inline vxbu vxbu_load(uint8_t p[16]) {
  return vld1q_u8(p);
}

static inline void vxbu_store(uint8_t p[16], vxbu x) {
  vst1q_u8(p, x);
}

static inline vxbu vxbu_dup(uint8_t x) {
  return vdupq_n_u8(x);
}

static inline float vxsf_get(vxsf x, size_t i) {
  return x[i];
}

static inline vxsf vxsf_dup(float x) {
  return vdupq_n_f32(x);
}

static inline vxsf vxsf_add(vxsf x, vxsf y) {
  return vaddq_f32(x, y);
}

static inline vxsf vxsf_mul(vxsf x, vxsf y) {
  return vmulq_f32(x, y);
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
    vzip2q_u8(x, x)
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
    vdupq_n_u16(x)
  }};
}

static inline vyhu vyhu_bitselect(vyhu p, vyhu x, vyhu y) {
  return (uint16x8x2_t) {{
    vbslq_u16(p.val[0], x.val[0], y.val[0]),
    vbslq_u16(p.val[1], x.val[1], y.val[1])
  }};
}

static const vzsf VZSF_ZERO = {{
  { 0.0f, 0.0f, 0.0f, 0.0f },
  { 0.0f, 0.0f, 0.0f, 0.0f },
  { 0.0f, 0.0f, 0.0f, 0.0f },
  { 0.0f, 0.0f, 0.0f, 0.0f }
}};

static inline vzsf vzsf_load(float p[16]) {
  return vld1q_f32_x4(p);
}

static inline void vzsf_store(float p[16], vzsf x) {
  vst1q_f32_x4(p, x);
}

static inline vzbu vzbu_load(uint8_t p[64]) {
  return vld1q_u8_x4(p);
}

static inline void vzbu_store(uint8_t p[64], vzbu x) {
  vst1q_u8_x4(p, x);
}

static inline vzsf vzsf_from_vxsf_x4(vxsf x0, vxsf x1, vxsf x2, vxsf x3) {
  return (float32x4x4_t) {{ x0, x1, x2, x3 }};
}

static inline vzbu vzbu_from_vxbu_x4(vxbu x0, vxbu x1, vxbu x2, vxbu x3) {
  return (uint8x16x4_t) {{ x0, x1, x2, x3 }};
}

static inline vzsf vzsf_dup(float x) {
  return (float32x4x4_t) {{ vdupq_n_f32(x), vdupq_n_f32(x), vdupq_n_f32(x), vdupq_n_f32(x) }};
}

static inline vzbu vzbu_dup(uint8_t x) {
  return (uint8x16x4_t) {{ vdupq_n_u8(x), vdupq_n_u8(x), vdupq_n_u8(x), vdupq_n_u8(x) }};
}

static inline vxbu vzsu_vxbu_movemask(vzsu x) {
  uint16x8_t a = vuzp1q_u16(vreinterpretq_u16_u32(x.val[0]), vreinterpretq_u16_u32(x.val[1]));
  uint16x8_t b = vuzp1q_u16(vreinterpretq_u16_u32(x.val[2]), vreinterpretq_u16_u32(x.val[3]));
  return vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
}

static inline vzsf vzsf_and(vzsf x, vzsu y) {
  return (float32x4x4_t) {{
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[0]), y.val[0])),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[1]), y.val[1])),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[2]), y.val[2])),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[3]), y.val[3]))
  }};
}

static inline vzsf vzsf_or(vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3])))
  }};
}

static inline vzbu vzbu_or(vzbu x, vzbu y) {
  return (uint8x16x4_t) {{
    vorrq_u8(x.val[0], y.val[0]),
    vorrq_u8(x.val[1], y.val[1]),
    vorrq_u8(x.val[2], y.val[2]),
    vorrq_u8(x.val[3], y.val[3])
  }};
}

static inline vzbu vzbu_and(vzbu x, vzbu y) {
  return (uint8x16x4_t) {{
    vandq_u8(x.val[0], y.val[0]),
    vandq_u8(x.val[1], y.val[1]),
    vandq_u8(x.val[2], y.val[2]),
    vandq_u8(x.val[3], y.val[3])
  }};
}

static inline vzsf vzsf_add(vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vaddq_f32(x.val[0], y.val[0]),
    vaddq_f32(x.val[1], y.val[1]),
    vaddq_f32(x.val[2], y.val[2]),
    vaddq_f32(x.val[3], y.val[3])
  }};
}

static inline vzsf vzsf_mul(vzsf x, vzsf y) {
  return (float32x4x4_t) {{
    vmulq_f32(x.val[0], y.val[0]),
    vmulq_f32(x.val[1], y.val[1]),
    vmulq_f32(x.val[2], y.val[2]),
    vmulq_f32(x.val[3], y.val[3])
  }};
}

static inline vzsu vzsf_le(vzsf x, vzsf y) {
  return (uint32x4x4_t) {{
    vcleq_f32(x.val[0], y.val[0]),
    vcleq_f32(x.val[1], y.val[1]),
    vcleq_f32(x.val[2], y.val[2]),
    vcleq_f32(x.val[3], y.val[3])
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
    vmaxq_f32(x.val[3], y.val[3])
  }};
}
