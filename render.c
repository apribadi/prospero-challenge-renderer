#include <arm_neon.h>
#include <assert.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simd.h"
#include "render.h"

#define MAX_SPECIALIZATION_LEVELS 3

// -------- PRIORITY QUEUE (AND SET) --------

typedef struct {
  size_t capacity;
  size_t size;
  uint16_t * data;
  bool * included;
} PQ;

static void pq_init(PQ * t, size_t capacity) {
  t->capacity = capacity;
  t->size = 0;
  t->data = malloc(capacity * sizeof(uint16_t));
  t->included = calloc(capacity, sizeof(bool));
}

static void pq_drop(PQ * t) {
  free(t->data);
  free(t->included);
}

static inline bool pq_is_empty(PQ * t) {
  return t->size == 0;
}

static void pq_insert(PQ * t, uint16_t x) {
  uint16_t * data = t->data;
  bool * included = t->included;

  if (included[x]) return;

  included[x] = true;

  size_t i = t->size ++;

  for (;;) {
    if (i == 0) break;
    size_t k = i - 1 >> 1;
    uint16_t u = data[k];
    if (x <= u) break;
    data[i] = u;
    i = k;
  }

  data[i] = x;
}

static uint16_t pq_pop(PQ * t) {
  uint16_t * data = t->data;
  bool * included = t->included;

  size_t n = -- t->size;
  size_t i = 0;
  uint16_t r = data[0];
  uint16_t x = data[n];

  included[r] = false;

  for (;;) {
    size_t a = 2 * i + 1;
    size_t b = 2 * i + 2;

    if (b < n) {
      uint16_t u = data[a];
      uint16_t v = data[b];
      if (u <= x && v <= x) break;
      bool p = u >= v;
      data[i] = p ? u : v;
      i = p ? a : b;
      continue;
    }

    if (a < n) {
      uint16_t u = data[a];
      if (u <= x) break;
      data[i] = u;
      i = a;
      continue;
    }

    break;
  }

  data[i] = x;

  return r;
}

// -------- BUFFER OF U16 --------

typedef struct { size_t capacity; uint16_t * data; } BufU16;

static void buf_u16_init(BufU16 * t, size_t capacity) {
  t->capacity = capacity;
  t->data = malloc(capacity * sizeof(uint16_t));
}

static void buf_u16_drop(BufU16 * t) {
  free(t->data);
}

// -------- BUFFER OF INSTRUCTIONS --------

typedef struct { size_t capacity; Inst * data; } BufInst;

static void buf_inst_init(BufInst * t, size_t capacity) {
  t->capacity = capacity;
  t->data = malloc(capacity * sizeof(Inst));
}

static void buf_inst_drop(BufInst * t) {
  free(t->data);
}

// -------- SLOT AND ENVIRONMENT TYPES FOR INTERPRETER LOOPS --------

typedef union {
  struct {
    float min[16];
    float max[16];
  } floats;
  struct {
    uint8_t is_f[16];
    uint8_t is_t[16];
    uint16_t link[16];
  } bools;
} Slot1;

typedef struct {
  size_t capacity;
  float x[5];
  float y[5];
  Slot1 * slots;
} Env1;

static void env1_init(Env1 * t, size_t capacity) {
  t->capacity = capacity;
  t->slots = malloc(capacity * sizeof(Slot1));
}

static void env1_drop(Env1 * t) {
  free(t->slots);
}

typedef union {
  float floats[256];
  uint8_t bools[256];
} Slot2;

typedef struct {
  size_t capacity;
  float x[16];
  float y[16];
  Slot2 * slots;
} Env2;

static void env2_init(Env2 * t, size_t capacity) {
  t->capacity = capacity;
  t->slots = malloc(capacity * sizeof(Slot2));
}

static void env2_drop(Env2 * t) {
  free(t->slots);
}

// -------- SCRATCH DYNAMICALLY SIZED DATA STRUCTURES --------

typedef struct {
  BufU16 rev;
  BufU16 map;
  PQ pq;
  Env1 env1;
  Env2 env2;
  BufInst code[MAX_SPECIALIZATION_LEVELS][16];
} Scratch;

static void scratch_init(Scratch * t) {
  // We're assuming that a zero size_t and a null pointer are both just zero
  // bytes.
  memset(t, 0, sizeof(Scratch));
}

static void scratch_drop(Scratch * t) {
  buf_u16_drop(&t->rev);
  buf_u16_drop(&t->map);
  pq_drop(&t->pq);
  env1_drop(&t->env1);
  env2_drop(&t->env2);
  for (size_t i = 0; i < MAX_SPECIALIZATION_LEVELS; i ++) {
    for (size_t j = 0; j < 16; j ++) {
      buf_inst_drop(&t->code[i][j]);
    }
  }
}

static uint16_t * scratch_rev(Scratch * t, size_t capacity) {
  if (capacity <= t->rev.capacity) return t->rev.data;
  buf_u16_drop(&t->rev);
  buf_u16_init(&t->rev, capacity);
  return t->rev.data;
}

static uint16_t * scratch_map(Scratch * t, size_t capacity) {
  if (capacity <= t->map.capacity) return t->map.data;
  buf_u16_drop(&t->map);
  buf_u16_init(&t->map, capacity);
  return t->map.data;
}

static PQ * scratch_pq(Scratch * t, size_t capacity) {
  if (capacity <= t->pq.capacity) return &t->pq;
  pq_drop(&t->pq);
  pq_init(&t->pq, capacity);
  return &t->pq;
}

static Env1 * scratch_env1(Scratch * t, size_t capacity) {
  if (capacity <= t->env1.capacity) return &t->env1;
  env1_drop(&t->env1);
  env1_init(&t->env1, capacity);
  return &t->env1;
}

static Env2 * scratch_env2(Scratch * t, size_t capacity) {
  if (capacity <= t->env2.capacity) return &t->env2;
  env2_drop(&t->env2);
  env2_init(&t->env2, capacity);
  return &t->env2;
}

static Inst * scratch_code(Scratch * t, size_t i, size_t j, size_t capacity) {
  BufInst * code = &t->code[i][j];
  if (capacity <= code->capacity) return code->data;
  buf_inst_drop(code);
  buf_inst_init(code, capacity);
  return code->data;
}

// -------- FORWARD ANALYSIS FOR 16 SUBREGIONS --------

typedef struct Tbl1_ {
  size_t (* ops[8])(Inst *, Env1 *, Slot1 *, struct Tbl1_ *, size_t, Inst);
} Tbl1;

static inline size_t op1_dispatch(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc) {
  // cp - code pointer
  // ep - environment pointer
  // sp - stack pointer
  // tp - table pointer
  // pc - program counter
  Inst inst = cp[pc];
  return tp->ops[inst.op](cp, ep, sp, tp, pc, inst);
}

static size_t op1_affine(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  float a = inst.affine.a;
  float b = inst.affine.b;
  float c = inst.affine.c;
  vxsf xmin = vxsf_load(&ep->x[0]);
  vxsf xmax = vxsf_load(&ep->x[1]);
  vxsf ymin = vxsf_load(&ep->y[1]);
  vxsf ymax = vxsf_load(&ep->y[0]);
  vxsf umin = vxsf_add(vxsf_mul(a < 0.0f ? xmax : xmin, vxsf_dup(a)), vxsf_dup(c));
  vxsf umax = vxsf_add(vxsf_mul(a < 0.0f ? xmin : xmax, vxsf_dup(a)), vxsf_dup(c));
  vxsf vmin = vxsf_mul(b < 0.0f ? ymax : ymin, vxsf_dup(b));
  vxsf vmax = vxsf_mul(b < 0.0f ? ymin : ymax, vxsf_dup(b));
  vzsf wmin =
    vzsf_from_vxsf_x4(
      vxsf_add(umin, vxsf_dup(vxsf_get(vmin, 0))),
      vxsf_add(umin, vxsf_dup(vxsf_get(vmin, 1))),
      vxsf_add(umin, vxsf_dup(vxsf_get(vmin, 2))),
      vxsf_add(umin, vxsf_dup(vxsf_get(vmin, 3))));
  vzsf_store(sp[pc].floats.min, wmin);
  vzsf wmax =
    vzsf_from_vxsf_x4(
      vxsf_add(umax, vxsf_dup(vxsf_get(vmax, 0))),
      vxsf_add(umax, vxsf_dup(vxsf_get(vmax, 1))),
      vxsf_add(umax, vxsf_dup(vxsf_get(vmax, 2))),
      vxsf_add(umax, vxsf_dup(vxsf_get(vmax, 3))));
  vzsf_store(sp[pc].floats.max, wmax);
  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_hypot2(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  vzsf xmin = vzsf_load(sp[inst.hypot2.x].floats.min);
  vzsf xmax = vzsf_load(sp[inst.hypot2.x].floats.max);
  vzsf ymin = vzsf_load(sp[inst.hypot2.y].floats.min);
  vzsf ymax = vzsf_load(sp[inst.hypot2.y].floats.max);
  vzsf umin =
    vzsf_or(
      vzsf_and(vzsf_mul(xmin, xmin), vzsf_le(VZSF_ZERO, xmin)),
      vzsf_and(vzsf_mul(xmax, xmax), vzsf_le(xmax, VZSF_ZERO)));
  vzsf vmin =
    vzsf_or(
      vzsf_and(vzsf_mul(ymin, ymin), vzsf_le(VZSF_ZERO, ymin)),
      vzsf_and(vzsf_mul(ymax, ymax), vzsf_le(ymax, VZSF_ZERO)));
  vzsf_store(sp[pc].floats.min, vzsf_add(umin, vmin));
  vzsf umax = vzsf_max(vzsf_mul(xmin, xmin), vzsf_mul(xmax, xmax));
  vzsf vmax = vzsf_max(vzsf_mul(ymin, ymin), vzsf_mul(ymax, ymax));
  vzsf_store(sp[pc].floats.max, vzsf_add(umax, vmax));
  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_le_const(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  vzsf xmin = vzsf_load(sp[inst.le_const.x].floats.min);
  vzsf xmax = vzsf_load(sp[inst.le_const.x].floats.max);
  vzsf a = vzsf_dup(inst.le_const.a);
  vxbu_store(sp[pc].bools.is_f, vzsu_vxbu_movemask(vzsf_lt(a, xmin)));
  vxbu_store(sp[pc].bools.is_t, vzsu_vxbu_movemask(vzsf_le(xmax, a)));
  vyhu_store(sp[pc].bools.link, vyhu_dup((uint16_t) pc));
  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_ge_const(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  vzsf xmin = vzsf_load(sp[inst.ge_const.x].floats.min);
  vzsf xmax = vzsf_load(sp[inst.ge_const.x].floats.max);
  vzsf a = vzsf_dup(inst.ge_const.a);
  vxbu_store(sp[pc].bools.is_f, vzsu_vxbu_movemask(vzsf_lt(xmax, a)));
  vxbu_store(sp[pc].bools.is_t, vzsu_vxbu_movemask(vzsf_le(a, xmin)));
  vyhu_store(sp[pc].bools.link, vyhu_dup((uint16_t) pc));
  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_and(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  vxbu x_is_f = vxbu_load(sp[inst.and.x].bools.is_f);
  vxbu x_is_t = vxbu_load(sp[inst.and.x].bools.is_t);
  vyhu x_link = vyhu_load(sp[inst.and.x].bools.link);
  vxbu y_is_f = vxbu_load(sp[inst.and.y].bools.is_f);
  vxbu y_is_t = vxbu_load(sp[inst.and.y].bools.is_t);
  vyhu y_link = vyhu_load(sp[inst.and.y].bools.link);
  vxbu_store(sp[pc].bools.is_f, vxbu_or(x_is_f, y_is_f));
  vxbu_store(sp[pc].bools.is_t, vxbu_and(x_is_t, y_is_t));
  vyhu link = vyhu_dup((uint16_t) pc);
  link = vyhu_bitselect(vxbu_vyhu_movemask(x_is_t), y_link, link);
  link = vyhu_bitselect(vxbu_vyhu_movemask(y_is_t), x_link, link);
  vyhu_store(sp[pc].bools.link, link);
  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_or(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  vxbu x_is_f = vxbu_load(sp[inst.or.x].bools.is_f);
  vxbu x_is_t = vxbu_load(sp[inst.or.x].bools.is_t);
  vyhu x_link = vyhu_load(sp[inst.or.x].bools.link);
  vxbu y_is_f = vxbu_load(sp[inst.or.y].bools.is_f);
  vxbu y_is_t = vxbu_load(sp[inst.or.y].bools.is_t);
  vyhu y_link = vyhu_load(sp[inst.or.y].bools.link);
  vxbu_store(sp[pc].bools.is_f, vxbu_and(x_is_f, y_is_f));
  vxbu_store(sp[pc].bools.is_t, vxbu_or(x_is_t, y_is_t));
  vyhu link = vyhu_dup((uint16_t) pc);
  link = vyhu_bitselect(vxbu_vyhu_movemask(x_is_f), y_link, link);
  link = vyhu_bitselect(vxbu_vyhu_movemask(y_is_f), x_link, link);
  vyhu_store(sp[pc].bools.link, link);
  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_ret(Inst *, Env1 *, Slot1 *, Tbl1 *, size_t, Inst inst) {
  return inst.ret.x;
}

static size_t op1_ret_const(Inst *, Env1 *, Slot1 *, Tbl1 *, size_t pc, Inst) {
  return pc;
}

static void analyze(Inst * cp, Env1 * ep, Slot1 * sp) {
  static Tbl1 TBL1 = {{
    op1_affine,
    op1_hypot2,
    op1_le_const,
    op1_ge_const,
    op1_and,
    op1_or,
    op1_ret,
    op1_ret_const
  }};

  (void) op1_dispatch(cp, ep, sp, &TBL1, 0);
}

// -------- RASTERIZE A 256 PIXEL REGION --------

typedef struct Tbl2_ {
  size_t (* ops[8])(Inst *, Env2 *, Slot2 *, struct Tbl2_ *, size_t, Inst);
} Tbl2;

static inline size_t op2_dispatch(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc) {
  Inst inst = cp[pc];
  return tp->ops[inst.op](cp, ep, sp, tp, pc, inst);
}

static size_t op2_affine(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  vzsf x = vzsf_load(ep->x);
  vzsf u = vzsf_add(vzsf_mul(x, vzsf_dup(inst.affine.a)), vzsf_dup(inst.affine.c));
  for (size_t h = 0; h < 4; h ++) {
    vxsf y = vxsf_load(&ep->y[4 * h]);
    vxsf v = vxsf_mul(y, vxsf_dup(inst.affine.b));
    vzsf_store(&sp[pc].floats[64 * h + 16 * 0], vzsf_add(u, vzsf_dup(vxsf_get(v, 0))));
    vzsf_store(&sp[pc].floats[64 * h + 16 * 1], vzsf_add(u, vzsf_dup(vxsf_get(v, 1))));
    vzsf_store(&sp[pc].floats[64 * h + 16 * 2], vzsf_add(u, vzsf_dup(vxsf_get(v, 2))));
    vzsf_store(&sp[pc].floats[64 * h + 16 * 3], vzsf_add(u, vzsf_dup(vxsf_get(v, 3))));
  }
  return op2_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op2_hypot2(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  for (size_t k = 0; k < 16; k ++) {
    vzsf x = vzsf_load(&sp[inst.hypot2.x].floats[16 * k]);
    vzsf y = vzsf_load(&sp[inst.hypot2.y].floats[16 * k]);
    vzsf_store(&sp[pc].floats[16 * k], vzsf_add(vzsf_mul(x, x), vzsf_mul(y, y)));
  }
  return op2_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op2_le_const(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  for (size_t h = 0; h < 4; h ++) {
    vzsf x0 = vzsf_load(&sp[inst.le_const.x].floats[64 * h + 16 * 0]);
    vzsf x1 = vzsf_load(&sp[inst.le_const.x].floats[64 * h + 16 * 1]);
    vzsf x2 = vzsf_load(&sp[inst.le_const.x].floats[64 * h + 16 * 2]);
    vzsf x3 = vzsf_load(&sp[inst.le_const.x].floats[64 * h + 16 * 3]);
    vxbu u0 = vzsu_vxbu_movemask(vzsf_le(x0, vzsf_dup(inst.le_const.a)));
    vxbu u1 = vzsu_vxbu_movemask(vzsf_le(x1, vzsf_dup(inst.le_const.a)));
    vxbu u2 = vzsu_vxbu_movemask(vzsf_le(x2, vzsf_dup(inst.le_const.a)));
    vxbu u3 = vzsu_vxbu_movemask(vzsf_le(x3, vzsf_dup(inst.le_const.a)));
    vzbu_store(&sp[pc].bools[64 * h], vzbu_from_vxbu_x4(u0, u1, u2, u3));
  }
  return op2_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op2_ge_const(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  for (size_t h = 0; h < 4; h ++) {
    vzsf x0 = vzsf_load(&sp[inst.ge_const.x].floats[64 * h + 16 * 0]);
    vzsf x1 = vzsf_load(&sp[inst.ge_const.x].floats[64 * h + 16 * 1]);
    vzsf x2 = vzsf_load(&sp[inst.ge_const.x].floats[64 * h + 16 * 2]);
    vzsf x3 = vzsf_load(&sp[inst.ge_const.x].floats[64 * h + 16 * 3]);
    vxbu u0 = vzsu_vxbu_movemask(vzsf_le(vzsf_dup(inst.ge_const.a), x0));
    vxbu u1 = vzsu_vxbu_movemask(vzsf_le(vzsf_dup(inst.ge_const.a), x1));
    vxbu u2 = vzsu_vxbu_movemask(vzsf_le(vzsf_dup(inst.ge_const.a), x2));
    vxbu u3 = vzsu_vxbu_movemask(vzsf_le(vzsf_dup(inst.ge_const.a), x3));
    vzbu_store(&sp[pc].bools[64 * h], vzbu_from_vxbu_x4(u0, u1, u2, u3));
  }
  return op2_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op2_and(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  for (size_t h = 0; h < 4; h ++) {
    vzbu x = vzbu_load(&sp[inst.and.x].bools[64 * h]);
    vzbu y = vzbu_load(&sp[inst.and.y].bools[64 * h]);
    vzbu_store(&sp[pc].bools[64 * h], vzbu_and(x, y));
  }
  return op2_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op2_or(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  for (size_t h = 0; h < 4; h ++) {
    vzbu x = vzbu_load(&sp[inst.or.x].bools[64 * h]);
    vzbu y = vzbu_load(&sp[inst.or.y].bools[64 * h]);
    vzbu_store(&sp[pc].bools[64 * h], vzbu_or(x, y));
  }
  return op2_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op2_ret(Inst *, Env2 *, Slot2 *, Tbl2 *, size_t, Inst inst) {
  return inst.ret.x;
}

static size_t op2_ret_const(Inst *, Env2 *, Slot2 * sp, Tbl2 *, size_t pc, Inst inst) {
  for (size_t h = 0; h < 4; h ++) {
    vzbu_store(&sp[pc].bools[64 * h], vzbu_dup(inst.ret_const.a ? 255 : 0));
  }
  return pc;
}

static void rasterize(
    Scratch * scratch,
    size_t code_len,
    Inst code[code_len],
    float xmin,
    float xlen,
    float ymax,
    float ylen,
    size_t stride,
    uint8_t * tile
  )
{
  static Tbl2 TBL2 = {{
    op2_affine,
    op2_hypot2,
    op2_le_const,
    op2_ge_const,
    op2_and,
    op2_or,
    op2_ret,
    op2_ret_const
  }};

  Env2 * ep = scratch_env2(scratch, code_len);
  Slot2 * sp = ep->slots;

  float dx = 0.0625f * xlen;
  float dy = 0.0625f * ylen;

  for (size_t k = 0; k < 16; k ++) {
    ep->x[k] = xmin + 0.5f * dx + dx * (float) k;
    ep->y[k] = ymax - 0.5f * dy - dy * (float) k;
  }

  size_t result = op2_dispatch(code, ep, sp, &TBL2, 0);

  for (size_t k = 0; k < 16; k ++) {
    memcpy(tile + stride * k, &sp[result].bools[16 * k], 16);
  }
}

// -------- SPECIALIZE CODE TO 16 SUBREGIONS --------

static void specialize(
    Scratch * scratch,
    size_t code_len,
    Inst code[code_len],
    float xmin,
    float xlen,
    float ymax,
    float ylen,
    size_t depth,
    size_t out_code_len[16],
    Inst * out_code[16]
  )
{
  Env1 * ep = scratch_env1(scratch, code_len);
  Slot1 * sp = ep->slots;

  for (size_t k = 0; k < 5; k ++) {
    ep->x[k] = xmin + 0.25f * xlen * (float) k;
    ep->y[k] = ymax - 0.25f * ylen * (float) k;
  }

  analyze(code, ep, sp);

  uint16_t * rev = scratch_rev(scratch, code_len);
  uint16_t * map = scratch_map(scratch, code_len);
  PQ * gray = scratch_pq(scratch, code_len);

  for (size_t t = 0; t < 16; t ++) {
    size_t sub_code_len = 0;

    {
      uint16_t k = (uint16_t) (code_len - 1);
      rev[sub_code_len ++] = k;
      Inst inst = code[k];
      if (inst.op == OP_RET) {
        if (! sp[inst.ret.x].bools.is_f[t] && ! sp[inst.ret.x].bools.is_t[t]) {
          pq_insert(gray, sp[inst.ret.x].bools.link[t]);
          pq_insert(gray, sp[inst.ret.x].bools.link[t]);
        }
      }
    }

    while (! pq_is_empty(gray)) {
      uint16_t k = pq_pop(gray);
      rev[sub_code_len ++] = k;
      Inst inst = code[k];

      switch (inst.op) {
      case OP_HYPOT2:
        pq_insert(gray, inst.hypot2.x);
        pq_insert(gray, inst.hypot2.y);
        break;
      case OP_LE_CONST:
        pq_insert(gray, inst.le_const.x);
        break;
      case OP_GE_CONST:
        pq_insert(gray, inst.ge_const.x);
        break;
      case OP_AND:
        pq_insert(gray, sp[inst.and.x].bools.link[t]);
        pq_insert(gray, sp[inst.and.y].bools.link[t]);
        break;
      case OP_OR:
        pq_insert(gray, sp[inst.or.x].bools.link[t]);
        pq_insert(gray, sp[inst.or.y].bools.link[t]);
        break;
      default:
        break;
      }
    }

    Inst * sub_code = scratch_code(scratch, depth, t, sub_code_len);

    out_code[t] = sub_code;
    out_code_len[t] = sub_code_len;

    for (size_t i = 0; i < sub_code_len; i ++) {
      uint16_t k = rev[sub_code_len - 1 - i];

      map[k] = (uint16_t) i;
      Inst inst = code[k];

      switch (inst.op) {
      case OP_HYPOT2:
        inst.hypot2.x = map[inst.hypot2.x];
        inst.hypot2.y = map[inst.hypot2.y];
        break;
      case OP_LE_CONST:
        inst.le_const.x = map[inst.le_const.x];
        break;
      case OP_GE_CONST:
        inst.ge_const.x = map[inst.ge_const.x];
        break;
      case OP_AND:
        inst.and.x = map[sp[inst.and.x].bools.link[t]];
        inst.and.y = map[sp[inst.and.y].bools.link[t]];
        break;
      case OP_OR:
        inst.or.x = map[sp[inst.or.x].bools.link[t]];
        inst.or.y = map[sp[inst.or.y].bools.link[t]];
        break;
      case OP_RET:
        if (sp[inst.ret.x].bools.is_f[t]) {
          inst = (Inst) { OP_RET_CONST, .ret_const = { false } };
        } else if (sp[inst.ret.x].bools.is_t[t]) {
          inst = (Inst) { OP_RET_CONST, .ret_const = { true } };
        } else {
          inst.ret.x = map[sp[inst.ret.x].bools.link[t]];
        }
        break;
      default:
        break;
      }

      sub_code[i] = inst;
    }
  }
}

// -------- RENDER --------

static void render_tile(
    Scratch * scratch,
    size_t code_len,
    Inst code[code_len],
    float xmin,
    float xlen,
    float ymax,
    float ylen,
    size_t depth,
    size_t resolution,
    size_t stride,
    uint8_t * tile
  )
{
  if (code_len == 1 && code[0].op == OP_RET_CONST) {
    uint8_t value = code[0].ret_const.a ? 255 : 0;

    for (size_t i = 0; i < resolution; i ++) {
      for (size_t j = 0; j < resolution; j += 16) {
        vxbu_store(tile + stride * i + j, vxbu_dup(value));
      }
    }

    return;
  }

  if (resolution == 16) {
    rasterize(
        scratch,
        code_len,
        code,
        xmin,
        xlen,
        ymax,
        ylen,
        stride,
        tile
      );

    return;
  }

  if (resolution == 32) {
    for (size_t t = 0; t < 4; t ++) {
      size_t i = t / 2;
      size_t j = t % 2;
      rasterize(
          scratch,
          code_len,
          code,
          xmin + 0.5f * xlen * (float) j,
          0.5f * xlen,
          ymax - 0.5f * ylen * (float) i,
          0.5f * ylen,
          stride,
          tile + resolution / 2 * stride * i + resolution / 2 * j
        );
    }

    return;
  }

  size_t sub_code_len[16];
  Inst * sub_code[16];

  specialize(
      scratch,
      code_len,
      code,
      xmin,
      xlen,
      ymax,
      ylen,
      depth,
      sub_code_len,
      sub_code
    );

  for (size_t t = 0; t < 16; t ++) {
    size_t i = t / 4;
    size_t j = t % 4;

    render_tile(
        scratch,
        sub_code_len[t],
        sub_code[t],
        xmin + 0.25f * xlen * (float) j,
        0.25f * xlen,
        ymax - 0.25f * ylen * (float) i,
        0.25f * ylen,
        depth + 1,
        resolution / 4,
        stride,
        tile + resolution / 4 * stride * i + resolution / 4 * j
      );
  }
}

void render(
    size_t code_len,
    Inst code[code_len],
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    size_t resolution,
    uint8_t image[resolution][resolution]
  )
{
  // cf MAX_SPECIALIZATION_LEVELS
  assert(256 <= resolution && resolution <= 8192);
  assert((resolution & (resolution - 1)) == 0);

#pragma omp parallel for
  for (size_t t = 0; t < 16; t ++) {
    size_t i = t / 4;
    size_t j = t % 4;

    Scratch scratch;
    scratch_init(&scratch);

    render_tile(
        &scratch,
        code_len,
        code,
        xmin + 0.25f * (xmax - xmin) * (float) j,
        0.25f * (xmax - xmin),
        ymax - 0.25f * (ymax - ymin) * (float) i,
        0.25f * (ymax - ymin),
        0,
        resolution / 4,
        resolution,
        &image[resolution / 4 * i][resolution / 4 * j]
      );

    scratch_drop(&scratch);
  }
}
