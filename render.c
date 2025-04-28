#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "simd.h"
#include "render.h"

// -------- ARENA ALLOCATOR --------
//
// a crude arena
//
// - fixed size
// - 8-byte aligned

typedef struct {
  unsigned char * base;
  size_t available;
} Arena;

__attribute__ ((noinline, noreturn))
static void arena_alloc_failed(void) {
  fprintf(stderr, "arena: allocation failed!\n");
  abort();
}

static void arena_init(Arena * arena, size_t capacity) {
  void * base = malloc(capacity);
  if (! base) arena_alloc_failed();
  arena->base = base;
  arena->available = capacity;
}

static void arena_drop(Arena * arena) {
  free(arena->base);
}

static inline void * arena_alloc_array(Arena * arena, size_t count, size_t size) {
  if (count > arena->available / size) arena_alloc_failed();
  size_t n = (arena->available - count * size) & - (size_t) 8;
  arena->available = n;
  return (void *) (arena->base + n);
}

#define ALLOC_ARRAY(arena, count, T) \
  (T *) arena_alloc_array((arena), (count), sizeof(T))

static inline void * arena_alloc_array_zeroed(Arena * arena, size_t count, size_t size) {
  return memset(arena_alloc_array(arena, count, size), 0, count * size);
}

#define ALLOC_ARRAY_ZEROED(arena, count, T) \
  (T *) arena_alloc_array_zeroed((arena), (count), sizeof(T))

// -------- PRIORITY QUEUE (AND SET) --------

typedef struct {
  size_t size;
  uint16_t * data;
  bool * included;
} PQ;

static void pq_init(PQ * t, Arena * arena, size_t capacity) {
  t->size = 0;
  t->data = ALLOC_ARRAY(arena, capacity, uint16_t);
  t->included = ALLOC_ARRAY_ZEROED(arena, capacity, bool);
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
    size_t k = 2 * i + 1;

    if (k + 1 < n) {
      uint16_t u = data[k];
      uint16_t v = data[k + 1];
      if (u <= x && v <= x) break;
      data[i] = u < v ? v : u;
      i = k + (u < v);
      continue;
    }

    if (k < n) {
      uint16_t u = data[k];
      if (u <= x) break;
      data[i] = u;
      i = k;
      continue;
    }

    break;
  }

  data[i] = x;

  return r;
}

// -------- FORWARD ANALYSIS FOR 16 SUBREGIONS --------

typedef struct {
  float x[5];
  float y[5];
} Input1;

typedef struct {
  uint8_t is_f[16];
  uint8_t is_t[16];
  uint16_t link[16];
} Slot1;

typedef struct Table1_ {
  void (* ops[6])(Geometry, Inst *, Input1 *, Slot1 *, struct Table1_ *, size_t, Inst);
} Table1;

#define ARGS1 \
  __attribute__((unused)) Geometry geometry, \
  __attribute__((unused)) Inst * code, \
  __attribute__((unused)) Input1 * input, \
  __attribute__((unused)) Slot1 * slots, \
  __attribute__((unused)) Table1 * table, \
  __attribute__((unused)) size_t pc, \
  __attribute__((unused)) Inst inst

#define DISPATCH1 \
  do { \
    Inst inst = code[pc + 1]; \
    return table->ops[inst.op](geometry, code, input, slots, table, pc + 1, inst); \
  } while (0)

static inline vzsf dup4x(vxsf x) {
  return vzsf_from_vxsf_x4(x, x, x, x);
}

static inline vzsf dup4y(vxsf x) {
  float a = vxsf_get(x, 0);
  float b = vxsf_get(x, 1);
  float c = vxsf_get(x, 2);
  float d = vxsf_get(x, 3);
  return vzsf_from_vxsf_x4(vxsf_dup(a), vxsf_dup(b), vxsf_dup(c), vxsf_dup(d));
}

static inline vzsf min_grid_lin(vxsf xmin, vxsf xmax, vxsf ymin, vxsf ymax, float a, float b) {
  vxsu p = vxsu_dup(a < 0.0f ? UINT32_MAX : 0);
  vxsu q = vxsu_dup(b < 0.0f ? UINT32_MAX : 0);
  vxsf u = vxsf_mul_n(vxsf_select(p, xmax, xmin), a);
  vxsf v = vxsf_mul_n(vxsf_select(q, ymax, ymin), b);
  return vzsf_add(dup4x(u), dup4y(v));
}

static inline vzsf max_grid_lin(vxsf xmin, vxsf xmax, vxsf ymin, vxsf ymax, float a, float b) {
  vxsu p = vxsu_dup(a < 0.0f ? UINT32_MAX : 0);
  vxsu q = vxsu_dup(b < 0.0f ? UINT32_MAX : 0);
  vxsf u = vxsf_mul_n(vxsf_select(p, xmin, xmax), a);
  vxsf v = vxsf_mul_n(vxsf_select(q, ymin, ymax), b);
  return vzsf_add(dup4x(u), dup4y(v));
}

static inline vzsf min_grid_aff(vxsf xmin, vxsf xmax, vxsf ymin, vxsf ymax, float a, float b, float c) {
  vxsu p = vxsu_dup(a < 0.0f ? UINT32_MAX : 0);
  vxsu q = vxsu_dup(b < 0.0f ? UINT32_MAX : 0);
  vxsf u = vxsf_fma_n(vxsf_select(p, xmax, xmin), a, vxsf_dup(c));
  vxsf v = vxsf_mul_n(vxsf_select(q, ymax, ymin), b);
  return vzsf_add(dup4x(u), dup4y(v));
}

static inline vzsf max_grid_aff(vxsf xmin, vxsf xmax, vxsf ymin, vxsf ymax, float a, float b, float c) {
  vxsu p = vxsu_dup(a < 0.0f ? UINT32_MAX : 0);
  vxsu q = vxsu_dup(b < 0.0f ? UINT32_MAX : 0);
  vxsf u = vxsf_fma_n(vxsf_select(p, xmin, xmax), a, vxsf_dup(c));
  vxsf v = vxsf_mul_n(vxsf_select(q, ymin, ymax), b);
  return vzsf_add(dup4x(u), dup4y(v));
}

static inline vzsf min_sq(vzsf xmin, vzsf xmax) {
  return
    vzsf_or(
      vzsf_and(vzsf_mul(xmin, xmin), vzsf_lt(vzsf_dup(0.0f), xmin)),
      vzsf_and(vzsf_mul(xmax, xmax), vzsf_lt(xmax, vzsf_dup(0.0f))));
}

static inline vzsf max_sq(vzsf xmin, vzsf xmax) {
  return vzsf_max(vzsf_mul(xmin, xmin), vzsf_mul(xmax, xmax));
}

static void op1_line(ARGS1) {
  Line l = geometry.line[inst.line.index];
  vxsf xmin = vxsf_load(&input->x[0]);
  vxsf xmax = vxsf_load(&input->x[1]);
  vxsf ymin = vxsf_load(&input->y[1]);
  vxsf ymax = vxsf_load(&input->y[0]);
  vzsf zmin = min_grid_lin(xmin, xmax, ymin, ymax, l.a, l.b);
  vzsf zmax = max_grid_lin(xmin, xmax, ymin, ymax, l.a, l.b);
  vxbu_store(slots[pc].is_f, vzsu_vxbu_movemask(vzsf_lt(vzsf_dup(- l.c), zmin)));
  vxbu_store(slots[pc].is_t, vzsu_vxbu_movemask(vzsf_le(zmax, vzsf_dup(- l.c))));
  vyhu_store(slots[pc].link, vyhu_dup((uint16_t) pc));
  DISPATCH1;
}

static void op1_ellipse(ARGS1) {
  Ellipse e = geometry.ellipse[inst.ellipse.index];
  vxsf xmin = vxsf_load(&input->x[0]);
  vxsf xmax = vxsf_load(&input->x[1]);
  vxsf ymin = vxsf_load(&input->y[1]);
  vxsf ymax = vxsf_load(&input->y[0]);
  vzsf umin = min_grid_aff(xmin, xmax, ymin, ymax, e.a, e.b, e.c);
  vzsf umax = max_grid_aff(xmin, xmax, ymin, ymax, e.a, e.b, e.c);
  vzsf vmin = min_grid_aff(xmin, xmax, ymin, ymax, e.d, e.e, e.f);
  vzsf vmax = max_grid_aff(xmin, xmax, ymin, ymax, e.d, e.e, e.f);
  vzsf wmin = vzsf_add_n(vzsf_add(min_sq(umin, umax), min_sq(vmin, vmax)), -1.0f);
  vzsf wmax = vzsf_add_n(vzsf_add(max_sq(umin, umax), max_sq(vmin, vmax)), -1.0f);
  vzsu p = vzsu_dup(inst.ellipse.outside ? UINT32_MAX : 0);
  vzsf zmin = vzsf_select(p, vzsf_neg(wmax), wmin);
  vzsf zmax = vzsf_select(p, vzsf_neg(wmin), wmax);
  vxbu_store(slots[pc].is_f, vzsu_vxbu_movemask(vzsf_lt(vzsf_dup(0.0f), zmin)));
  vxbu_store(slots[pc].is_t, vzsu_vxbu_movemask(vzsf_le(zmax, vzsf_dup(0.0f))));
  vyhu_store(slots[pc].link, vyhu_dup((uint16_t) pc));
  DISPATCH1;
}

static void op1_and(ARGS1) {
  vxbu x_is_f = vxbu_load(slots[inst.and.x].is_f);
  vxbu x_is_t = vxbu_load(slots[inst.and.x].is_t);
  vyhu x_link = vyhu_load(slots[inst.and.x].link);
  vxbu y_is_f = vxbu_load(slots[inst.and.y].is_f);
  vxbu y_is_t = vxbu_load(slots[inst.and.y].is_t);
  vyhu y_link = vyhu_load(slots[inst.and.y].link);
  vxbu_store(slots[pc].is_f, vxbu_or(x_is_f, y_is_f));
  vxbu_store(slots[pc].is_t, vxbu_and(x_is_t, y_is_t));
  vyhu link = vyhu_dup((uint16_t) pc);
  link = vyhu_select(vxbu_vyhu_movemask(x_is_t), y_link, link);
  link = vyhu_select(vxbu_vyhu_movemask(y_is_t), x_link, link);
  vyhu_store(slots[pc].link, link);
  DISPATCH1;
}

static void op1_or(ARGS1) {
  vxbu x_is_f = vxbu_load(slots[inst.or.x].is_f);
  vxbu x_is_t = vxbu_load(slots[inst.or.x].is_t);
  vyhu x_link = vyhu_load(slots[inst.or.x].link);
  vxbu y_is_f = vxbu_load(slots[inst.or.y].is_f);
  vxbu y_is_t = vxbu_load(slots[inst.or.y].is_t);
  vyhu y_link = vyhu_load(slots[inst.or.y].link);
  vxbu_store(slots[pc].is_f, vxbu_and(x_is_f, y_is_f));
  vxbu_store(slots[pc].is_t, vxbu_or(x_is_t, y_is_t));
  vyhu link = vyhu_dup((uint16_t) pc);
  link = vyhu_select(vxbu_vyhu_movemask(x_is_f), y_link, link);
  link = vyhu_select(vxbu_vyhu_movemask(y_is_f), x_link, link);
  vyhu_store(slots[pc].link, link);
  DISPATCH1;
}

static void op1_ret(ARGS1) {
}

static void op1_ret_const(ARGS1) {
}

static void analyze(Geometry geometry, Inst * code, Input1 * input, Slot1 * slots) {
  static Table1 TABLE = {{
    op1_line,
    op1_ellipse,
    op1_and,
    op1_or,
    op1_ret,
    op1_ret_const,
  }};

  Inst inst = code[0];
  TABLE.ops[inst.op](geometry, code, input, slots, &TABLE, 0, inst);
}

// -------- SPECIALIZE CODE TO 16 SUBREGIONS --------

static void specialize(
    Arena arena,
    Arena * code_arena,
    Geometry geometry,
    size_t code_len,
    Inst code[code_len],
    float xmin,
    float xlen,
    float ymax,
    float ylen,
    size_t out_code_len[16],
    Inst * out_code[16]
  )
{
  Input1 input;

  for (size_t k = 0; k < 5; k ++) {
    input.x[k] = xmin + 0.25f * xlen * (float) k;
    input.y[k] = ymax - 0.25f * ylen * (float) k;
  }

  Slot1 * slots = ALLOC_ARRAY(&arena, code_len, Slot1);

  analyze(geometry, code, &input, slots);

  uint16_t * black = ALLOC_ARRAY(&arena, code_len, uint16_t);
  uint16_t * remap = ALLOC_ARRAY(&arena, code_len, uint16_t);

  PQ gray;
  pq_init(&gray, &arena, code_len);

  for (size_t t = 0; t < 16; t ++) {
    size_t sub_code_len = 0;

    {
      uint16_t k = (uint16_t) (code_len - 1);
      black[sub_code_len ++] = k;
      Inst inst = code[k];
      if (inst.op == OP_RET) {
        if (! slots[inst.ret.x].is_f[t] && ! slots[inst.ret.x].is_t[t]) {
          pq_insert(&gray, slots[inst.ret.x].link[t]);
        }
      }
    }

    while (! pq_is_empty(&gray)) {
      uint16_t k = pq_pop(&gray);
      black[sub_code_len ++] = k;
      Inst inst = code[k];

      switch (inst.op) {
      case OP_AND:
        pq_insert(&gray, slots[inst.and.x].link[t]);
        pq_insert(&gray, slots[inst.and.y].link[t]);
        break;
      case OP_OR:
        pq_insert(&gray, slots[inst.or.x].link[t]);
        pq_insert(&gray, slots[inst.or.y].link[t]);
        break;
      default:
        break;
      }
    }

    Inst * sub_code = ALLOC_ARRAY(code_arena, sub_code_len, Inst);

    out_code[t] = sub_code;
    out_code_len[t] = sub_code_len;

    for (size_t i = 0; i < sub_code_len; i ++) {
      uint16_t k = black[sub_code_len - 1 - i];

      remap[k] = (uint16_t) i;
      Inst inst = code[k];

      switch (inst.op) {
      case OP_AND:
        inst.and.x = remap[slots[inst.and.x].link[t]];
        inst.and.y = remap[slots[inst.and.y].link[t]];
        break;
      case OP_OR:
        inst.or.x = remap[slots[inst.or.x].link[t]];
        inst.or.y = remap[slots[inst.or.y].link[t]];
        break;
      case OP_RET:
        if (slots[inst.ret.x].is_f[t]) {
          inst = (Inst) { OP_RET_CONST, .ret_const = { false } };
        } else if (slots[inst.ret.x].is_t[t]) {
          inst = (Inst) { OP_RET_CONST, .ret_const = { true } };
        } else {
          inst.ret.x = remap[slots[inst.ret.x].link[t]];
        }
        break;
      default:
        break;
      }

      sub_code[i] = inst;
    }
  }
}

// -------- DRAW A 256 PIXEL REGION --------

typedef struct {
  float x[16];
  float y[16];
} Input2;

typedef struct {
  uint8_t bitset[32];
} Slot2;

typedef struct Table2_ {
  size_t (* ops[6])(Geometry, Inst *, Input2 *, Slot2 *, struct Table2_ *, size_t, Inst);
} Table2;

#define ARGS2 \
  __attribute__((unused)) Geometry geometry, \
  __attribute__((unused)) Inst * code, \
  __attribute__((unused)) Input2 * input, \
  __attribute__((unused)) Slot2 * slots, \
  __attribute__((unused)) Table2 * table, \
  __attribute__((unused)) size_t pc, \
  __attribute__((unused)) Inst inst

#define DISPATCH2 \
  do { \
    Inst inst = code[pc + 1]; \
    return table->ops[inst.op](geometry, code, input, slots, table, pc + 1, inst); \
  } while (0)

static size_t op2_line(ARGS2) {
  Line l = geometry.line[inst.line.index];
  vzsf x = vzsf_load(input->x);
  vzsf z = vzsf_fma_n(x, l.a, vzsf_dup(l.c));
  for (size_t h = 0; h < 2; h ++) {
    vxbu m = vxbu_dup(0);
    for (size_t i = 8; i -- != 0; ) {
      float y = input->y[8 * h + i];
      vxbu w = vzsu_vxbu_movemask(vzsf_le(z, vzsf_dup(- l.b * y)));
      m = vxbu_add(m, m);
      m = vxbu_sub(m, w);
    }
    vxbu_store(&slots[pc].bitset[16 * h], m);
  }
  DISPATCH2;
}

static size_t op2_ellipse(ARGS2) {
  Ellipse e = geometry.ellipse[inst.ellipse.index];
  vzsf x = vzsf_load(input->x);
  vzsu p = vzsu_dup(inst.ellipse.outside ? 1ul << 31 : 0);
  vzsf r = vzsf_fma_n(x, e.a, vzsf_dup(e.c));
  vzsf s = vzsf_fma_n(x, e.d, vzsf_dup(e.f));
  for (size_t h = 0; h < 2; h ++) {
    vxbu m = vxbu_dup(0);
    for (size_t i = 8; i -- != 0; ) {
      float y = input->y[8 * h + i];
      vzsf u = vzsf_add_n(r, e.b * y);
      vzsf v = vzsf_add_n(s, e.e * y);
      vzsf z = vzsf_xor(vzsf_fma(u, u, vzsf_fma(v, v, vzsf_dup(-1.0f))), p);
      vxbu w = vzsu_vxbu_movemask(vzsf_le(z, vzsf_dup(0.0f)));
      m = vxbu_add(m, m);
      m = vxbu_sub(m, w);
    }
    vxbu_store(&slots[pc].bitset[16 * h], m);
  }
  DISPATCH2;
}

static size_t op2_and(ARGS2) {
  vybu x = vybu_load(slots[inst.and.x].bitset);
  vybu y = vybu_load(slots[inst.and.y].bitset);
  vybu_store(slots[pc].bitset, vybu_and(x, y));
  DISPATCH2;
}

static size_t op2_or(ARGS2) {
  vybu x = vybu_load(slots[inst.or.x].bitset);
  vybu y = vybu_load(slots[inst.or.y].bitset);
  vybu_store(slots[pc].bitset, vybu_or(x, y));
  DISPATCH2;
}

static size_t op2_ret(ARGS2) {
  return inst.ret.x;
}

static size_t op2_ret_const(ARGS2) {
  vybu_store(slots[pc].bitset, vybu_dup(inst.ret_const.value ? UINT8_MAX : 0));
  return pc;
}

static void draw_tile_16(
    Arena arena,
    Geometry geometry,
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
  static Table2 TABLE = {{
    op2_line,
    op2_ellipse,
    op2_and,
    op2_or,
    op2_ret,
    op2_ret_const,
  }};

  Input2 input;

  Slot2 * slots = ALLOC_ARRAY(&arena, code_len, Slot2);

  float dx = 0.0625f * xlen;
  float dy = 0.0625f * ylen;

  for (size_t k = 0; k < 16; k ++) {
    input.x[k] = xmin + 0.5f * dx + dx * (float) k;
    input.y[k] = ymax - 0.5f * dy - dy * (float) k;
  }

  Inst inst = code[0];
  size_t result = TABLE.ops[inst.op](geometry, code, &input, slots, &TABLE, 0, inst);

  for (size_t h = 0; h < 2; h ++) {
    vxbu m = vxbu_load(&slots[result].bitset[16 * h]);
    for (size_t i = 0; i < 8; i ++) {
      vxbu w = vxbu_test(m, vxbu_dup((uint8_t) (1 << i)));
      vxbu_store(tile + 8 * stride * h + stride * i, w);
    }
  }
}

static void fill_tile_16(Inst * code, size_t stride, uint8_t * tile) {
  // Per the spec, this should be `? 255 : 0`, but we add some grays for
  // illustrative purposes.
  uint8_t value = code[0].ret_const.value ? 192 : 64;

  for (size_t i = 0; i < 16; i ++) {
    memset(tile + stride * i, value, 16);
  }
}

// -------- DRAW RECURSIVELY --------

static void fill_tile(Inst * code, size_t resolution, size_t stride, uint8_t * tile) {
  assert(resolution >= 64);

  uint8_t value = code[0].ret_const.value ? 160 : 96;

  for (size_t i = 0; i < resolution; i ++) {
    for (size_t j = 0; j < resolution; j += 64) {
      // it would be nice to use non-temporal stores
      memset(tile + stride * i + j, value, 64);
    }
  }
}

static void draw_tile(
    Arena arena,
    Arena code_arena,
    Geometry geometry,
    size_t code_len,
    Inst code[code_len],
    float xmin,
    float xlen,
    float ymax,
    float ylen,
    size_t resolution,
    size_t stride,
    uint8_t * tile
  )
{
  if (resolution == 128) {
    for (size_t t = 0; t < 4; t ++) {
      size_t i = t / 2;
      size_t j = t % 2;

      draw_tile(
          arena,
          code_arena,
          geometry,
          code_len,
          code,
          xmin + 0.5f * xlen * (float) j,
          0.5f * xlen,
          ymax - 0.5f * ylen * (float) i,
          0.5f * ylen,
          resolution / 2,
          stride,
          tile + stride * resolution / 2 * i + resolution / 2 * j
        );
    }

    return;
  }

  size_t sub_code_len[16];
  Inst * sub_code[16];

  specialize(
      arena,
      &code_arena,
      geometry,
      code_len,
      code,
      xmin,
      xlen,
      ymax,
      ylen,
      sub_code_len,
      sub_code
    );

  if (resolution == 64) {
    for (size_t t = 0; t < 16; t ++) {
      size_t i = t / 4;
      size_t j = t % 4;

      if (sub_code[t][0].op == OP_RET_CONST) {
        fill_tile_16(
            sub_code[t],
            stride,
            tile + stride * resolution / 4 * i + resolution / 4 * j
          );

        continue;
      }

      draw_tile_16(
          arena,
          geometry,
          sub_code_len[t],
          sub_code[t],
          xmin + 0.25f * xlen * (float) j,
          0.25f * xlen,
          ymax - 0.25f * ylen * (float) i,
          0.25f * ylen,
          stride,
          tile + stride * resolution / 4 * i + resolution / 4 * j
        );
    }

    return;
  }

  for (size_t t = 0; t < 16; t ++) {
    size_t i = t / 4;
    size_t j = t % 4;

    if (sub_code[t][0].op == OP_RET_CONST) {
      fill_tile(
          sub_code[t],
          resolution / 4,
          stride,
          tile + stride * resolution / 4 * i + resolution / 4 * j
        );

      continue;
    }

    draw_tile(
        arena,
        code_arena,
        geometry,
        sub_code_len[t],
        sub_code[t],
        xmin + 0.25f * xlen * (float) j,
        0.25f * xlen,
        ymax - 0.25f * ylen * (float) i,
        0.25f * ylen,
        resolution / 4,
        stride,
        tile + stride * resolution / 4 * i + resolution / 4 * j
      );
  }
}

void render(
    Geometry geometry,
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
  assert(512 <= resolution && resolution <= 4096);
  assert((resolution & (resolution - 1)) == 0);

  size_t sub_code_len[64];
  Inst * sub_code[64];

#pragma omp parallel
  {
    Arena arena;
    Arena code_arena;
    arena_init(&arena, 1 << 19);
    arena_init(&code_arena, 1 << 17);

#pragma omp for schedule(dynamic, 1)
    for (size_t t = 0; t < 4; t ++) {
      size_t i = t / 2;
      size_t j = t % 2;

      specialize(
          arena,
          &code_arena,
          geometry,
          code_len,
          code,
          xmin + 0.5f * (xmax - xmin) * (float) j,
          0.5f * (xmax - xmin),
          ymax - 0.5f * (ymax - ymin) * (float) i,
          0.5f * (ymax - ymin),
          &sub_code_len[16 * t],
          &sub_code[16 * t]
        );
    }

#pragma omp for schedule(dynamic, 1)
    for (size_t t = 0; t < 64; t ++) {
      // we've sliced the image into 64 regions
      //
      //      0  1  2  3  4  5  6  7
      //   ┌────────────────────────
      // 0 │ 00 01 02 03 16 17 18 19
      // 1 │ 04 05 06 07
      // 2 │ 08 09 10 11
      // 3 │ 12 13 14 15
      // 4 │ 32          48 49 50 51
      // 5 │ 36          52
      // 6 │ 40          56
      // 7 │ 44          60

      size_t i = t / 16 / 2 * 4 + t % 16 / 4;
      size_t j = t / 16 % 2 * 4 + t % 16 % 4;

      if (sub_code[t][0].op == OP_RET_CONST) {
        fill_tile(
            sub_code[t],
            resolution / 8,
            resolution,
            &image[resolution / 8 * i][resolution / 8 * j]
          );

        continue;
      }

      draw_tile(
          arena,
          code_arena,
          geometry,
          sub_code_len[t],
          sub_code[t],
          xmin + 0.125f * (xmax - xmin) * (float) j,
          0.125f * (xmax - xmin),
          ymax - 0.125f * (ymax - ymin) * (float) i,
          0.125f * (ymax - ymin),
          resolution / 8,
          resolution,
          &image[resolution / 8 * i][resolution / 8 * j]
        );
    }

    arena_drop(&arena);
    arena_drop(&code_arena);
  }
}
