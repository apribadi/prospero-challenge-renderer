#include <arm_neon.h>
#include <assert.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
static void arena_oom(void) {
  fprintf(stderr, "arena: allocation failed!\n");
  abort();
}

static void arena_init(Arena * arena, size_t capacity) {
  void * base = malloc(capacity);
  if (! base) arena_oom();
  arena->base = base;
  arena->available = capacity;
}

static void arena_drop(Arena * arena) {
  free(arena->base);
}

static inline void * arena_alloc(Arena * arena, size_t size) {
  if (size > arena->available) arena_oom();
  size_t n = (arena->available - size) & - (size_t) 8;
  arena->available = n;
  return (void *) (arena->base + n);
}

static inline void * arena_alloc_zeroed(Arena * arena, size_t size) {
  return memset(arena_alloc(arena, size), 0, size);
}

// -------- PRIORITY QUEUE (AND SET) --------

typedef struct {
  size_t size;
  uint16_t * data;
  bool * included;
} PQ;

static void pq_init(PQ * t, Arena * arena, size_t capacity) {
  t->size = 0;
  t->data = arena_alloc(arena, capacity * sizeof(uint16_t));
  t->included = arena_alloc_zeroed(arena, capacity * sizeof(bool));
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

// -------- 16-WIDE INTERVAL ARITHMETIC --------

typedef struct {
  vzsf lo;
  vzsf hi;
} Range;

static inline Range range_add(Range x, Range y) {
  return (Range) {
    vzsf_add(x.lo, y.lo),
    vzsf_add(x.hi, y.hi),
  };
}

static inline Range range_add_n(Range x, float y) {
  return (Range) {
    vzsf_add_n(x.lo, y),
    vzsf_add_n(x.hi, y),
  };
}

static inline Range range_mul_n(Range x, float y) {
  vzsu p = vzsu_dup(y < 0.0f ? UINT32_MAX : 0);
  return (Range) {
    vzsf_mul_n(vzsf_select(p, x.hi, x.lo), y),
    vzsf_mul_n(vzsf_select(p, x.lo, x.hi), y),
  };
}

static inline Range range_sq(Range x) {
  return (Range) {
    vzsf_or(
      vzsf_and(vzsf_mul(x.lo, x.lo), vzsf_lt(vzsf_dup(0.0f), x.lo)),
      vzsf_and(vzsf_mul(x.hi, x.hi), vzsf_lt(x.hi, vzsf_dup(0.0f)))),
    vzsf_max(vzsf_mul(x.lo, x.lo), vzsf_mul(x.hi, x.hi)),
  };
}

// -------- FORWARD ANALYSIS FOR 16 SUBREGIONS --------

typedef struct {
  uint8_t is_f[16];
  uint8_t is_t[16];
  uint16_t link[16];
} Slot1;

typedef struct Tbl1_ {
  void (* ops[6])(Shapes, Inst *, Slot1 *, struct Tbl1_ *, vxsf, vxsf, vxsf, vxsf, size_t, Inst);
} Tbl1;

static inline vzsf broadcast_x1(vxsf x) {
  return vzsf_from_vxsf_x4(x, x, x, x);
}

static inline vzsf broadcast_y1(vxsf y) {
  return
    vzsf_from_vxsf_x4(
      vxsf_dup(vxsf_get(y, 0)),
      vxsf_dup(vxsf_get(y, 1)),
      vxsf_dup(vxsf_get(y, 2)),
      vxsf_dup(vxsf_get(y, 3)));
}

#define ARGS1 \
  __attribute__((unused)) Shapes shapes, \
  __attribute__((unused)) Inst * code, \
  __attribute__((unused)) Slot1 * slots, \
  __attribute__((unused)) Tbl1 * tbl, \
  __attribute__((unused)) vxsf xmin, \
  __attribute__((unused)) vxsf xmax, \
  __attribute__((unused)) vxsf ymin, \
  __attribute__((unused)) vxsf ymax, \
  __attribute__((unused)) size_t pc, \
  __attribute__((unused)) Inst inst

#define DISPATCH1 \
  do { \
    Inst inst = code[pc + 1]; \
    tbl->ops[inst.op](shapes, code, slots, tbl, xmin, xmax, ymin, ymax, pc + 1, inst); \
  } while (0)

static void op1_line(ARGS1) {
  Range x = { broadcast_x1(xmin), broadcast_x1(xmax) };
  Range y = { broadcast_y1(ymin), broadcast_y1(ymax) };
  Line line = shapes.lines[inst.line.index];
  float a = line.a;
  float b = line.b;
  float c = line.c;
  Range z = range_add(range_mul_n(x, a), range_mul_n(y, b));
  vxbu_store(slots[pc].is_f, vzsu_vxbu_movemask(vzsf_lt(vzsf_dup(- c), z.lo)));
  vxbu_store(slots[pc].is_t, vzsu_vxbu_movemask(vzsf_le(z.hi, vzsf_dup(- c))));
  vyhu_store(slots[pc].link, vyhu_dup((uint16_t) pc));
  DISPATCH1;
}

static void op1_oval(ARGS1) {
  Range x = { broadcast_x1(xmin), broadcast_x1(xmax) };
  Range y = { broadcast_y1(ymin), broadcast_y1(ymax) };
  Oval oval = shapes.ovals[inst.oval.index];
  float a = oval.a;
  float b = oval.b;
  float c = oval.c;
  float d = oval.d;
  float e = oval.e;
  float f = oval.f;
  Range z =
    range_add_n(
      range_add(
        range_sq(range_add_n(range_add(range_mul_n(x, a), range_mul_n(y, b)), c)),
        range_sq(range_add_n(range_add(range_mul_n(x, d), range_mul_n(y, e)), f))),
      -1.0f);
  vzsu p = vzsu_dup(inst.oval.outside ? UINT32_MAX : 0);
  vzsf u = vzsf_select(p, vzsf_neg(z.hi), z.lo);
  vzsf v = vzsf_select(p, vzsf_neg(z.lo), z.hi);
  vxbu_store(slots[pc].is_f, vzsu_vxbu_movemask(vzsf_lt(vzsf_dup(0.0f), u)));
  vxbu_store(slots[pc].is_t, vzsu_vxbu_movemask(vzsf_le(v, vzsf_dup(0.0f))));
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

static void analyze(
    Shapes shapes,
    Inst * code,
    Slot1 * slots,
    float xmin[4],
    float xmax[4],
    float ymin[4],
    float ymax[4]
  )
{
  static Tbl1 TBL = {{
    op1_line,
    op1_oval,
    op1_and,
    op1_or,
    op1_ret,
    op1_ret_const,
  }};

  Inst inst = code[0];

  TBL.ops[inst.op](
      shapes,
      code,
      slots,
      &TBL,
      vxsf_load(xmin),
      vxsf_load(xmax),
      vxsf_load(ymin),
      vxsf_load(ymax),
      0,
      inst
    );
}

// -------- SPECIALIZE CODE TO 16 SUBREGIONS --------

static void specialize(
    Arena * arena,
    Shapes shapes,
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
  float x[5];
  float y[5];

  for (size_t k = 0; k < 5; k ++) {
    x[k] = xmin + 0.25f * xlen * (float) k;
    y[k] = ymax - 0.25f * ylen * (float) k;
  }

  Slot1 * slots = arena_alloc(arena, code_len * sizeof(Slot1));

  analyze(shapes, code, slots, x + 0, x + 1, y + 1, y + 0);

  uint16_t * black = arena_alloc(arena, code_len * sizeof(uint16_t));
  uint16_t * remap = arena_alloc(arena, code_len * sizeof(uint16_t));

  PQ gray;
  pq_init(&gray, arena, code_len);

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

    Inst * sub_code = arena_alloc(arena, sub_code_len * sizeof(Inst));

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

// -------- RASTERIZE A 256 PIXEL REGION --------

typedef struct {
  uint8_t bits[32];
} Slot2;

typedef struct {
  float x[16];
  float y[16];
} Env2;

typedef struct Tbl2_ {
  size_t (* ops[6])(Shapes, Inst *, Env2 *, Slot2 *, struct Tbl2_ *, size_t, Inst);
} Tbl2;

#define ARGS2 \
  __attribute__((unused)) Shapes shapes, \
  __attribute__((unused)) Inst * code, \
  __attribute__((unused)) Env2 * ep, \
  __attribute__((unused)) Slot2 * slots, \
  __attribute__((unused)) Tbl2 * tbl, \
  __attribute__((unused)) size_t pc, \
  __attribute__((unused)) Inst inst

#define DISPATCH2 \
  do { \
    Inst inst = code[pc + 1]; \
    return tbl->ops[inst.op](shapes, code, ep, slots, tbl, pc + 1, inst); \
  } while (0)

static size_t op2_line(ARGS2) {
  Line line = shapes.lines[inst.line.index];
  float a = line.a;
  float b = line.b;
  float c = line.c;

  vzsf x = vzsf_load(ep->x);

  for (size_t h = 0; h < 2; h ++) {
    vxbu r = vxbu_dup(0);

    for (size_t i = 0; i < 8; i ++) {
      float y = ep->y[8 * h + i];
      vzsf z = vzsf_add_n(vzsf_mul_n(x, a), b * y);
      vxbu w = vzsu_vxbu_movemask(vzsf_le(z, vzsf_dup(- c)));
      r = vxbu_select(vxbu_dup((uint8_t) (1 << i)), w, r);
    }

    vxbu_store(&slots[pc].bits[16 * h], r);
  }

  DISPATCH2;
}

static size_t op2_oval(ARGS2) {
  Oval oval = shapes.ovals[inst.oval.index];
  float a = oval.a;
  float b = oval.b;
  float c = oval.c;
  float d = oval.d;
  float e = oval.e;
  float f = oval.f;

  vzsf x = vzsf_load(ep->x);

  for (size_t h = 0; h < 2; h ++) {
    vxbu r = vxbu_dup(0);

    for (size_t i = 0; i < 8; i ++) {
      float y = ep->y[8 * h + i];

      vzsf z =
        vzsf_add_n(
          vzsf_add(
            vzsf_sq(vzsf_add_n(vzsf_mul_n(x, a), y * b + c)),
            vzsf_sq(vzsf_add_n(vzsf_mul_n(x, d), y * e + f))),
          -1.0f);

      vzsu p = vzsu_dup(inst.oval.outside ? UINT32_MAX : 0);
      vxbu w = vzsu_vxbu_movemask(vzsf_le(vzsf_select(p, vzsf_neg(z), z), vzsf_dup(0.0f)));

      r = vxbu_select(vxbu_dup((uint8_t) (1 << i)), w, r);
    }

    vxbu_store(&slots[pc].bits[16 * h], r);
  }

  DISPATCH2;
}

static size_t op2_and(ARGS2) {
  vybu x = vybu_load(slots[inst.and.x].bits);
  vybu y = vybu_load(slots[inst.and.y].bits);
  vybu_store(slots[pc].bits, vybu_and(x, y));

  DISPATCH2;
}

static size_t op2_or(ARGS2) {
  vybu x = vybu_load(slots[inst.or.x].bits);
  vybu y = vybu_load(slots[inst.or.y].bits);
  vybu_store(slots[pc].bits, vybu_or(x, y));

  DISPATCH2;
}

static size_t op2_ret(ARGS2) {
  return inst.ret.x;
}

static size_t op2_ret_const(ARGS2) {
  vybu_store(slots[pc].bits, vybu_dup(inst.ret_const.value ? UINT8_MAX : 0));

  return pc;
}

static void rasterize(
    Arena arena,
    Shapes shapes,
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
  static Tbl2 TBL = {{
    op2_line,
    op2_oval,
    op2_and,
    op2_or,
    op2_ret,
    op2_ret_const,
  }};

  Env2 env;

  Slot2 * slots = arena_alloc(&arena, code_len * sizeof(Slot2));

  float dx = 0.0625f * xlen;
  float dy = 0.0625f * ylen;

  for (size_t k = 0; k < 16; k ++) {
    env.x[k] = xmin + 0.5f * dx + dx * (float) k;
    env.y[k] = ymax - 0.5f * dy - dy * (float) k;
  }

  Inst inst = code[0];
  size_t result = TBL.ops[inst.op](shapes, code, &env, slots, &TBL, 0, inst);

  for (size_t h = 0; h < 2; h ++) {
    vxbu r = vxbu_load(&slots[result].bits[16 * h]);
    for (size_t i = 0; i < 8; i ++) {
      size_t k = 8 * h + i;
      vxbu_store(tile + stride * k, vxbu_test(r, vxbu_dup((uint8_t) (1 << i))));
    }
  }
}

// -------- RENDER --------

static void render_tile(
    Arena arena,
    Shapes shapes,
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
  if (code_len == 1 && code[0].op == OP_RET_CONST) {
    uint8_t value = code[0].ret_const.value ? 255 : 0;

    for (size_t i = 0; i < resolution; i ++) {
      for (size_t j = 0; j < resolution; j += 16) {
        memset(tile + stride * i + j, value, 16);
      }
    }

    return;
  }

  if (resolution == 16) {
    rasterize(
        arena,
        shapes,
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
          arena,
          shapes,
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
      &arena,
      shapes,
      code_len,
      code,
      xmin,
      xlen,
      ymax,
      ylen,
      sub_code_len,
      sub_code
    );

  for (size_t t = 0; t < 16; t ++) {
    size_t i = t / 4;
    size_t j = t % 4;

    render_tile(
        arena,
        shapes,
        sub_code_len[t],
        sub_code[t],
        xmin + 0.25f * xlen * (float) j,
        0.25f * xlen,
        ymax - 0.25f * ylen * (float) i,
        0.25f * ylen,
        resolution / 4,
        stride,
        tile + resolution / 4 * stride * i + resolution / 4 * j
      );
  }
}

void render(
    Shapes shapes,
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
  assert(resolution >= 256);
  assert((resolution & (resolution - 1)) == 0);

#pragma omp parallel for
  for (size_t t = 0; t < 16; t ++) {
    size_t i = t / 4;
    size_t j = t % 4;

    Arena arena;
    arena_init(&arena, 1 << 19);

    render_tile(
        arena,
        shapes,
        code_len,
        code,
        xmin + 0.25f * (xmax - xmin) * (float) j,
        0.25f * (xmax - xmin),
        ymax - 0.25f * (ymax - ymin) * (float) i,
        0.25f * (ymax - ymin),
        resolution / 4,
        resolution,
        &image[resolution / 4 * i][resolution / 4 * j]
      );

    arena_drop(&arena);
  }
}
