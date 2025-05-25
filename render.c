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

typedef struct {
  uint8_t is_f[16];
  uint8_t is_t[16];
  uint16_t link[16];
} Slot1;

typedef struct {
  size_t capacity;
  float x[5];
  float y[5];
  Line * line;
  Oval * oval;
  Slot1 * slots;
} Env1;

static void env1_init(Env1 * t, size_t capacity) {
  t->capacity = capacity;
  t->slots = malloc(capacity * sizeof(Slot1));
}

static void env1_drop(Env1 * t) {
  free(t->slots);
}

typedef struct {
  uint8_t bools[256];
} Slot2;

typedef struct {
  size_t capacity;
  float x[16];
  float y[16];
  Line * line;
  Oval * oval;
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

// -------- 16-wide interval arithmetic --------

typedef struct {
  vzsf lo;
  vzsf hi;
} range_;

static inline range_ add_(range_ x, range_ y) {
  return (range_) {
    vzsf_add(x.lo, y.lo),
    vzsf_add(x.hi, y.hi),
  };
}

static inline range_ add_n_(range_ x, float y) {
  return (range_) {
    vzsf_add_n(x.lo, y),
    vzsf_add_n(x.hi, y),
  };
}

static inline range_ mul_n_(range_ x, float y) {
  vzsu p = vzsu_dup(y < 0.0f ? UINT32_MAX : 0);
  return (range_) {
    vzsf_mul_n(vzsf_select(p, x.hi, x.lo), y),
    vzsf_mul_n(vzsf_select(p, x.lo, x.hi), y),
  };
}

static inline range_ square_(range_ x) {
  return (range_) {
    vzsf_or(
      vzsf_and(vzsf_mul(x.lo, x.lo), vzsf_lt(VZSF_ZERO, x.lo)),
      vzsf_and(vzsf_mul(x.hi, x.hi), vzsf_lt(x.hi, VZSF_ZERO))),
    vzsf_max(vzsf_mul(x.lo, x.lo), vzsf_mul(x.hi, x.hi)),
  };
}

// -------- FORWARD ANALYSIS FOR 16 SUBREGIONS --------

typedef struct Tbl1_ {
  size_t (* ops[6])(Inst *, Env1 *, Slot1 *, struct Tbl1_ *, size_t, Inst);
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

static inline vzsf env1_xmin(Env1 * ep) {
  vxsf x = vxsf_load(&ep->x[0]);
  return vzsf_from_vxsf_x4(x, x, x, x);
}

static inline vzsf env1_xmax(Env1 * ep) {
  vxsf x = vxsf_load(&ep->x[1]);
  return vzsf_from_vxsf_x4(x, x, x, x);
}

static inline vzsf env1_ymin(Env1 * ep) {
  vxsf y = vxsf_load(&ep->y[1]);
  return
    vzsf_from_vxsf_x4(
      vxsf_dup(vxsf_get(y, 0)),
      vxsf_dup(vxsf_get(y, 1)),
      vxsf_dup(vxsf_get(y, 2)),
      vxsf_dup(vxsf_get(y, 3)));
}

static inline vzsf env1_ymax(Env1 * ep) {
  vxsf y = vxsf_load(&ep->y[0]);
  return
    vzsf_from_vxsf_x4(
      vxsf_dup(vxsf_get(y, 0)),
      vxsf_dup(vxsf_get(y, 1)),
      vxsf_dup(vxsf_get(y, 2)),
      vxsf_dup(vxsf_get(y, 3)));
}

static size_t op1_line(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  Line line = ep->line[inst.line.index];
  float a = line.a;
  float b = line.b;
  float c = line.c;

  range_ x = { env1_xmin(ep), env1_xmax(ep) };
  range_ y = { env1_ymin(ep), env1_ymax(ep) };

  range_ z = add_(mul_n_(x, a), mul_n_(y, b));

  vxbu_store(sp[pc].is_f, vzsu_vxbu_movemask(vzsf_lt(vzsf_dup(- c), z.lo)));
  vxbu_store(sp[pc].is_t, vzsu_vxbu_movemask(vzsf_le(z.hi, vzsf_dup(- c))));
  vyhu_store(sp[pc].link, vyhu_dup((uint16_t) pc));

  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_oval(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  Oval oval = ep->oval[inst.oval.index];
  float a = oval.a;
  float b = oval.b;
  float c = oval.c;
  float d = oval.d;
  float e = oval.e;
  float f = oval.f;

  range_ x = { env1_xmin(ep), env1_xmax(ep) };
  range_ y = { env1_ymin(ep), env1_ymax(ep) };

  range_ z =
    add_n_(
      add_(
        square_(add_n_(add_(mul_n_(x, a), mul_n_(y, b)), c)),
        square_(add_n_(add_(mul_n_(x, d), mul_n_(y, e)), f))),
      -1.0f);

  vzsu p = vzsu_dup(inst.oval.outside ? UINT32_MAX : 0);
  vzsf u = vzsf_select(p, vzsf_neg(z.hi), z.lo);
  vzsf v = vzsf_select(p, vzsf_neg(z.lo), z.hi);

  vxbu_store(sp[pc].is_f, vzsu_vxbu_movemask(vzsf_lt(vzsf_dup(0.0f), u)));
  vxbu_store(sp[pc].is_t, vzsu_vxbu_movemask(vzsf_le(v, vzsf_dup(0.0f))));

  /*
  vxbu_store(sp[pc].is_f, vzsu_vxbu_movemask(vzsu_dup(0)));
  vxbu_store(sp[pc].is_t, vzsu_vxbu_movemask(vzsu_dup(0)));
  */

  vyhu_store(sp[pc].link, vyhu_dup((uint16_t) pc));

  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_and(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  vxbu x_is_f = vxbu_load(sp[inst.and.x].is_f);
  vxbu x_is_t = vxbu_load(sp[inst.and.x].is_t);
  vyhu x_link = vyhu_load(sp[inst.and.x].link);
  vxbu y_is_f = vxbu_load(sp[inst.and.y].is_f);
  vxbu y_is_t = vxbu_load(sp[inst.and.y].is_t);
  vyhu y_link = vyhu_load(sp[inst.and.y].link);
  vxbu_store(sp[pc].is_f, vxbu_or(x_is_f, y_is_f));
  vxbu_store(sp[pc].is_t, vxbu_and(x_is_t, y_is_t));
  vyhu link = vyhu_dup((uint16_t) pc);
  link = vyhu_select(vxbu_vyhu_movemask(x_is_t), y_link, link);
  link = vyhu_select(vxbu_vyhu_movemask(y_is_t), x_link, link);
  vyhu_store(sp[pc].link, link);
  return op1_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op1_or(Inst * cp, Env1 * ep, Slot1 * sp, Tbl1 * tp, size_t pc, Inst inst) {
  vxbu x_is_f = vxbu_load(sp[inst.or.x].is_f);
  vxbu x_is_t = vxbu_load(sp[inst.or.x].is_t);
  vyhu x_link = vyhu_load(sp[inst.or.x].link);
  vxbu y_is_f = vxbu_load(sp[inst.or.y].is_f);
  vxbu y_is_t = vxbu_load(sp[inst.or.y].is_t);
  vyhu y_link = vyhu_load(sp[inst.or.y].link);
  vxbu_store(sp[pc].is_f, vxbu_and(x_is_f, y_is_f));
  vxbu_store(sp[pc].is_t, vxbu_or(x_is_t, y_is_t));
  vyhu link = vyhu_dup((uint16_t) pc);
  link = vyhu_select(vxbu_vyhu_movemask(x_is_f), y_link, link);
  link = vyhu_select(vxbu_vyhu_movemask(y_is_f), x_link, link);
  vyhu_store(sp[pc].link, link);
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
    op1_line,
    op1_oval,
    op1_and,
    op1_or,
    op1_ret,
    op1_ret_const,
  }};

  (void) op1_dispatch(cp, ep, sp, &TBL1, 0);
}

// -------- RASTERIZE A 256 PIXEL REGION --------

typedef struct Tbl2_ {
  size_t (* ops[6])(Inst *, Env2 *, Slot2 *, struct Tbl2_ *, size_t, Inst);
} Tbl2;

static inline size_t op2_dispatch(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc) {
  Inst inst = cp[pc];
  return tp->ops[inst.op](cp, ep, sp, tp, pc, inst);
}

static size_t op2_line(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  Line line = ep->line[inst.line.index];
  float a = line.a;
  float b = line.b;
  float c = line.c;

  vzsf x = vzsf_load(ep->x);

  for (size_t i = 0; i < 16; i ++) {
    float y = ep->y[i];
    vzsf z = vzsf_add_n(vzsf_mul_n(x, a), b * y);
    vxbu w = vzsu_vxbu_movemask(vzsf_le(z, vzsf_dup(- c)));
    vxbu_store(&sp[pc].bools[16 * i], w);
  }

  return op2_dispatch(cp, ep, sp, tp, pc + 1);
}

static size_t op2_oval(Inst * cp, Env2 * ep, Slot2 * sp, Tbl2 * tp, size_t pc, Inst inst) {
  Oval oval = ep->oval[inst.oval.index];
  float a = oval.a;
  float b = oval.b;
  float c = oval.c;
  float d = oval.d;
  float e = oval.e;
  float f = oval.f;

  vzsf x = vzsf_load(ep->x);

  for (size_t i = 0; i < 16; i ++) {
    float y = ep->y[i];

    vzsf z =
      vzsf_add_n(
        vzsf_add(
          vzsf_square(vzsf_add_n(vzsf_mul_n(x, a), y * b + c)),
          vzsf_square(vzsf_add_n(vzsf_mul_n(x, d), y * e + f))),
        -1.0f);

    vzsu p = vzsu_dup(inst.oval.outside ? UINT32_MAX : 0);
    vxbu w = vzsu_vxbu_movemask(vzsf_le(vzsf_select(p, vzsf_neg(z), z), vzsf_dup(0.0f)));

    vxbu_store(&sp[pc].bools[16 * i], w);
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
    vzbu_store(&sp[pc].bools[64 * h], vzbu_dup(inst.ret_const.value ? UINT8_MAX : 0));
  }
  return pc;
}

static void rasterize(
    Scratch * scratch,
    size_t code_len,
    Inst code[code_len],
    Line * line,
    Oval * oval,
    float xmin,
    float xlen,
    float ymax,
    float ylen,
    size_t stride,
    uint8_t * tile
  )
{
  static Tbl2 TBL2 = {{
    op2_line,
    op2_oval,
    op2_and,
    op2_or,
    op2_ret,
    op2_ret_const,
  }};

  Env2 * ep = scratch_env2(scratch, code_len);

  ep->line = line;
  ep->oval = oval;

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
    Line * line,
    Oval * oval,
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

  ep->line = line;
  ep->oval = oval;

  for (size_t k = 0; k < 5; k ++) {
    ep->x[k] = xmin + 0.25f * xlen * (float) k;
    ep->y[k] = ymax - 0.25f * ylen * (float) k;
  }

  Slot1 * sp = ep->slots;

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
        if (! sp[inst.ret.x].is_f[t] && ! sp[inst.ret.x].is_t[t]) {
          pq_insert(gray, sp[inst.ret.x].link[t]);
        }
      }
    }

    while (! pq_is_empty(gray)) {
      uint16_t k = pq_pop(gray);
      rev[sub_code_len ++] = k;
      Inst inst = code[k];

      switch (inst.op) {
      case OP_AND:
        pq_insert(gray, sp[inst.and.x].link[t]);
        pq_insert(gray, sp[inst.and.y].link[t]);
        break;
      case OP_OR:
        pq_insert(gray, sp[inst.or.x].link[t]);
        pq_insert(gray, sp[inst.or.y].link[t]);
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
      case OP_AND:
        inst.and.x = map[sp[inst.and.x].link[t]];
        inst.and.y = map[sp[inst.and.y].link[t]];
        break;
      case OP_OR:
        inst.or.x = map[sp[inst.or.x].link[t]];
        inst.or.y = map[sp[inst.or.y].link[t]];
        break;
      case OP_RET:
        if (sp[inst.ret.x].is_f[t]) {
          inst = (Inst) { OP_RET_CONST, .ret_const = { false } };
        } else if (sp[inst.ret.x].is_t[t]) {
          inst = (Inst) { OP_RET_CONST, .ret_const = { true } };
        } else {
          inst.ret.x = map[sp[inst.ret.x].link[t]];
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
    Line * line,
    Oval * oval,
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
    uint8_t value = code[0].ret_const.value ? 255 : 0;

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
        line,
        oval,
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
          line,
          oval,
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
      line,
      oval,
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
        line,
        oval,
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
    Prog * prog,
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
        prog->code_len,
        prog->code,
        prog->line,
        prog->oval,
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
