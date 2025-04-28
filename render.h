#pragma once

typedef enum {
  OP_AFFINE,
  OP_HYPOT2,
  OP_LE_CONST,
  OP_GE_CONST,
  OP_AND,
  OP_OR,
  OP_RET,
  OP_RET_CONST,
} Op;

typedef struct {
  Op op;
  union {
    struct { float a; float b; float c; } affine;
    struct { uint16_t x; uint16_t y; } hypot2;
    struct { uint16_t x; float a; } le_const;
    struct { uint16_t x; float a; } ge_const;
    struct { uint16_t x; uint16_t y; } and;
    struct { uint16_t x; uint16_t y; } or;
    struct { uint16_t x; } ret;
    struct { bool a; } ret_const;
  };
} Inst;

// PRECONDITIONS:
//
// - `code` must be a valid program
// - `resolution` must be a power of two between 256 and 8192

void render(
    size_t code_len,
    Inst code[code_len],
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    size_t resolution,
    uint8_t image[resolution][resolution]
  );
