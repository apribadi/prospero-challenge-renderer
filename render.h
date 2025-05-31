#pragma once

typedef enum {
  OP_LINE,
  OP_ELLIPSE,
  OP_AND,
  OP_OR,
  OP_RET,
  OP_RET_CONST,
} Op;

typedef struct {
  Op op;
  union {
    struct { uint16_t index; } line;
    struct { uint16_t index; bool outside; } ellipse;
    struct { uint16_t x; uint16_t y; } and;
    struct { uint16_t x; uint16_t y; } or;
    struct { uint16_t x; } ret;
    struct { bool value; } ret_const;
  };
} Inst;

// ax + by + c <= 0

typedef struct {
  float a;
  float b;
  float c;
} Line;

// if outside:
//   (ax + by + c) ** 2 + (dx + ey + f) ** 2 >= 1
// else:
//   (ax + by + c) ** 2 + (dx + ey + f) ** 2 <= 1

typedef struct {
  float a;
  float b;
  float c;
  float d;
  float e;
  float f;
} Ellipse;

typedef struct {
  Line * line;
  Ellipse * ellipse;
} Geometry;

// PRECONDITIONS:
//
// - `code` must be a valid program
// - `resolution` must be a power of two between 256 and 4096, inclusive

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
  );
