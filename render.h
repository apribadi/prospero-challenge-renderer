#pragma once

typedef enum {
  OP_LINE,
  OP_OVAL,
  OP_AND,
  OP_OR,
  OP_RET,
  OP_RET_CONST,
} Op;

typedef struct {
  Op op;
  union {
    struct { uint16_t index; } line;
    struct { uint16_t index; bool outside; } oval;
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

// (ax + by + c) ** 2 + (dx + ey + f) ** 2 <= 1
//
// or, if `outside`, then instead >= 1

typedef struct {
  float a;
  float b;
  float c;
  float d;
  float e;
  float f;
} Oval;

typedef struct {
  size_t code_len;
  Inst * code;
  size_t line_len;
  Line * line;
  size_t oval_len;
  Oval * oval;
} Prog;

// PRECONDITIONS:
//
// - `prog` must be a valid program
// - `resolution` must be a power of two between 256 and 8192
// - ...

void render(
    Prog * prog,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    size_t resolution,
    uint8_t image[resolution][resolution]
  );
