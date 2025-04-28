# Synopsis

This is a renderer for the [Prospero Challenge](https://www.mattkeeter.com/projects/prospero/).

To run, do `make` and `./pcr`.

The code and build rules are platform specific and will essentially only work
on an Apple Mac with an M-series CPU. Also, you'll need to have LLVM installed
via Homebrew for OpenMP support. 

The platform limitations are not fundamental and it should be straightforward
to port the code.

This implementation is much faster than any other submission (as of late April
2025) running on a CPU.

Highlights:

- Given a preprocessed bytecode representation of the program, the renderer has
  the following performance numbers in milliseconds per frame, sustained, on an
  M1 Macbook Air running on either four or eight cores.

  | Resolution | 4 Firestorm     | 4 Firestorm + 4 Icestorm |
  | ---------- | --------------- | ------------------------ |
  | 512        | 0.49 ms / frame | 0.43 ms / frame          |
  | 1024       | 0.58 ms / frame | 0.48 ms / frame          |
  | 2048       | 0.97 ms / frame | 0.75 ms / frame          |
  | 4096       | 1.45 ms / frame | 1.16 ms / frame          |

  On this machine with eight cores and a 1024 resolution, it is approximately
  5x faster than Janos Meny's renderer (the next fastest CPU renderer) and
  approximately 15x faster than Fidget in JIT mode.

- Propagating the outer inequality `... <= 0` inwards generates more boolean
  expressions that are constant in subregions. This allows us to prune strictly
  more expressions than the min/max interval optimization.

- An `affine a b c -> a * x_coord + b * y_coord + c` instruction conveniently
  groups together other operations.

- We recursively specialize the bytecode program to 16 subregions at a time.

- A vectorized forward analysis pass propagates intervals for float variables,
  and propagates constant values and variable equivalences for boolean
  variables.

- A sparse (non-vectorized) backward pass extracts instructions, doing work
  that scales with the number of instructions in the specialized program.

- The code is manually vectorized with Neon intrinsics.

> TODO [^1]: Target other platforms. Using AVX2 shouldn't require any
> interesting changes, but for AVX-512 we'd want to take advantage of bitmask
> registers.

# Bytecode Preprocessing

We use a Python script `preprocess.py` to transform the provided instruction
stream into an array of own bytecode instructions (jankily represented as a C
array literal).

The target instruction stream has the following operations. A variable
parameter `x` or `y` refers to the output of a previous instruction, while an
immediate parameter `a`, `b`, or `c` is a constant value embedded in the
instruction itself.

```
AFFINE a : float imm, b : float imm, c : float imm -> float
  a * x_coord + b * y_coord + c

HYPOT2 x : float var, y : float var -> float
  x * x + y * y

LE_CONST x : float var, a : float imm -> bool
  x <= a

GE_CONST x : float var, a : float imm -> bool
  x >= a

AND x : bool var, y : bool var -> bool
  x & y

OR x : bool var, y : bool var -> bool
  x | y

RET x : bool var
  return x
  
RET_CONST a : bool imm
  return a
```

The first thing to notice is that we've included boolean operations. This is
motivated by the observation that the original specification has an implicit
`... <= 0.0` at the end of the program. Then the transformations

```
min(x, y) <= t --> (x <= t) or (y <= t)
```

and

```
max(x, y) <= t --> (x <= t) and (y <= t)
```

allow us to push the inequalities into the expression DAG.

How is this helpful? It allows interval analysis to discover more constant
expressions, and in fact we're able to prune strictly more expressions than
with the corresponding min/max simplifications.

Consider a case where we're able to simplify a `min`/`max`. For example,
(really WLOG) suppose that `min(x, y) <= t` where `a <= x <= b`, `c <= y <= d`,
and `b <= c`. Then `min(x, y) <= t` simplifies to just `x <= t`. However, when
`t` is a constant, then either one or both of `x <= t` and `y <= t` must be
constant, so `(x <= t) or (y <= t)` is either just `x <= t` or itself a
constant.

An example of where we're able to do strictly more simplification is `min(x, y)
<= 0` where `-0.5 <= x <= 0.5`, `0.25 <= y <= 1.25`. Because `y <= 0` is false,
we can simplify `(x <= 0) or (y <= 0)` to just `x <= 0`.

The instructions `AFFINE` and `HYPOT2` allow us to combine some operations. I
wanted to be forced to handle float variables, so I didn't simplify the code
all the way down to just half planes and axis aligned ellipses.

> TODO: Consider wider boolean operations. For instance, 3-ary ORs and ANDs
> would approximately halve the number of those instructions we need.

> TODO: The bytecode preprocessing is (relatively) slow because it's a Python
> script. We could implement a high performance version, as the algorithm
> doesn't do anything complicated.
>
> ```
> fun optimize(Array[Instrucion]) -> Array[Instruction]
> ```

> TODO: Our program is unsafe if given invalid code. We ought to check the
> precondition that the program is valid, for an appropriate definition of
> valid.
>
> ```
> fun is_valid(Array[Instruction]) -> bool
> ```

# Specialization to Subregions

When rendering a region, we can cut it up into 16 subregions and simplify the
program for each subregion using interval arithmetic and constant propagation.

One important trick is that we track when boolean variables are equal to each
other, linking to as old an instruction as possible.

```
x0 = ...
x1 = ...      (constant false)
x2 = x0 or x1 (equal to x0)
x3 = ...      (constant false)
x4 = x2 or x3 (equal to x0)
```

The logic to analyze the last instruction in that example is something like

```
info[x4].is_false = info[x2].is_false && info[x3].is_false
info[x4].is_true = info[x2].is_true || info[x3].is_true
info[x4].link = x4
info[x4].link = info[x2].is_false ? info[x3].link : info[x4].link
info[x4].link = info[x3].is_false ? info[x2].link : info[x4].link
```

which is easy to vectorize.

Then when we traverse backward to extract live instructions, we only need to
do one extra hop to find the argument that we need, instead of arbitrarily many
extra hops.

Also, in the backward traversal we use a priority queue to determine which
instruction to inspect next. I tried a few data structures, but a boring
implicit binary heap performed the best. (I really thought that a radix heap
had a shot at being better.)

> TODO: Our parallelism is coarse grained, and will not optimally use a machine
> with many cores.

> TODO: Handle resolutions that aren't powers of two.

> TODO: Handle images that aren't square.

# Evaluating Pixels

Our actual pixel-by-pixel evaluation doesn't do too much that's interesting.

We evaluate 256 pixels at a time, so the instruction dispatch overhead
shouldn't matter too much, but we still use an efficient tail-calling scheme.
The program is already very much not platform independent, so we don't worry
too much about whether tail-call elimination will be possible.

Another tidbit is that a single ARM Neon instruction can load or store four
vectors at once, so it's useful to work at a 512-bit granularity.

Also, the fact that we embed immediate values in some instructions reduces
memory traffic over just having a load-constant instruction, because we can
extract the immediate and then broadcast it to many lanes.

> TODO: If the program remains large at a leaf tile (which happens, for
> instance, when rendering at low resolution), we ought to take steps to reduce
> memory usage, either by evaluating fewer pixels at a time or by doing
> register allocation.

[^1]: The TODOs are mostly meant in the sense of "exercise left to the reader".
