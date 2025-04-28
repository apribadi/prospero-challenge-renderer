# Synopsis

This is a renderer for the [Prospero Challenge](https://www.mattkeeter.com/projects/prospero/).

To run, do `make` and `./pcr`.

You can limit the number of cores used by the program by instead running
`OMP_NUM_THREADS=$N ./pcr` with an appropriate value of `N`.

The code and build rules are platform specific and will essentially only work
on an Apple Mac with an M-series CPU. Also, you'll need to have LLVM installed
via Homebrew for OpenMP support. 

The platform limitations are not fundamental and it should be straightforward
to port the code.

This implementation is much faster than any other submission (as of late May
2025) running on a CPU.

# Highlights

- The renderer (when given a preprocessed bytecode representation of the
  program) can draw frames at the following rates, benchmarked on an M1 Macbook
  Air using four (Firestorm) cores.

  | resolution | ms/frame |
  | ---------- | -------- |
  | 512        | 0.219    |
  | 1024       | 0.284    |
  | 2048       | 0.402    |
  | 4096       | 0.836    |

  At a 1024 pixel resolution, it is approximately 8x faster than the next
  fastest CPU renderer [^1], and approximately 25x faster than Fidget in JIT
  mode [^2].

- In a preprocessing step, we propagate the outer inequality `... <= 0` inwards
  to discover which quantities should be interpreted as implicit boundaries.
  The boundaries turn out to be either straight lines or ellipses.

- We recursively specialize the bytecode program to 16 subregions at a time.
  We use interval analysis to generate boolean constants, and then propagate
  constants and variable equivalences. Constant propagation on boolean
  expressions allows us to prune strictly more subexpressions than the min/max
  interval optimization described elsewhere.

- In the specialization procedure, the forward analysis pass is vectorized, and
  the backward passes to extract optimized programs are sparse (i.e. scale
  with the number of instructions in the output program).

- The vectorized portions of the program are written with Arm Neon intrinsics.

> TODO [^3]: Target other platforms. An AVX2 version should do basically the
> same thing as the Neon version, but for AVX-512 we'd want to take advantage
> of mask registers.

# Bytecode Preprocessing

We use a Python script `preprocess.py` to transform the provided instruction
stream into our own bytecode instructions (we just output C code which we
compile into the end program, which is admittedly a bit hacky ...).

The target instruction stream has the following operations (see render.h):

```
LINE (a, b, c):
  ax + by + c <= 0

ELLIPSE (a, b, c, d, e, f, outside):
  if outside:
    (ax + by + c) ** 2 + (dx + ey + f) ** 2 >= 1
  else:
    (ax + by + c) ** 2 + (dx + ey + f) ** 2 <= 1

AND (x, y):
  x & y

OR (x, y):
  x | y

RET (x):
  return x
  
RET_CONST (a)
  return a
```

Here, the `LINE` and `ELLIPSE` instructions use the `x` and `y` coordinates of
the current pixel and produce a boolean value. The `AND` and `OR` instructions
operate on booleans produced by previous instructions. And the `RET` and
`RET_CONST` instructions terminate the program, returning either a value
produced by a previous instruction or a constant value, respectively.

We're able to make the program operate on boolean values because the original
specification has an implicit `... <= 0.0` at the end of the program. Then the
transformations

```
min(x, y) <= t --> (x <= t) | (y <= t)
max(x, y) <= t --> (x <= t) & (y <= t)
```

allow us to push the inequalities into the expression DAG.

The preprocessor itself is a relatively straightforward pipeline involving
local pattern matching, hash-based deduplication, and dead code elimination.

> TODO: Consider alternate representations for the boolean operations. For
> instance, 3-ary ORs and ANDs would approximately halve the number of those
> instructions we need. Or we could normalize into a form with arrays of
> operands.

> TODO: We could segregate the leaf `LINE` and `ELLIPSE` instructions, and
> compute them in separate dispatch-less loops. We already place their
> parameters in separate arrays in order to keep instructions compact.

> TODO: The bytecode preprocessing is (relatively) slow because it's a Python
> script. We could implement a high performance version.
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

Then when we traverse backwards to extract live instructions, we only need to
do one extra hop to find the argument that we need, instead of arbitrarily many
extra hops.

Also, in the backward traversal we use a priority queue to determine which
instruction to inspect next. A boring implicit binary heap turned out to
perform the best out of a few priority queue implementations.

> TODO: Handle resolutions that aren't powers of two.

> TODO: Handle images that aren't square.

# Evaluating Pixels

The interpreter loop that does our actual pixel-by-pixel evaluation doesn't do
too much that's interesting.

We evaluate 256 pixels at a time, so the instruction dispatch overhead
shouldn't matter too much, but we still use an efficient tail-calling scheme.
The program is already very much not platform independent, so we don't worry
too much about whether tail-call elimination will be possible.

> TODO: A funny idea for a more general version of this renderer would be to
> JIT-compile arithmetic subgraphs but keep the `UNION` and `INTERSECTION`
> instructions interpreted. Then the whole thing could be structured as a
> [direct-threaded interpreter](https://jilp.org/vol5/v5paper12.pdf) ...!

[^1]: Janos Meny's renderer <https://github.com/Janos95/prospero_vm/>, which I
    timed at 2.44 ms/frame on my machine.

[^2]: I timed Fidget <https://www.mattkeeter.com/projects/fidget> at 7.54
    ms/frame on my machine.

[^3]: The TODOs are mostly meant in the sense of "exercise left to the reader".
