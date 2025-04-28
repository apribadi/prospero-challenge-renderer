#!/usr/bin/env python3

import sys

class Var(int):
    __match_args__ = ("as_var",)

    def __repr__(self):
        return f"{int(self)}"

    @property
    def as_var(self):
        return self

# Read input from stdin and generate instructions line by line

code = []

for line in sys.stdin:
    if line.startswith("#"):
        continue
    [_, op, *args] = line.split()
    match op:
        case "var-x" | "var-y":
            code.append((op,))
        case "const":
            code.append((op, float(args[0])))
        case "add" | "sub" | "mul" | "max" | "min" | "neg" | "square" | "sqrt":
            code.append((op, *(Var(int(arg.lstrip("_"), 16)) for arg in args)))
        case _:
            raise RuntimeError("cant't parse input instruction: {line}")

code.append(("le_const", Var(len(code) - 1), 0.0))
code.append(("ret", Var(len(code) - 1)))

def push(x, y):
    n = len(x)
    x.append(y)
    return n

def emit(out, cse, ins):
    if ins in cse:
        return cse[ins]
    i = Var(push(out, ins))
    cse[ins] = i
    return i

def simplify(out, cse, ins):
    match ins:
        case "var-x",:
            return emit(out, cse, ("affine", 1.0, 0.0, 0.0))
        case "var-y",:
            return emit(out, cse, ("affine", 0.0, 1.0, 0.0))
        case "const", float(_):
            return emit(out, cse, ins)
        case op, Var(x):
            match (op, out[x]):
                case "neg", ("affine", a, b, c):
                    return emit(out, cse, ("affine", - a, - b, - c))
                case "square", ("affine", a, b, c):
                    return emit(out, cse, ("square_affine", a, b, c))
                case _:
                    return emit(out, cse, (op, x))
        case op, Var(x), float(t):
            match (op, out[x]):
                case "le_const", ("affine", a, b, c):
                    return emit(out, cse, ("line", a + 0, b + 0, - t + c + 0))
                case "le_const", ("sqrt", x):
                    if t < 0.0:
                        return emit(out, cse, ("false",))
                    return simplify(out, cse, ("le_const_sqrt", x, t))
                case "le_const", ("const", x):
                    return emit(out, cse, (("true",) if x <= t else ("false",)))
                case "le_const", ("neg", x):
                    return simplify(out, cse, ("ge_const", x, - t))
                case "le_const", ("add_const", x, c):
                    return simplify(out, cse, ("le_const", x, t - c))
                case "le_const", ("min", x, y):
                    x = simplify(out, cse, ("le_const", x, t))
                    y = simplify(out, cse, ("le_const", y, t))
                    return simplify(out, cse, ("or", x, y))
                case "le_const", ("max", x, y):
                    x = simplify(out, cse, ("le_const", x, t))
                    y = simplify(out, cse, ("le_const", y, t))
                    return simplify(out, cse, ("and", x, y))
                case "le_const_sqrt", ("add_square_affine", a, b, c, d, e, f):
                    return emit(out, cse, ("ellipse", a / t + 0, b / t + 0, c / t + 0, d / t + 0, e / t + 0, f / t + 0, False))
                case "ge_const", ("affine", a, b, c):
                    return emit(out, cse, ("line", - a + 0, - b + 0, t - c + 0))
                case "ge_const", ("sqrt", x):
                    if t <= 0.0:
                        return emit(out, cse, ("true",))
                    return simplify(out, cse, ("ge_const_sqrt", x, t))
                case "ge_const", ("add_const", x, c):
                    return simplify(out, cse, ("ge_const", x, t - c))
                case "ge_const", ("min", x, y):
                    x = simplify(out, cse, ("ge_const", x, t))
                    y = simplify(out, cse, ("ge_const", y, t))
                    return simplify(out, cse, ("and", x, y))
                case "ge_const", ("max", x, y):
                    x = simplify(out, cse, ("ge_const", x, t))
                    y = simplify(out, cse, ("ge_const", y, t))
                    return simplify(out, cse, ("or", x, y))
                case "ge_const_sqrt", ("add_square_affine", a, b, c, d, e, f):
                    return emit(out, cse, ("ellipse", a / t + 0, b / t + 0, c / t + 0, d / t + 0, e / t + 0, f / t + 0, True))
                case _:
                    return emit(out, cse, (op, x, t))
        case op, Var(x), Var(y):
            match (op, out[x], out[y]):
                case "add", ("affine", a, b, c), ("affine", d, e, f):
                    return emit(out, cse, ("affine", a + d, b + e, c + f))
                case "add", ("affine", a, b, c), ("const", d):
                    return emit(out, cse, ("affine", a, b, c + d))
                case "add", ("const", d), ("affine", a, b, c):
                    return emit(out, cse, ("affine", a, b, c + d))
                case "add", ("square_affine", a, b, c), ("square_affine", d, e, f):
                    return emit(out, cse, ("add_square_affine", a, b, c, d, e, f))
                case "sub", ("affine", a, b, c), ("affine", d, e, f):
                    return emit(out, cse, ("affine", a - d, b - e, c - f))
                case "sub", ("affine", a, b, c), ("const", d):
                    return emit(out, cse, ("affine", a, b, c - d))
                case "sub", ("const", d), ("affine", a, b, c):
                    return emit(out, cse, ("affine", - a, - b, - c + d))
                case "sub", ("const", x), _:
                    return emit(out, cse, ("neg", emit(out, cse, ("add_const", y, - x))))
                case "sub", _, ("const", y):
                    return emit(out, cse, ("add_const", x, - y))
                case "mul", ("affine", a, b, c), ("const", d):
                    return emit(out, cse, ("affine", a * d, b * d, c * d))
                case "mul", ("const", d), ("affine", a, b, c):
                    return emit(out, cse, ("affine", a * d, b * d, c * d))
                case "and", _, _:
                    return emit(out, cse, ("and", min(x, y), max(x, y)))
                case "or", _, ("false",):
                    return x
                case "or", _, _:
                    return emit(out, cse, ("or", min(x, y), max(x, y)))
                case _:
                    return emit(out, cse, (op, x, y))
        case _:
            raise RuntimeError("can't simplify instruction: {ins}")

def substitute_vars(ins, f):
    out = []
    for x in ins:
        if isinstance(x, Var):
            out.append(f(x))
        else:
            out.append(x)
    return tuple(out)

# Simplification and Common Subexpression Elimination

out = []
cse = {}
map = [] # old var -> new var

for ins in code:
    ins = substitute_vars(ins, lambda i: map[i])
    map.append(simplify(out, cse, ins))

code = out

# Dead Code Elimination

used = [False for _ in code]
used[-1] = True

for i, ins in reversed(list(enumerate(code))):
    if used[i]:
        for x in ins:
            if isinstance(x, Var):
                used[x] = True

out = []
map = [] # old var -> new var

for i, ins in enumerate(code):
    if used[i]:
        ins = substitute_vars(ins, lambda i: map[i])
        map.append(Var(push(out, ins)))
    else:
        map.append(None)

code = out

line = []
ellipse = []

print(f"static size_t PROSPERO_CODE_LEN = {len(code)};")
print(f"")
print(f"static Inst PROSPERO_CODE[{len(code)}] = {{")

for i, ins in enumerate(code):
    match ins:
        case "line", a, b, c:
            k = push(line, (a, b, c))
            print(f"  [{i}] = {{ OP_LINE, .line = {{ {k} }} }},")
        case "ellipse", a, b, c, d, e, f, outside:
            k = push(ellipse, (a, b, c, d, e, f))
            o = "true" if outside else "false"
            print(f"  [{i}] = {{ OP_ELLIPSE, .ellipse = {{ {k}, {o} }} }},")
        case "and", x, y:
            print(f"  [{i}] = {{ OP_AND, .and = {{ {x}, {y} }} }},")
        case "or", x, y:
            print(f"  [{i}] = {{ OP_OR, .or = {{ {x}, {y} }} }},")
        case "ret", x:
            print(f"  [{i}] = {{ OP_RET, .ret = {{ {x} }} }}")
        case _:
            raise RuntimeError(f"can't lower instruction: {ins}")

print(f"}};")
print(f"")
print(f"static Line PROSPERO_LINE[{len(line)}] = {{")

for (a, b, c) in line:
    print(f"  {{ {a:+.9e}f, {b:+.9e}f, {c:+.9e}f }} ,")

print(f"}};")
print(f"")
print(f"static Ellipse PROSPERO_ELLIPSE[{len(ellipse)}] = {{")

for (a, b, c, d, e, f) in ellipse:
    print(f"  {{ {a:+.9e}f, {b:+.9e}f, {c:+.9e}f, {d:+.9e}f, {e:+.9e}f, {f:+.9e}f }},")

print(f"}};")
print(f"")
print(f"static Geometry PROSPERO_GEOMETRY = {{")
print(f"  .line = PROSPERO_LINE,")
print(f"  .ellipse = PROSPERO_ELLIPSE,")
print(f"}};")
