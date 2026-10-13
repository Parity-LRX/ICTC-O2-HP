"""Single-channel second input: fused paths and one channel reduction per entry.

Sparse coupling terms are compile-time metadata. The entry axis holds every
forward/left accumulator in one program, letting the compiler reuse loads from
paths sharing an input entry. Right adjoints sum paths before reducing channels.
"""

import triton
import triton.language as tl


@triton.jit
def _fused_forward(x, y, w, out, N: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr,
                   U: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
                   FIRST: tl.constexpr, SECOND: tl.constexpr, PATHS: tl.constexpr,
                   B: tl.constexpr, E: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    row, u = q // U, q % U
    active = row < N
    entry = tl.arange(0, E)[:, None]
    acc0 = tl.full((E, B), 0, tl.float32)
    acc1 = tl.full((E, B), 0, tl.float32)
    for p in tl.static_range(len(PATHS.value)):
        k = PATHS.value[p][2]
        xo = FIRST.value[PATHS.value[p][0]][0]
        dx = FIRST.value[PATHS.value[p][0]][1]
        yo = SECOND.value[PATHS.value[p][1]][0]
        wi = p * U + u
        if not SHARED:
            wi += row * WNUM
        weight = tl.load(w + wi, active, 0)
        value0 = tl.full((B,), 0, tl.float32)
        value1 = tl.full((B,), 0, tl.float32)
        for term in tl.static_range(len(PATHS.value[p][3])):
            a = PATHS.value[p][3][term][0]
            b = PATHS.value[p][3][term][1]
            c = PATHS.value[p][3][term][2]
            coefficient = PATHS.value[p][3][term][3]
            left = tl.load(x + row * WX + xo + u * dx + a, active, 0)
            right = tl.load(y + row * WY + yo + b, active, 0)
            contribution = coefficient * left * right
            if c == 0:
                value0 += contribution
            else:
                value1 += contribution
        acc0 += tl.where(entry == k, (weight * value0)[None, :], 0)
        acc1 += tl.where(entry == k, (weight * value1)[None, :], 0)
    for k in tl.static_range(len(FIRST.value)):
        offset = FIRST.value[k][0]
        dim = FIRST.value[k][1]
        value0 = tl.sum(tl.where(entry == k, acc0, 0), 0)
        tl.store(out + row * WX + offset + u * dim, value0, active)
        if dim == 2:
            value1 = tl.sum(tl.where(entry == k, acc1, 0), 0)
            tl.store(out + row * WX + offset + u * dim + 1, value1, active)


@triton.jit
def _fused_left(y, w, g, out, N: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr,
                U: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
                FIRST: tl.constexpr, SECOND: tl.constexpr, PATHS: tl.constexpr,
                B: tl.constexpr, E: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    row, u = q // U, q % U
    active = row < N
    entry = tl.arange(0, E)[:, None]
    acc0 = tl.full((E, B), 0, tl.float32)
    acc1 = tl.full((E, B), 0, tl.float32)
    for p in tl.static_range(len(PATHS.value)):
        i = PATHS.value[p][0]
        go = FIRST.value[PATHS.value[p][2]][0]
        dg = FIRST.value[PATHS.value[p][2]][1]
        yo = SECOND.value[PATHS.value[p][1]][0]
        wi = p * U + u
        if not SHARED:
            wi += row * WNUM
        weight = tl.load(w + wi, active, 0)
        value0 = tl.full((B,), 0, tl.float32)
        value1 = tl.full((B,), 0, tl.float32)
        for term in tl.static_range(len(PATHS.value[p][3])):
            a = PATHS.value[p][3][term][0]
            b = PATHS.value[p][3][term][1]
            c = PATHS.value[p][3][term][2]
            coefficient = PATHS.value[p][3][term][3]
            cotangent = tl.load(g + row * WX + go + u * dg + c, active, 0)
            right = tl.load(y + row * WY + yo + b, active, 0)
            contribution = coefficient * cotangent * right
            if a == 0:
                value0 += contribution
            else:
                value1 += contribution
        acc0 += tl.where(entry == i, (weight * value0)[None, :], 0)
        acc1 += tl.where(entry == i, (weight * value1)[None, :], 0)
    for i in tl.static_range(len(FIRST.value)):
        offset = FIRST.value[i][0]
        dim = FIRST.value[i][1]
        value0 = tl.sum(tl.where(entry == i, acc0, 0), 0)
        tl.store(out + row * WX + offset + u * dim, value0, active)
        if dim == 2:
            value1 = tl.sum(tl.where(entry == i, acc1, 0), 0)
            tl.store(out + row * WX + offset + u * dim + 1, value1, active)


@triton.jit
def _right(x, w, g, out, N: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr,
           U: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
           FIRST: tl.constexpr, SECOND: tl.constexpr, PATHS: tl.constexpr,
           J: tl.constexpr, SELECTED: tl.constexpr, B: tl.constexpr, BU: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    u = tl.arange(0, BU)
    active = (row[:, None] < N) & (u[None, :] < U)
    acc0 = tl.full((B, BU), 0, tl.float32)
    acc1 = tl.full((B, BU), 0, tl.float32)
    for index in tl.static_range(len(SELECTED.value)):
        p = SELECTED.value[index]
        xo = FIRST.value[PATHS.value[SELECTED.value[index]][0]][0]
        dx = FIRST.value[PATHS.value[SELECTED.value[index]][0]][1]
        go = FIRST.value[PATHS.value[SELECTED.value[index]][2]][0]
        dg = FIRST.value[PATHS.value[SELECTED.value[index]][2]][1]
        # Keep the row axis in shared-weight pointers as well as the mask.
        wi = p * U + u[None, :] + row[:, None] * 0
        if not SHARED:
            wi += row[:, None] * WNUM
        weight = tl.load(w + wi, active, 0)
        value0 = tl.full((B, BU), 0, tl.float32)
        value1 = tl.full((B, BU), 0, tl.float32)
        for term in tl.static_range(len(PATHS.value[SELECTED.value[index]][3])):
            a = PATHS.value[SELECTED.value[index]][3][term][0]
            b = PATHS.value[SELECTED.value[index]][3][term][1]
            c = PATHS.value[SELECTED.value[index]][3][term][2]
            coefficient = PATHS.value[SELECTED.value[index]][3][term][3]
            left = tl.load(x + row[:, None] * WX + xo + u[None, :] * dx + a, active, 0)
            cotangent = tl.load(g + row[:, None] * WX + go + u[None, :] * dg + c, active, 0)
            contribution = coefficient * left * cotangent
            if b == 0:
                value0 += contribution
            else:
                value1 += contribution
        acc0 += weight * value0
        acc1 += weight * value1
    offset = SECOND.value[J][0]
    dim = SECOND.value[J][1]
    tl.store(out + row * WY + offset, tl.sum(acc0, 1), row < N)
    if dim == 2:
        tl.store(out + row * WY + offset + 1, tl.sum(acc1, 1), row < N)


@triton.jit
def _weights(x, y, g, out, N: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr,
              U: tl.constexpr, WNUM: tl.constexpr,
              FIRST: tl.constexpr, SECOND: tl.constexpr, PATHS: tl.constexpr,
              SELECTED: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    row, u = q // U, q % U
    active = row < N
    # One program handles all paths to an output entry. Static coefficients
    # remove zero terms and reuse cotangents and feature loads across paths.
    for index in tl.static_range(len(SELECTED.value)):
        p = SELECTED.value[index]
        xo = FIRST.value[PATHS.value[SELECTED.value[index]][0]][0]
        dx = FIRST.value[PATHS.value[SELECTED.value[index]][0]][1]
        yo = SECOND.value[PATHS.value[SELECTED.value[index]][1]][0]
        dy = SECOND.value[PATHS.value[SELECTED.value[index]][1]][1]
        go = FIRST.value[PATHS.value[SELECTED.value[index]][2]][0]
        dg = FIRST.value[PATHS.value[SELECTED.value[index]][2]][1]
        value0 = tl.full((B,), 0, tl.float32)
        value1 = tl.full((B,), 0, tl.float32)
        for term in tl.static_range(len(PATHS.value[SELECTED.value[index]][3])):
            a = PATHS.value[SELECTED.value[index]][3][term][0]
            b = PATHS.value[SELECTED.value[index]][3][term][1]
            c = PATHS.value[SELECTED.value[index]][3][term][2]
            coefficient = PATHS.value[SELECTED.value[index]][3][term][3]
            left = tl.load(x + row * WX + xo + u * dx + a, active, 0)
            cotangent = tl.load(g + row * WX + go + u * dg + c, active, 0)
            contribution = left * (coefficient * cotangent)
            if b == 0:
                value0 += contribution
            else:
                value1 += contribution
        angular0 = tl.load(y + row * WY + yo, active, 0)
        value = value0 * angular0
        if dy == 2:
            angular1 = tl.load(y + row * WY + yo + 1, active, 0)
            value += value1 * angular1
        tl.store(out + row * WNUM + p * U + u, value, active)
