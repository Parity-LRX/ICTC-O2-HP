"""Sparse uvu kernels for a multi-channel second input."""

import triton
import triton.language as tl


@triton.jit
def _project(y, w, row, u, p, yo, dy, active, WY: tl.constexpr,
             U: tl.constexpr, V: tl.constexpr, WNUM: tl.constexpr,
             SHARED: tl.constexpr, B: tl.constexpr):
    # Small V stays in scalar-vector registers instead of introducing a
    # separate reduction layout into the fused entry accumulators.
    r0 = tl.full((B,), 0, tl.float32)
    r1 = tl.full((B,), 0, tl.float32)
    for v in tl.static_range(V):
        wi = p * U * V + u * V + v
        if not SHARED:
            wi += row * WNUM
        weight = tl.load(w + wi, active, 0)
        b0 = tl.load(y + row * WY + yo + v * dy, active, 0)
        b1 = tl.load(y + row * WY + yo + v * dy + 1, active & (dy == 2), 0)
        r0 += weight * b0
        r1 += weight * b1
    return r0, r1


@triton.jit
def _forward(x, y, w, out, N: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr,
             U: tl.constexpr, V: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
             FIRST: tl.constexpr, SECOND: tl.constexpr, PATHS: tl.constexpr,
             B: tl.constexpr, E: tl.constexpr, BV: tl.constexpr):
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
        dy = SECOND.value[PATHS.value[p][1]][1]
        r0, r1 = _project(y, w, row, u, p, yo, dy, active, WY, U, V, WNUM, SHARED, B)
        value0 = tl.full((B,), 0, tl.float32)
        value1 = tl.full((B,), 0, tl.float32)
        for term in tl.static_range(len(PATHS.value[p][3])):
            a = PATHS.value[p][3][term][0]
            b = PATHS.value[p][3][term][1]
            c = PATHS.value[p][3][term][2]
            coefficient = PATHS.value[p][3][term][3]
            left = tl.load(x + row * WX + xo + u * dx + a, active, 0)
            projected = r0 if b == 0 else r1
            contribution = coefficient * left * projected
            if c == 0:
                value0 += contribution
            else:
                value1 += contribution
        acc0 += tl.where(entry == k, value0[None, :], 0)
        acc1 += tl.where(entry == k, value1[None, :], 0)
    for k in tl.static_range(len(FIRST.value)):
        offset = FIRST.value[k][0]
        dim = FIRST.value[k][1]
        tl.store(out + row * WX + offset + u * dim,
                 tl.sum(tl.where(entry == k, acc0, 0), 0), active)
        if dim == 2:
            tl.store(out + row * WX + offset + u * dim + 1,
                     tl.sum(tl.where(entry == k, acc1, 0), 0), active)


@triton.jit
def _left(y, w, g, out, N: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr,
          U: tl.constexpr, V: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
          FIRST: tl.constexpr, SECOND: tl.constexpr, PATHS: tl.constexpr,
          B: tl.constexpr, E: tl.constexpr, BV: tl.constexpr):
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
        dy = SECOND.value[PATHS.value[p][1]][1]
        r0, r1 = _project(y, w, row, u, p, yo, dy, active, WY, U, V, WNUM, SHARED, B)
        value0 = tl.full((B,), 0, tl.float32)
        value1 = tl.full((B,), 0, tl.float32)
        for term in tl.static_range(len(PATHS.value[p][3])):
            a = PATHS.value[p][3][term][0]
            b = PATHS.value[p][3][term][1]
            c = PATHS.value[p][3][term][2]
            coefficient = PATHS.value[p][3][term][3]
            cotangent = tl.load(g + row * WX + go + u * dg + c, active, 0)
            projected = r0 if b == 0 else r1
            contribution = coefficient * cotangent * projected
            if a == 0:
                value0 += contribution
            else:
                value1 += contribution
        acc0 += tl.where(entry == i, value0[None, :], 0)
        acc1 += tl.where(entry == i, value1[None, :], 0)
    for i in tl.static_range(len(FIRST.value)):
        offset = FIRST.value[i][0]
        dim = FIRST.value[i][1]
        tl.store(out + row * WX + offset + u * dim,
                 tl.sum(tl.where(entry == i, acc0, 0), 0), active)
        if dim == 2:
            tl.store(out + row * WX + offset + u * dim + 1,
                     tl.sum(tl.where(entry == i, acc1, 0), 0), active)


@triton.jit
def _right(x, w, g, out, N: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr,
           U: tl.constexpr, V: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
           FIRST: tl.constexpr, SECOND: tl.constexpr, PATHS: tl.constexpr,
           J: tl.constexpr, SELECTED: tl.constexpr, B: tl.constexpr,
           BU: tl.constexpr, BV: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    u = tl.arange(0, BU)
    v = tl.program_id(1) * BV + tl.arange(0, BV)
    active = (row[:, None, None] < N) & (u[None, :, None] < U) & (v[None, None, :] < V)
    input_mask = (row[:, None] < N) & (u[None, :] < U)
    acc0 = tl.full((B, BU, BV), 0, tl.float32)
    acc1 = tl.full((B, BU, BV), 0, tl.float32)
    for index in tl.static_range(len(SELECTED.value)):
        p = SELECTED.value[index]
        xo = FIRST.value[PATHS.value[SELECTED.value[index]][0]][0]
        dx = FIRST.value[PATHS.value[SELECTED.value[index]][0]][1]
        go = FIRST.value[PATHS.value[SELECTED.value[index]][2]][0]
        dg = FIRST.value[PATHS.value[SELECTED.value[index]][2]][1]
        wi = p * U * V + u[None, :, None] * V + v[None, None, :] + row[:, None, None] * 0
        if not SHARED:
            wi += row[:, None, None] * WNUM
        weight = tl.load(w + wi, active, 0)
        value0 = tl.full((B, BU), 0, tl.float32)
        value1 = tl.full((B, BU), 0, tl.float32)
        for term in tl.static_range(len(PATHS.value[SELECTED.value[index]][3])):
            a = PATHS.value[SELECTED.value[index]][3][term][0]
            b = PATHS.value[SELECTED.value[index]][3][term][1]
            c = PATHS.value[SELECTED.value[index]][3][term][2]
            coefficient = PATHS.value[SELECTED.value[index]][3][term][3]
            left = tl.load(x + row[:, None] * WX + xo + u[None, :] * dx + a, input_mask, 0)
            cotangent = tl.load(g + row[:, None] * WX + go + u[None, :] * dg + c, input_mask, 0)
            contribution = coefficient * left * cotangent
            if b == 0:
                value0 += contribution
            else:
                value1 += contribution
        acc0 += weight * value0[:, :, None]
        acc1 += weight * value1[:, :, None]
    offset = SECOND.value[J][0]
    dim = SECOND.value[J][1]
    output_mask = (row[:, None] < N) & (v[None, :] < V)
    tl.store(out + row[:, None] * WY + offset + v[None, :] * dim, tl.sum(acc0, 1), output_mask)
    if dim == 2:
        tl.store(out + row[:, None] * WY + offset + v[None, :] * dim + 1,
                 tl.sum(acc1, 1), output_mask)
