"""Independent O(2) algebra for native-cost-v1.

Dependency-closed extraction of the existing project's real-basis algebra,
retaining its signs and normalization for reference parity. No CHORUS imports,
O(3) carrier, geometry, projection, or lifting. Provenance is recorded in the
contract; the old source is only loaded by the separate verification process.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple
import torch
O2IrrepKey = Tuple[str, int]

def so2_irrep_dim(m: int) -> int:
    m = int(m)
    if m < 0:
        raise ValueError(f"frequency m must be >= 0, got {m}")
    return 1 if m == 0 else 2

def _counts_list_2d(L: int) -> List[Tuple[int, int]]:
    """All (a, b) with a + b = L in deterministic order."""
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    return [(a, L - a) for a in range(L + 1)]

def _double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out

def _gaussian_moment(n: int) -> float:
    if n % 2 == 1:
        return 0.0
    return float(_double_factorial(n - 1))

@lru_cache(maxsize=None)
def _gram_gaussian_2d(L: int) -> torch.Tensor:
    """
    O(2)-invariant Gram matrix on Sym^L in the monomial basis x^a y^b.
    """
    counts = _counts_list_2d(L)
    D = len(counts)
    G = torch.zeros(D, D, dtype=torch.float64)
    for i, (a, b) in enumerate(counts):
        for j, (a2, b2) in enumerate(counts):
            G[i, j] = _gaussian_moment(a + a2) * _gaussian_moment(b + b2)
    return G

@lru_cache(maxsize=None)
def _build_r2k_lift_2d(L_src: int, k: int) -> torch.Tensor:
    """
    Multiply by (x^2 + y^2)^k in the 2D monomial basis.
    """
    L_src = int(L_src)
    k = int(k)
    if L_src < 0 or k < 0:
        raise ValueError(f"L_src and k must be >= 0, got {L_src}, {k}")
    L_dst = L_src + 2 * k
    src = _counts_list_2d(L_src)
    dst = _counts_list_2d(L_dst)
    idx_dst = {t: i for i, t in enumerate(dst)}
    M = torch.zeros(len(dst), len(src), dtype=torch.float64)
    for col, (a, b) in enumerate(src):
        for q in range(k + 1):
            a_dst = a + 2 * q
            b_dst = b + 2 * (k - q)
            M[idx_dst[(a_dst, b_dst)], col] += float(math.comb(k, q))
    return M.contiguous()

def _fix_vector_sign(v: torch.Tensor) -> torch.Tensor:
    out = v.clone()
    for idx in range(out.numel()):
        val = float(out[idx].item())
        if abs(val) > 1e-12:
            if val < 0.0:
                out = -out
            break
    return out

@lru_cache(maxsize=None)
def _harmonic_basis_cpu_f64_2d(m: int) -> torch.Tensor:
    """
    Harmonic basis in monomial coefficients for the 2D real basis:
      m = 0: [1]
      m > 0: [Re((x+i y)^m), Im((x+i y)^m)] normalized under the Gaussian Gram.
    """
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if m == 0:
        return torch.ones(1, 1, dtype=torch.float64)

    counts = _counts_list_2d(m)
    cos_vec = torch.zeros(len(counts), dtype=torch.float64)
    sin_vec = torch.zeros(len(counts), dtype=torch.float64)
    for idx, (a, b) in enumerate(counts):
        coef = float(math.comb(m, a))
        phase = b % 4
        if phase == 0:
            cos_val = 1.0
            sin_val = 0.0
        elif phase == 1:
            cos_val = 0.0
            sin_val = 1.0
        elif phase == 2:
            cos_val = -1.0
            sin_val = 0.0
        else:
            cos_val = 0.0
            sin_val = -1.0
        cos_vec[idx] = coef * cos_val
        sin_vec[idx] = coef * sin_val

    G = _gram_gaussian_2d(m)

    cos_norm = torch.sqrt(torch.clamp(cos_vec @ (G @ cos_vec), min=1e-30))
    cos_vec = cos_vec / cos_norm

    proj = cos_vec @ (G @ sin_vec)
    sin_vec = sin_vec - proj * cos_vec
    sin_norm = torch.sqrt(torch.clamp(sin_vec @ (G @ sin_vec), min=1e-30))
    sin_vec = sin_vec / sin_norm

    cos_vec = _fix_vector_sign(cos_vec)
    sin_vec = _fix_vector_sign(sin_vec)
    if (m % 2) == 0:
        cos_vec = -cos_vec
    return torch.stack([cos_vec, sin_vec], dim=-1).contiguous()

def _harmonic_basis_t_2d(m: int, device=None, dtype=None) -> torch.Tensor:
    return _harmonic_basis_cpu_f64_2d(int(m)).to(device=device, dtype=dtype)

@dataclass(frozen=True)
class HarmonicProjectors2D:
    Lmax: int
    P: Dict[Tuple[int, int], torch.Tensor]  # (L, m) -> (dim(m), Dsym(L))

@lru_cache(maxsize=None)
def build_harmonic_projectors_2d(Lmax: int) -> HarmonicProjectors2D:
    P: Dict[Tuple[int, int], torch.Tensor] = {}
    Lmax = int(Lmax)
    for L in range(Lmax + 1):
        G_L = _gram_gaussian_2d(L)
        for k in range(L // 2 + 1):
            m = L - 2 * k
            B_m = _harmonic_basis_t_2d(m, dtype=torch.float64)
            M = _build_r2k_lift_2d(m, k)
            V = (M @ B_m).contiguous()
            G = V.T @ G_L @ V
            P[(L, m)] = torch.linalg.solve(G, V.T @ G_L).contiguous()
    return HarmonicProjectors2D(Lmax=Lmax, P=P)

def o2_valid_output_keys(
    key1: str | O2IrrepKey,
    key2: str | O2IrrepKey,
) -> List[O2IrrepKey]:
    """Return every O(2) irrep admitted by ``key1 tensor key2``."""
    return _o2_valid_output_keys(
        _normalize_o2_irrep_key(key1),
        _normalize_o2_irrep_key(key2),
    )

def o2_cg_tensor(
    key1: str | O2IrrepKey,
    key2: str | O2IrrepKey,
    key3: str | O2IrrepKey,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a real Cartesian O(2) Clebsch--Gordan tensor."""
    out = _build_cg_tensor_o2(
        _normalize_o2_irrep_key(key1),
        _normalize_o2_irrep_key(key2),
        _normalize_o2_irrep_key(key3),
    )
    if dtype is not None or device is not None:
        out = out.to(dtype=dtype, device=device)
    return out

def o2_representation(
    g: torch.Tensor,
    key: str | O2IrrepKey,
) -> torch.Tensor:
    """Representation matrix of a batched O(2) matrix.

    ``g`` has shape ``(..., 2, 2)`` and acts on column components.  Frequency
    irreps use the real ``[cos(m theta), sin(m theta)]`` convention.  A
    reflection acts only on this spatial representation; it does not conjugate
    any latent U(1) axis carried by the caller.
    """
    key_n = _normalize_o2_irrep_key(key)
    if g.shape[-2:] != (2, 2):
        raise ValueError(f"g must end with (2, 2), got {tuple(g.shape)}")
    det = torch.linalg.det(g)
    if key_n[0] == "scalar":
        value = torch.ones_like(det) if key_n[1] == 1 else det
        return value.unsqueeze(-1).unsqueeze(-1)

    m = int(key_n[1])
    theta = torch.atan2(g[..., 1, 0], g[..., 0, 0])
    c = torch.cos(float(m) * theta)
    s = torch.sin(float(m) * theta)
    return torch.stack(
        (
            torch.stack((c, -det * s), dim=-1),
            torch.stack((s, det * c), dim=-1),
        ),
        dim=-2,
    )

def _normalize_o2_irrep_key(key: str | O2IrrepKey) -> O2IrrepKey:
    if isinstance(key, tuple):
        tag, value = key
        tag = str(tag).strip().lower()
        value_i = int(value)
        if tag == "scalar" and value_i in (-1, 1):
            return ("scalar", value_i)
        if tag == "freq" and value_i >= 1:
            return ("freq", value_i)
        raise ValueError(f"Invalid O(2) key {key!r}")

    tok = str(key).strip().lower()
    if tok == "0e":
        return ("scalar", 1)
    if tok == "0o":
        return ("scalar", -1)
    if tok.isdigit() and int(tok) >= 1:
        return ("freq", int(tok))
    raise ValueError(f"Invalid O(2) irrep token {key!r}")

def _o2_irrep_dim(key: O2IrrepKey) -> int:
    key = _normalize_o2_irrep_key(key)
    return 1 if key[0] == "scalar" else 2

@lru_cache(maxsize=None)
def _build_poly_mult_matrix_2d(m1: int, m2: int, M: int) -> torch.Tensor:
    counts1 = _counts_list_2d(m1)
    counts2 = _counts_list_2d(m2)
    countsM = _counts_list_2d(M)
    idxM = {t: i for i, t in enumerate(countsM)}
    D1, D2, DM = len(counts1), len(counts2), len(countsM)
    out = torch.zeros(DM, D1 * D2, dtype=torch.float64)
    for i, c1 in enumerate(counts1):
        for j, c2 in enumerate(counts2):
            k = idxM[(c1[0] + c2[0], c1[1] + c2[1])]
            out[k, i * D2 + j] = 1.0
    return out.contiguous()

@lru_cache(maxsize=None)
def build_cg_tensor_so2(m1: int, m2: int, m3: int) -> torch.Tensor:
    """
    Symmetric-product SO(2) coupling in the real [cos, sin] basis.
    """
    m1 = int(m1)
    m2 = int(m2)
    m3 = int(m3)
    if min(m1, m2, m3) < 0:
        raise ValueError(f"m1, m2, m3 must be >= 0, got {(m1, m2, m3)}")
    if m3 > m1 + m2 or ((m1 + m2 + m3) % 2 == 1):
        return torch.zeros(so2_irrep_dim(m1), so2_irrep_dim(m2), so2_irrep_dim(m3), dtype=torch.float64)

    M = m1 + m2
    proj = build_harmonic_projectors_2d(M)
    P = proj.P[(M, m3)]
    B1 = _harmonic_basis_t_2d(m1, dtype=torch.float64)
    B2 = _harmonic_basis_t_2d(m2, dtype=torch.float64)
    M_poly = _build_poly_mult_matrix_2d(m1, m2, M)

    outer = torch.einsum("im,jn->ijmn", B1, B2)
    outer_flat = outer.reshape(B1.shape[0] * B2.shape[0], so2_irrep_dim(m1) * so2_irrep_dim(m2))
    tM = M_poly @ outer_flat
    c3 = P @ tM
    return c3.T.reshape(so2_irrep_dim(m1), so2_irrep_dim(m2), so2_irrep_dim(m3)).contiguous()

@lru_cache(maxsize=None)
def _build_o2_scalar_odd_cg(m: int) -> torch.Tensor:
    m = int(m)
    if m <= 0:
        raise ValueError(f"odd scalar coupling requires m>=1, got {m}")
    if so2_irrep_dim(m) != 2:
        raise AssertionError("odd scalar coupling only exists for 2D frequency blocks")
    C = torch.zeros(2, 2, 1, dtype=torch.float64)
    # Antisymmetric bilinear form; this is the O(2)-odd scalar branch.
    C[0, 1, 0] = 1.0
    C[1, 0, 0] = -1.0
    return C

@lru_cache(maxsize=None)
def _build_cg_tensor_o2(key1: O2IrrepKey, key2: O2IrrepKey, key3: O2IrrepKey) -> torch.Tensor:
    key1 = _normalize_o2_irrep_key(key1)
    key2 = _normalize_o2_irrep_key(key2)
    key3 = _normalize_o2_irrep_key(key3)

    d1 = _o2_irrep_dim(key1)
    d2 = _o2_irrep_dim(key2)
    d3 = _o2_irrep_dim(key3)

    if key1[0] == "scalar" and key2[0] == "scalar":
        if key3 == ("scalar", key1[1] * key2[1]):
            return torch.ones(d1, d2, d3, dtype=torch.float64)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    if key1[0] == "scalar" and key2[0] == "freq":
        if key3 == key2:
            # Tensoring the reflection-odd scalar with E_m is equivalent to
            # E_m, but the intertwiner is the complex structure J rather than
            # the identity.  Identity is only SO(2)-equivariant and fails for
            # reflections.  The CG convention stores C[input, output], hence
            # J^T below so that y_out = J y_in.
            if key1[1] == 1:
                intertwiner_t = torch.eye(2, dtype=torch.float64)
            else:
                J = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
                intertwiner_t = J.transpose(0, 1)
            return intertwiner_t.unsqueeze(0)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    if key1[0] == "freq" and key2[0] == "scalar":
        if key3 == key1:
            if key2[1] == 1:
                intertwiner_t = torch.eye(2, dtype=torch.float64)
            else:
                J = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
                intertwiner_t = J.transpose(0, 1)
            return intertwiner_t.unsqueeze(1)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    if key1[0] == "freq" and key2[0] == "freq":
        m1 = key1[1]
        m2 = key2[1]
        if key3[0] == "freq":
            return build_cg_tensor_so2(m1, m2, key3[1])
        if key3 == ("scalar", 1) and m1 == m2:
            return build_cg_tensor_so2(m1, m2, 0)
        if key3 == ("scalar", -1) and m1 == m2:
            return _build_o2_scalar_odd_cg(m1)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    return torch.zeros(d1, d2, d3, dtype=torch.float64)

def _o2_valid_output_keys(key1: O2IrrepKey, key2: O2IrrepKey) -> List[O2IrrepKey]:
    key1 = _normalize_o2_irrep_key(key1)
    key2 = _normalize_o2_irrep_key(key2)

    if key1[0] == "scalar" and key2[0] == "scalar":
        return [("scalar", key1[1] * key2[1])]

    if key1[0] == "scalar" and key2[0] == "freq":
        return [key2]
    if key1[0] == "freq" and key2[0] == "scalar":
        return [key1]

    m1 = key1[1]
    m2 = key2[1]
    outs: List[O2IrrepKey] = []
    if m1 == m2:
        outs.extend([("scalar", 1), ("scalar", -1)])
    else:
        outs.append(("freq", abs(m1 - m2)))
    outs.append(("freq", m1 + m2))
    dedup: List[O2IrrepKey] = []
    for key in outs:
        if key not in dedup:
            dedup.append(key)
    return dedup
