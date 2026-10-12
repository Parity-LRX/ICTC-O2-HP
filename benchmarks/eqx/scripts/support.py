"""Small benchmark utilities, separate from the installed library."""
import torch


def leaves(value):
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(t for part in value for t in leaves(part))


def unpack_flat(module, value):
    parts, cursor = [], 0
    for block in module.blocks:
        width = block.multiplicity * block.irrep_dimension
        parts.append(value[..., cursor:cursor+width].reshape(
            *value.shape[:-1], block.multiplicity, block.irrep_dimension))
        cursor += width
    return tuple(parts)


def pack_blocks(blocks):
    return torch.cat([z.flatten(start_dim=-2) for z in blocks], dim=-1)


def unpack_backend(module, value, backend):
    return module._unpack(value) if backend == "xu_layout_triton" else unpack_flat(module, value)


def pack_backend(module, blocks, backend):
    return module._pack(blocks) if backend == "xu_layout_triton" else pack_blocks(blocks)


def max_error(a, b):
    return max((x-y).abs().max().item() / max(1e-7, x.abs().max().item(), y.abs().max().item())
               for x, y in zip(a, b))


def run_vjp(module, inputs, cot):
    ys = leaves(module(*inputs))
    grads = torch.autograd.grad(ys, (*leaves(inputs), *tuple(module.parameters())), cot, allow_unused=True)
    if any(g is None for g in grads):
        raise AssertionError("missing input/path-weight gradient")
    return ys, grads


def eqx_layout(blocks, value, channels):
    result, cursor = [], 0
    for block in blocks:
        width = block.multiplicity * block.irrep_dimension
        part = value[..., cursor:cursor + width]
        if block.irrep_dimension == 2:
            part = part.reshape(*value.shape[:-1], block.parity_copies, channels, 2)
            if block.parity_copies == 2:
                even, odd = part.unbind(dim=-3)
                part = torch.stack((even, torch.stack((-odd[..., 1], odd[..., 0]), dim=-1)), dim=-3)
            part = part.transpose(-1, -2)
        result.append(part.reshape(*value.shape[:-1], width))
        cursor += width
    return torch.cat(result, dim=-1)
