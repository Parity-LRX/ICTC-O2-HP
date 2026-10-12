"""Large-N candidate VJP checked against streamed original Cartesian algebra."""
import argparse
import json
import os
from pathlib import Path
import traceback
import torch
from ictc_o2_hp import TensorProduct, ReferenceTensorProduct
from support import unpack_flat, pack_blocks, run_vjp, max_error
from eqx_adapter import build_reference_o2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, required=True)
    p.add_argument("--order", type=int, required=True)
    p.add_argument("--parity", choices=("canonical", "full"), required=True)
    p.add_argument("--edges", type=int, default=131072)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("requires sbatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(51721)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    import ictc_o2_hp
    record = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"],
              "package_version": ictc_o2_hp.__version__, "package_path": str(ictc_o2_hp.__file__),
              "N": a.edges, "C": a.channels, "m_max": a.order, "parity": a.parity,
              "reference_chunk": a.chunk, "dtype": "FP32", "tf32": False, "amp": False}
    a.out.write_text(json.dumps(record, indent=2)+"\n")
    try:
        module = TensorProduct(a.channels, a.order, parity=a.parity).cuda().float()
        ref = ReferenceTensorProduct(a.channels, a.order, parity=a.parity).cuda().float()
        module.copy_weights_to_r5(ref)
        width = module.representation_dimension
        x, y = (torch.randn(a.edges, width, device="cuda").mul_(.1).requires_grad_() for _ in range(2))
        cot = torch.randn_like(x) * .1
        compiled = torch.compile(module, fullgraph=True, dynamic=False)
        record["phase"] = "candidate_compiled_full_VJP"
        a.out.write_text(json.dumps(record, indent=2)+"\n")
        actual_y, actual_g = run_vjp(compiled, (x, y), (cot,))
        # Compare every row of output and both input adjoints. Sum each original
        # path's reference weight adjoint across all chunks, in strict FP32.
        numerators = torch.zeros(3, device="cuda")
        denominators = torch.full((3,), 1e-7, device="cuda")
        weight_reference = torch.zeros_like(module.weight)
        all_finite = True
        record["phase"] = "streamed_reference_and_detached_statistics"
        a.out.write_text(json.dumps(record, indent=2)+"\n")
        for start in range(0, a.edges, a.chunk):
            end = min(a.edges, start+a.chunk)
            args = tuple(unpack_flat(module, z[start:end].detach()) for z in (x, y))
            args = tuple(tuple(z.requires_grad_() for z in blocks) for blocks in args)
            ref_y, ref_g = run_vjp(ref, args, unpack_flat(module, cot[start:end]))
            count = len(module.blocks)
            # These are verification statistics after complete adjoints have
            # been computed. Keeping a graph here retains every reference
            # chunk and inflates verifier memory; it is unrelated to the TP.
            with torch.no_grad():
                expected = (pack_blocks(ref_y), pack_blocks(ref_g[:count]), pack_blocks(ref_g[count:2*count]))
                actual = (actual_y[0][start:end], actual_g[0][start:end], actual_g[1][start:end])
                for i, (r, z) in enumerate(zip(expected, actual)):
                    numerators[i] = torch.maximum(numerators[i], (r-z).abs().max())
                    denominators[i] = torch.maximum(denominators[i], torch.maximum(r.abs().max(), z.abs().max()))
                    all_finite = all_finite and torch.isfinite(r).all().item() and torch.isfinite(z).all().item()
                weight_reference.add_(torch.cat(ref_g[2*count:]))
            del args, ref_y, ref_g, expected, actual
        errors = dict(zip(("output", "dx", "dy"), (numerators/denominators).tolist()))
        errors["dw"] = max_error((weight_reference,), actual_g[2:])
        if max(errors.values()) > 1e-5 or not all_finite or not torch.isfinite(actual_g[2]).all().item():
            raise AssertionError(errors)
        record.update(candidate_errors=errors, candidate_gate=1e-5,
                      every_row_checked=True, every_weight_checked=True, paths=len(module.paths),
                      finite_check="all reference/actual output and input-gradient rows in chunks; every weight",
                      parameters=module.parameter_count, width=width)
        print("LARGE_CANDIDATE_VALIDATED", a.channels, a.order, a.parity, a.edges, errors, flush=True)
        del actual_y, actual_g, compiled, module, ref, x, y, cot, weight_reference
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        # Qualify the comparator's native, metadata-adapted and compiled source
        # at C256 as well as the original widths; the adapter changes metadata only.
        record["phase"] = "equivariantx_native_adapter_compiled_N32"
        native, _ = build_reference_o2(a.channels, a.order, a.parity, compile_ready=False)
        adapted, _ = build_reference_o2(a.channels, a.order, a.parity, compile_ready=True)
        adapted.load_state_dict(native.state_dict())
        native, adapted = native.cuda().float(), adapted.cuda().float()
        args = tuple(torch.randn(32, native.irreps_in1.dim, device="cuda")
                     .mul_(.1).requires_grad_() for _ in range(2))
        cot = (torch.randn_like(args[0])*.1,)
        ref_y, ref_g = run_vjp(native, args, cot)
        direct_y, direct_g = run_vjp(adapted, args, cot)
        compiled = torch.compile(adapted, fullgraph=True, dynamic=False)
        new_y, new_g = run_vjp(compiled, args, cot)
        errors = {"adapter_output": max_error(ref_y, direct_y), "adapter_VJP": max_error(ref_g, direct_g),
                  "compiled_output": max_error(ref_y, new_y), "compiled_VJP": max_error(ref_g, new_g)}
        if max(errors.values()) > 1e-3:
            raise AssertionError(errors)
        record.update(equivariantx_errors=errors, equivariantx_gate=1e-3, equivariantx_validation_N=32, status="ok")
    except BaseException as exc:
        record.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                      error=repr(exc), traceback=traceback.format_exc())
    a.out.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k != "traceback"}), flush=True)
    if record["status"] != "ok":
        print(record["traceback"], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
