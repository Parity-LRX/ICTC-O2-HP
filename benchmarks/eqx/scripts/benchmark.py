"""Paired complete-VJP timings and isolated-process memory for O(2)."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import ictc_o2_hp

from ictc_o2_hp import o2_math
import support as worker
from ictc_o2_hp import TensorProduct as Candidate
from ictc_o2_hp import triton_o2 as kernels
from eqx_adapter import build_reference_o2
from memory_sampler import ProcessMemorySampler
ROOT = Path(os.environ.get("O2HP_RUN_ROOT", str(Path(__file__).resolve().parents[1] / "_runs")))


def glass_layout(blocks, value, channels):
    result = []
    cursor = 0
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


def make(a, names):
    torch.manual_seed(34527)
    candidate = Candidate(a.channels, a.order, parity=a.parity)
    gen = torch.Generator(device="cuda").manual_seed(78211)
    width = candidate.representation_dimension
    x = torch.randn((a.edges, width), device="cuda", generator=gen) * 0.1
    y = torch.randn((a.edges, width), device="cuda", generator=gen) * 0.1
    cot = torch.randn((a.edges, width), device="cuda", generator=gen) * 0.1
    modules, arguments, cotangents, metadata = {}, {}, {}, {}
    for name in names:
        if name == "candidate":
            module = candidate
            module = module.to(device="cuda", dtype=torch.float32)
            inputs, c = (x.detach().requires_grad_(), y.detach().requires_grad_()), cot
            metadata[name] = {"paths": len(module.paths), "width": width,
                              "parameters": module.parameter_count,
                              "effective_weight_schedule": getattr(kernels, "_WEIGHT_SCHEDULE", "static"),
                              "package_version": ictc_o2_hp.__version__,
                              "package_path": str(ictc_o2_hp.__file__)}
        else:
            module, metadata[name] = build_reference_o2(
                a.channels, a.order, a.parity, compile_ready=True)
            module = module.to(device="cuda", dtype=torch.float32)
            inputs = tuple(glass_layout(candidate.blocks, z, a.channels)
                           .detach().requires_grad_() for z in (x, y))
            c = glass_layout(candidate.blocks, cot, a.channels)
        if not all(p.requires_grad for p in module.parameters()):
            raise AssertionError("all path weights must remain trainable")
        modules[name], arguments[name], cotangents[name] = module, inputs, (c,)
    del x, y, cot
    if "candidate" not in names:
        del candidate
    return modules, arguments, cotangents, metadata


def run(module, inputs, cot, metric):
    if metric == "forward":
        with torch.no_grad():
            return module(*inputs)
    ys, grads = worker.run_vjp(module, inputs, cot)
    if any(g is None for g in grads) or len(grads) != 3:
        raise AssertionError("two input gradients and all packed path weights required")
    return ys, grads


def event_time(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def summarize(samples):
    q = statistics.quantiles(samples, n=4)
    return {"median_ms": statistics.median(samples), "q25_ms": q[0], "q75_ms": q[2],
            "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("paired", "memory"), default="paired")
    p.add_argument("--implementations", nargs="+", choices=("candidate", "equivariantx"), default=["candidate", "equivariantx"])
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--order", type=int, choices=range(1, 7), default=6)
    p.add_argument("--parity", choices=("canonical", "full"), default="full")
    p.add_argument("--edges", type=int, default=6400)
    p.add_argument("--samples-per-order", type=int, default=3)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("numerical execution requires sbatch")
    install_root = Path(os.environ.get("O2HP_INSTALL_ROOT", str(Path(__file__).resolve().parents[3] / "src"))).resolve()
    if install_root not in Path(ictc_o2_hp.__file__).resolve().parents or ictc_o2_hp.__version__ != "0.1.5":
        raise AssertionError("candidate must be the current standalone 0.1.5 source or install")
    if a.channels < 1 or a.edges < 1 or a.samples_per_order < 2:
        p.error("channels and edges must be positive; use at least two samples per order")
    if a.mode == "memory" and len(a.implementations) != 1:
        p.error("memory must use one implementation in a fresh process")
    out = a.out or ROOT / "results" / os.environ["SLURM_JOB_ID"] / (
        f"{a.mode}-C{a.channels}-{a.parity}-m{a.order}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": "started", "mode": a.mode, "channels": a.channels,
              "order": a.order, "parity": a.parity, "edges": a.edges,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "job_id": os.environ["SLURM_JOB_ID"], "script": str(Path(__file__).resolve()),
              "dtype": "FP32", "tf32": False, "amp": False,
              "package_path": str(ictc_o2_hp.__file__),
              "package_version": ictc_o2_hp.__version__,
              "candidate_library": "ICTC-O2 HP", "reference_library": "EquivariantX",
              "strategy": "representation-dependent only; no positive-N dispatch",
              "weight_reduction": "fixed 1024x32 chunk stage and fixed 32x32 final stage",
              "low_order_fused": os.environ.get("O2_LOW_ORDER_FUSED", "auto"),
              "backward_input_fused": os.environ.get("O2_BACKWARD_INPUT_FUSED", "auto"),
              "weight_channel_tile": kernels._WEIGHT_CHANNEL_TILE,
              "weight_edge_tile": os.environ.get("O2_WEIGHT_TILE", "auto"),
              "compact_weight_edge_tile": getattr(kernels, "_RUNTIME_EDGE_TILE", 0),
              "weight_schedule": getattr(kernels, "_WEIGHT_SCHEDULE", "static"),
              "weight_warps": getattr(kernels, "_WEIGHT_WARPS", 4),
              "forward_deferred_setting": os.environ.get("O2_FORWARD_DEFERRED", "auto"),
              "qualification": "independent wheel gates plus streamed large-N reference checks",
              "layout_conversion": "preparation only; each operator consumes its public native packed layout",
              "cotangent": "same sample; J maps the second copy to EquivariantX canonical basis, then component/channel transpose",
              "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable"))}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out.write_text(json.dumps(record, indent=2) + "\n")
    sampler = None
    try:
        modules, inputs, cot, record["implementations"] = make(a, a.implementations)
        compiled = {name: torch.compile(module, fullgraph=True, dynamic=False)
                    for name, module in modules.items()}
        record["timing"], record["first_calls_s"] = {}, {}
        for metric in ("forward", "forward_vjp"):
            record["phase"] = metric + "/compile_warmup"
            out.write_text(json.dumps(record, indent=2)+"\n")
            for name in modules:
                t0 = time.perf_counter()
                run(compiled[name], inputs[name], cot[name], metric)
                torch.cuda.synchronize()
                record["first_calls_s"][name + "/" + metric] = time.perf_counter() - t0
                for _ in range(3):
                    run(compiled[name], inputs[name], cot[name], metric)
            torch.cuda.synchronize()
            samples = {name: [] for name in modules}
            orders = list(itertools.permutations(tuple(modules)))
            if a.mode == "memory":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                sampler = ProcessMemorySampler()
                sampler.start()
                orders *= 3
            for round_id, order in enumerate(orders):
                record["phase"] = metric + "/measurement"
                for _ in range(a.samples_per_order):
                    for name in order:
                        samples[name].append(event_time(
                            lambda n=name: run(compiled[n], inputs[n], cot[n], metric)))
            record["timing"][metric] = {name: summarize(vals) for name, vals in samples.items()}
            for name in modules:
                record["timing"][metric][name]["rows_per_second"] = 1000*a.edges/statistics.median(samples[name])
            record["timing"][metric]["order_schedule"] = orders
            if a.mode == "memory":
                record["timing"][metric][a.implementations[0]].update(
                    peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
                record["timing"][metric][a.implementations[0]].update(sampler.stop())
                sampler = None
            out.write_text(json.dumps(record, indent=2) + "\n")
        record["status"] = "ok"
    except BaseException as exc:
        record["status"] = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "error"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        record["failure_peak_allocated_MiB"] = torch.cuda.max_memory_allocated()/2**20
        record["failure_peak_reserved_MiB"] = torch.cuda.max_memory_reserved()/2**20
        record["failure_is_successful_measurement"] = False
        if sampler is not None:
            record["failure_process_memory"] = sampler.stop()
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "traceback"}), flush=True)
    if record["status"] != "ok":
        print(record["traceback"], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
