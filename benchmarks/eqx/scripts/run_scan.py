"""One channel group per batch task; paired timing and fresh-process memory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import ictc_o2_hp


def main():
    if not os.environ.get("SLURM_JOB_ID") or ictc_o2_hp.__version__ != "0.1.5":
        raise RuntimeError("requires sbatch and current ICTC-O2 HP 0.1.5")
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, choices=(32, 64, 96, 128, 256))
    parser.add_argument("--phase", choices=("max", "rest", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    channels = args.channels or (256, 128, 96, 64, 32)[int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))]
    sizes = (131072,) if args.phase == "max" else (65536, 32768, 16384) if args.phase == "rest" else (131072, 65536, 32768, 16384)
    folder = args.out.resolve() / f"C{channels}"
    folder.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    manifest = dict(status="running", version="0.1.5", C=channels, phase=args.phase,
                    N_order=list(sizes), calls=[], job_id=os.environ["SLURM_JOB_ID"],
                    fixed_positive_N_strategy=True)

    def execute(label, script, m, parity, n, extra):
        path = folder / (label + ".json")
        if path.exists():
            raise RuntimeError(f"refuse to overwrite existing record: {path}")
        command = [sys.executable, str(scripts / script), "--channels", str(channels),
                   "--order", str(m), "--parity", parity, "--edges", str(n),
                   *extra, "--out", str(path)]
        print("CALL", label, flush=True)
        started = time.monotonic()
        process = subprocess.run(command, check=False)
        data = json.loads(path.read_text())
        manifest["calls"].append(dict(label=label, command=command, source=str(path),
                                      exit_code=process.returncode, elapsed_s=time.monotonic() - started))
        (folder / ("manifest_" + args.phase + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
        if data.get("package_version") != "0.1.5" or data.get("status") not in ("ok", "oom"):
            raise RuntimeError(f"measurement failed: {path}")
        if process.returncode and data["status"] != "oom":
            raise RuntimeError(f"worker failed: {path}")
        return data

    for m in (6, 5, 4, 3, 2, 1):
        for parity in ("full", "canonical"):
            label = f"qualification-C{channels}-m{m}-{parity}-N131072"
            if args.phase != "rest":
                q = execute(label, "validate_math.py", m, parity, 131072, [])
            else:
                q = json.loads((folder / (label + ".json")).read_text())
            assert q["status"] == "ok" and q["package_version"] == "0.1.5"
            assert q["every_row_checked"] and q["every_weight_checked"]
            assert max(q["candidate_errors"].values()) <= 1e-5
    for n in sizes:
        for m in (6, 5, 4, 3, 2, 1):
            for parity in ("full", "canonical"):
                suffix = f"C{channels}-m{m}-{parity}-N{n}"
                for name in ("candidate", "equivariantx"):
                    execute("memory-" + name + "-" + suffix, "benchmark.py", m, parity, n,
                            ["--mode", "memory", "--implementations", name, "--samples-per-order", "6"])
                execute("paired-candidate-equivariantx-" + suffix, "benchmark.py", m, parity, n,
                        ["--mode", "paired", "--implementations", "candidate", "equivariantx", "--samples-per-order", "9"])
    manifest["status"] = "ok"
    (folder / ("manifest_" + args.phase + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print("CHANNEL_COMPLETE", channels, len(manifest["calls"]), flush=True)


if __name__ == "__main__":
    main()
