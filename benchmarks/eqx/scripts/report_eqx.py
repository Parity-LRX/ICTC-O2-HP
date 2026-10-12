"""Collect current ICTC-O2 HP / eqx records without historical controls."""
import argparse
import csv
import json
import os
from pathlib import Path

CHANNELS = (32, 64, 96, 128, 256)
SIZES = (16384, 32768, 65536, 131072)
ORDERS = tuple(range(1, 7))
PARITIES = ("canonical", "full")
FIELDS = ("peak_allocated_MiB", "peak_reserved_MiB", "sampled_process_peak_MiB")


def collect(raw_root, preview=False):
    raw_root = Path(raw_root)
    rows, gates, missing, ooms = [], [], [], []
    for c in CHANNELS:
        for m in ORDERS:
            for parity in PARITIES:
                folder = raw_root / f"C{c}"
                qpath = folder / f"qualification-C{c}-m{m}-{parity}-N131072.json"
                if not qpath.exists():
                    if preview:
                        missing.append(str(qpath))
                        continue
                    raise AssertionError(f"missing mathematical-reference record: {qpath}")
                q = json.loads(qpath.read_text())
                assert q["status"] == "ok" and q["package_version"] == "0.1.5"
                assert q["every_row_checked"] and q["every_weight_checked"]
                assert max(q["candidate_errors"].values()) <= 1e-5
                gates.append(q)
                for n in SIZES:
                    stem = f"C{c}-m{m}-{parity}-N{n}"
                    paths = {"paired": folder / f"paired-candidate-equivariantx-{stem}.json"}
                    paths.update({name: folder / f"memory-{name}-{stem}.json"
                                  for name in ("candidate", "equivariantx")})
                    if not all(p.exists() for p in paths.values()):
                        if preview:
                            missing.append(stem)
                            continue
                        raise AssertionError(f"missing measurement: {stem}")
                    records = {name: json.loads(p.read_text()) for name, p in paths.items()}
                    assert len({r["gpu_uuid"] for r in records.values()}) == 1
                    for name, record in records.items():
                        assert record["status"] in ("ok", "oom")
                        assert record["package_version"] == "0.1.5"
                        assert record["dtype"] == "FP32" and not record["tf32"] and not record["amp"]
                        assert (record["edges"], record["channels"], record["order"], record["parity"]) == (n, c, m, parity)
                        if record["status"] == "oom":
                            ooms.append(dict(N=n, C=c, m_max=m, parity=parity,
                                             process=name, phase=record.get("phase"),
                                             source=paths[name].relative_to(raw_root.parent).as_posix(),
                                             error=record.get("error")))
                    for mode in ("forward", "forward_vjp"):
                        pair = records["paired"].get("timing", {}).get(mode, {})
                        memory = {name: records[name].get("timing", {}).get(mode, {}).get(name, {})
                                  for name in ("candidate", "equivariantx")}
                        paired = "candidate" in pair and "equivariantx" in pair
                        ct, et = (pair["candidate"], pair["equivariantx"]) if paired else (memory["candidate"], memory["equivariantx"])
                        assert ct and memory["candidate"], (stem, mode)
                        row = dict(N=n, C=c, m_max=m, parity=parity, mode=mode,
                                   source_version="0.1.5", fixed_positive_N_strategy=True,
                                   paired_complete=paired,
                                   timing_source="same-process AB/BA" if paired else "fresh isolated processes on same allocated GPU",
                                   paired_source=paths["paired"].relative_to(raw_root.parent).as_posix(),
                                   qualification_source=qpath.relative_to(raw_root.parent).as_posix(),
                                   gpu_uuid=records["paired"]["gpu_uuid"],
                                   throughput_ratio=et["median_ms"] / ct["median_ms"] if et else None,
                                   conservative_IQR_ratio=et["q25_ms"] / ct["q75_ms"] if paired else None,
                                   paired_IQR_relation=("candidate_faster" if paired and ct["q75_ms"] < et["q25_ms"] else
                                                        "equivariantx_faster" if paired and et["q75_ms"] < ct["q25_ms"] else
                                                        "overlap" if paired else "isolated_unpaired"))
                        for name, timing in (("candidate", ct), ("equivariantx", et)):
                            row[name + "_status"] = "ok" if memory[name] else records[name]["status"]
                            row[name + "_memory_source"] = paths[name].relative_to(raw_root.parent).as_posix()
                            for field, original in (("ms", "median_ms"), ("q25_ms", "q25_ms"),
                                                    ("q75_ms", "q75_ms"), ("rows_per_second", "rows_per_second")):
                                row[name + "_" + field] = timing.get(original)
                            for field in FIELDS:
                                row[name + "_" + field] = memory[name].get(field)
                        for quantity, field in zip(("allocated", "reserved", "process"), FIELDS):
                            a, b = row["candidate_" + field], row["equivariantx_" + field]
                            row[quantity + "_memory_ratio"] = b / a if a is not None and b is not None else None
                        rows.append(row)
    if not preview:
        assert len(rows) == 480 and len(gates) == 60 and not missing
    return rows, gates, missing, ooms


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(raw_root, out):
    rows, gates, missing, ooms = collect(raw_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    comparable = [r for r in rows if r["throughput_ratio"] is not None]
    per_order = []
    for m in ORDERS:
        group = [r for r in comparable if r["m_max"] == m]
        per_order.append(dict(m_max=m, comparable=len(group),
                              faster=sum(r["throughput_ratio"] > 1 for r in group),
                              slower=sum(r["throughput_ratio"] < 1 for r in group),
                              equal=sum(r["throughput_ratio"] == 1 for r in group)))
    accounting, increases = [], []
    for mode in ("forward", "forward_vjp"):
        for field in FIELDS:
            group = [r for r in rows if r["mode"] == mode and r["equivariantx_" + field] is not None]
            differences = [r["candidate_" + field] - r["equivariantx_" + field] for r in group]
            accounting.append(dict(control="equivariantx", mode=mode, metric=field,
                                   comparable=len(group), lower=sum(d < 0 for d in differences),
                                   equal=sum(d == 0 for d in differences), higher=sum(d > 0 for d in differences),
                                   maximum_increase_MiB=max([0, *differences])))
            for r, difference in zip(group, differences):
                if difference > 0:
                    increases.append(dict(control="equivariantx", mode=mode, metric=field,
                                          N=r["N"], C=r["C"], m_max=r["m_max"], parity=r["parity"],
                                          candidate_MiB=r["candidate_" + field], control_MiB=r["equivariantx_" + field],
                                          increase_MiB=difference))
    summary = dict(status="ok", version="0.1.5", comparator="EquivariantX (eqx)",
                   shapes=len(rows) // 2, metric_rows=len(rows), qualifications=len(gates),
                   candidate_completed=sum(r["candidate_status"] == "ok" for r in rows),
                   EQX_completed=sum(r["equivariantx_status"] == "ok" for r in rows),
                   versus_EQX_comparable=len(comparable),
                   versus_EQX_faster=sum(r["throughput_ratio"] > 1 for r in comparable),
                   versus_EQX_slower=sum(r["throughput_ratio"] < 1 for r in comparable),
                   versus_EQX_equal=sum(r["throughput_ratio"] == 1 for r in comparable),
                   fixed_positive_N_strategy=True, dtype="FP32", tf32=False, amp=False,
                   numerical_gate=1e-5,
                   numerical_maxima={key: max(q["candidate_errors"][key] for q in gates)
                                     for key in ("output", "dx", "dy", "dw")},
                   per_order=per_order, memory_accounting=accounting, oom_records=ooms,
                   missing=missing, report_job_id=os.environ["SLURM_JOB_ID"])
    write_csv(out / "complete_matrix.csv", rows)
    write_csv(out / "per_order.csv", per_order)
    write_csv(out / "memory_accounting.csv", accounting)
    write_csv(out / "memory_increases.csv", increases,
              ["control", "mode", "metric", "N", "C", "m_max", "parity", "candidate_MiB", "control_MiB", "increase_MiB"])
    write_csv(out / "slower_than_EQX.csv", [r for r in comparable if r["throughput_ratio"] < 1], list(rows[0]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return rows, summary


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("data analysis requires sbatch")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, summary = write_report(args.raw.resolve(), args.out.resolve())
    print("EQX_REPORT_OK", summary["metric_rows"], summary["versus_EQX_faster"], summary["EQX_completed"], flush=True)


if __name__ == "__main__":
    main()
