"""Export only current eqx comparison records from the original measurement archive."""
import argparse
import csv
import json
import os
from pathlib import Path
from report_eqx import CHANNELS, ORDERS, PARITIES, SIZES, FIELDS, write_report


def remove_unrelated_control(value):
    if isinstance(value, dict):
        return {key: remove_unrelated_control(item) for key, item in value.items()
                if not key.startswith(("baseline_", "old_new_"))}
    if isinstance(value, list):
        return [remove_unrelated_control(item) for item in value]
    return value


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("record export and validation require sbatch")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--original-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    raw_out = out / "raw"
    if raw_out.exists():
        raise RuntimeError(f"refuse to overwrite a previous export: {raw_out}")
    manifest = []
    for c in CHANNELS:
        for m in ORDERS:
            for parity in PARITIES:
                filenames = [f"qualification-C{c}-m{m}-{parity}-N131072.json"]
                for n in SIZES:
                    suffix = f"C{c}-m{m}-{parity}-N{n}.json"
                    filenames.extend(["paired-candidate-equivariantx-" + suffix,
                                      "memory-candidate-" + suffix, "memory-equivariantx-" + suffix])
                for filename in filenames:
                    source = args.raw_source / f"C{c}" / filename
                    target = raw_out / f"C{c}" / filename
                    original = json.loads(source.read_text())
                    assert original["package_version"] == "0.1.5"
                    selected = remove_unrelated_control(original)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")
                    manifest.append(target.relative_to(out).as_posix())
    rows, summary = write_report(raw_out, out)
    key = lambda r: (int(r["N"]), int(r["C"]), int(r["m_max"]), r["parity"], r["mode"])
    original = {key(r): r for r in csv.DictReader(args.original_csv.open())}
    assert len(original) == len(rows) == 480 and len(manifest) == 780
    numeric_fields = ["throughput_ratio", "conservative_IQR_ratio"]
    for name in ("candidate", "equivariantx"):
        numeric_fields.extend(name + "_" + field for field in ("ms", "q25_ms", "q75_ms", "rows_per_second", *FIELDS))
    for row in rows:
        reference = original[key(row)]
        for field in numeric_fields:
            expected = float(reference[field]) if reference[field] else None
            assert row[field] == expected, (key(row), field, row[field], expected)
        for field in ("candidate_status", "equivariantx_status", "paired_IQR_relation"):
            assert row[field] == reference[field], (key(row), field)
        assert row["paired_complete"] == (reference["paired_complete"] == "True")
    assert (summary["versus_EQX_faster"], summary["versus_EQX_slower"], summary["versus_EQX_equal"]) == (455, 22, 2)
    assert summary["candidate_completed"] == 480 and summary["EQX_completed"] == 479
    audit = dict(status="ok", source_version="0.1.5", record_files=len(manifest),
                 records=manifest, compared_metric_rows=len(rows), unchanged_numeric_fields=numeric_fields,
                 metadata_change="removed unrelated older ICTC control fields; source paths and job IDs retained",
                 omitted="historical ICTC controls, old/new pairs, driver manifests and unrelated experiments",
                 export_job_id=os.environ["SLURM_JOB_ID"])
    (out / "export_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("EQX_EXPORT_OK", len(manifest), "records; all", len(rows), "metric rows unchanged", flush=True)


if __name__ == "__main__":
    main()
