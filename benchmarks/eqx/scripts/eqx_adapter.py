"""Configure the latest EquivariantX tensor product for the ICTC O(2) workload."""
from dataclasses import dataclass
import ast
import inspect
import os
from pathlib import Path
import textwrap

import torch
from eqx.o2 import TensorProduct

from ictc_o2_hp import o2_math
from ictc_o2_hp.native_o2 import canonical_blocks, full_parity_blocks

UPSTREAM_ROOT = Path(os.environ["O2_UPSTREAM_ROOT"])
UPSTREAM_REVISION = "ac3aed47f5b35d49f1f237912150f0d9dd759ae7"

def metadata_forward():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Source adaptation and validation require sbatch.")
    fields = {"irreps_in1": "in1", "irreps_in2": "in2", "irreps_out": "out"}
    tree = ast.parse(textwrap.dedent(inspect.getsource(TensorProduct.forward)))
    class StaticMetadata(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if (node.attr == "dim" and isinstance(node.value, ast.Attribute)
                    and node.value.attr in fields and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"):
                return ast.copy_location(ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_static_dim_"+fields[node.value.attr], ctx=ast.Load()), node)
            if (node.attr in fields and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                return ast.copy_location(ast.Attribute(value=node.value,
                    attr="_static_"+node.attr, ctx=node.ctx), node)
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(StaticMetadata().visit(tree))
    output_path = Path(os.environ["O2_GENERATED_FORWARD_PATH"])
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text("# Derived from the latest EquivariantX TensorProduct.forward.\n"
        "# Only representation metadata attributes are replaced; tensor operations are unchanged.\n"
        +ast.unparse(tree)+"\n")
    namespace = dict(TensorProduct.forward.__globals__)
    exec(compile(tree, str(output_path), "exec"), namespace)
    return namespace["forward"]

class CompileReadyReference(TensorProduct):
    forward = metadata_forward()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field, suffix in [("irreps_in1", "in1"), ("irreps_in2", "in2"), ("irreps_out", "out")]:
            representation = getattr(self, field)
            setattr(self, "_static_"+field, tuple(representation))
            setattr(self, "_static_dim_"+suffix, representation.dim)

@dataclass(frozen=True)
class Entry:
    key: tuple[str, int]
    copy: int
    label: str

def build_reference_o2(channels: int, order: int, parity: str, *, compile_ready=False):
    source = Path(inspect.getfile(TensorProduct)).resolve()
    if UPSTREAM_ROOT.resolve() not in source.parents:
        raise RuntimeError(f"Expected the latest task snapshot, imported {source}")
    blocks = canonical_blocks(order, channels) if parity == "canonical" else full_parity_blocks(order, channels)
    entries = []
    for block in blocks:
        if block.frequency == 0:
            entries.append(Entry(block.key, 0, block.token))
        else:
            for copy in range(block.parity_copies):
                entries.append(Entry(block.key, copy, f"{block.frequency}m/copy{copy}"))
    indices = {(entry.key, entry.copy): i for i, entry in enumerate(entries)}
    instructions, base_paths = [], []
    output_keys = {block.key for block in blocks}
    for first in blocks:
        for second in blocks:
            for out_key in o2_math.o2_valid_output_keys(first.key, second.key):
                if out_key not in output_keys:
                    continue
                base_paths.append((first.key, second.key, out_key))
                for out_entry in entries:
                    if out_entry.key != out_key:
                        continue
                    first_copy = out_entry.copy if first.frequency else 0
                    second_copy = out_entry.copy if second.frequency else 0
                    instructions.append((indices[(first.key, first_copy)],
                        indices[(second.key, second_copy)], indices[(out_key, out_entry.copy)],
                        "uuu", True))
    irreps = [(entry.label.split("/")[0], channels) for entry in entries]
    module_type = CompileReadyReference if compile_ready else TensorProduct
    module = module_type(irreps, irreps, irreps, instructions,
        irrep_normalization="component", path_normalization="none",
        internal_weights=True, shared_weights=True)
    with torch.no_grad():
        module.weight.mul_(0.02)
    return module, dict(width=module.irreps_out.dim, parameters=module.weight_numel,
        paths=len(base_paths), instructions=len(instructions),
        representation=[dict(label=entry.label, irrep=str(ir), multiplicity=mul)
                        for entry, (ir, mul) in zip(entries, module.irreps_out)],
        connection_mode="uuu", path_normalization="none", compile_adapter=compile_ready,
        reference_paper="Through the Looking-Glass", reference_library="EquivariantX",
        upstream_revision=UPSTREAM_REVISION, operator_source=str(source))
