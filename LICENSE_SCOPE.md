# License scope — ICTC-O2 HP

The project-owned ICTC-O2 HP implementation is offered under the unmodified
[PolyForm Noncommercial License 1.0.0](LICENSE), using the same noncommercial
license and separate-commercial-permission framework as ICTC-O3-HP
(formerly MACE-ICTC-HP). Uses outside the standard permitted purposes require
a separate written grant from the relevant rights holder. See
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for the request procedure.

## Covered project-owned material

The offer covers the following material to the extent its copyright is
controlled by the ICTC-O2 HP rights holder:

- `src/ictc_o2_hp/`: public API, Cartesian/Fischer coefficient construction,
  mathematical reference, packed tensor products, runtime-weight adjoints
  and Triton kernels.
- Project-authored `examples/`, documentation and packaging configuration.
- Project-authored `benchmarks/eqx/` measurement, validation, reporting,
  adapter and plotting scripts, and project-owned figure/documentation assets.

The license does not create exclusive copyright in mathematical formulas,
measurement facts or other material that is not subject to such rights.

## Standard permitted purposes

All permitted-purpose terms in the standard PolyForm text apply unchanged.
In particular, its personal-use and noncommercial-organization provisions
remain as written. Use by the educational, public-research and other
organizations listed in the license is permitted regardless of funding
source or obligations resulting from that funding. Neither this scope nor
the commercial-permission procedure narrows those provisions.

## Third-party and previously granted rights

Dependencies and third-party source retain their own licenses and notices.
PyTorch, Triton, benchmark/plotting dependencies and the external EquivariantX
checkout are not relicensed by this offer. EquivariantX source is not bundled
with this library; any runtime-generated adaptation derived from that source
remains subject to its upstream terms. Only project-authored adapter code
is covered by the offer above.

Any pre-existing permissive rights, including applicable MIT permissions
previously granted by a rights holder, remain in force for that material.
This offer does not retroactively revoke those permissions or convert
third-party source into restricted project-owned code by renaming it.

The O2 source package does not include the MACE/e3nn source portions listed
in the O3 combined edition's license scope. Its project-owned package metadata
therefore identifies `PolyForm-Noncommercial-1.0.0`; the O3 combined edition's
`AND MIT` expression is not copied onto unrelated O2 files.

Commercial agreements can cover only rights their issuer is authorized to
grant. Third-party patents, model weights, data and trademarks keep their
own scope; no rights to them are implied by this offer.
