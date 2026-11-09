# SM120 inference kernels

Source for the images embedded in the `xkb_*` Python modules and
`xmoe_experts.py`. These images are loaded only on capability 12.0; other
devices use the existing engine kernels. They do not replace the base wheel.

Provenance: `conf_best.zip`, SHA-256
`4f27d46b6056615a98546af45e014a42e285123b71694e3ee7a14dd5ad95b492`,
the 2026-10-30 handoff. The handoff's engine base is
`3d4ea3d3fc88dc99c1b44f1677b73a7f65f045ff`.

Compiler commands and flags: `build.json`. Rebuilding requires CUDA 13
and `sm_120a`; the result must replace the corresponding embedded image
and its SHA-256 together. Numeric equivalence and trained-model evaluation
are required after a rebuild.
