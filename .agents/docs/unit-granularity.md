# Unit granularity

## Decision

A model is split into independently loadable units: `embedding`, `final`, `perception:<part>`, one `layer` unit per decoder layer (attention, norms, router, shared projections) and one `expert` unit per expert. Experts are written 16 per file; an expert depends on its layer unit.

## Why

- A worker that cannot hold a whole layer can still load a layer unit plus a subset of its experts.
- Muse has 15,872 experts; one file each would be unmanageable on the Hub, so files group experts while units stay separately addressable.
- Fused expert tensors (leading expert dimension) are cut into contiguous per-expert slices and rejoined by concatenation, so packed NVFP4 payloads and scales are never decoded.
- Quantization may differ per layer, so expert tensor sets are compared within a layer only.
