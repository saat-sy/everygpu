# Worker weight placement

## Decision

The joining worker chooses resident OLMoE units from observed DHT coverage and available VRAM. Clients later choose execution routes among ready workers. The learned model router chooses experts per token; neither decision is weight placement.

Personal laptops and rented machines run the same NVIDIA/CUDA worker. For Plan 2, assume continuous availability and full GPU use. Weight budget is `0.8 * min(free VRAM, optional --max-vram GiB)`. The remaining space is a heuristic reserve for runtime and future KV state, not a measured inference-capacity guarantee.

Select completely uncovered groups first, in model order: embedding, each shared layer with its experts, final. Take a whole group when it fits, then fill individual missing units. Add copies only once observed providers plus the new assignment cover all units, preferring the fewest existing copies. Keep assignments fixed until shutdown.

## Why

- Owners need not choose layer or expert IDs. Filling missing coverage makes an incomplete model usable before adding redundant capacity.
- Whole contiguous layers keep expert execution local where possible. Individual expert loading lets smaller remaining budgets contribute.
- A manifest unit is the placement granularity; a file is the download granularity. Expert files group 16 experts, but only selected expert tensors occupy GPU memory. An expert's shared-layer dependency need not be colocated.
- OLMoE has 16 layers and 64 experts per layer. One expert is 12 MiB, one complete layer about 800.27 MiB, and all weights about 12.89 GiB, excluding runtime allocations. A large enough worker can hold the full model; distributed validation must not imply placement inherently forces remote expert execution.
- DHT coverage is an observation, not a global snapshot or capacity reservation. Initial placement validation uses sequential joins. Concurrent workers can choose overlapping gaps; coordination, live rebalancing and performance-aware placement are deferred.
