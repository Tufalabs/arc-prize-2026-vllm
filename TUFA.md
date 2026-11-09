# The Tufa line of vLLM

*Register: as-built.*

This repository is upstream vLLM plus what Tufa's ARC agents need of the
engine, kept as one release line per upstream tag (`tufa/v0.29.0` today).
arc3 and arc2 pin a revision of it. This file is the one
account of what the line carries: every PR into the line adds its groups
under "What this line carries", so the file at a pinned revision describes
exactly that revision. How a consumer pins, builds and ships it -- the
precompiled build, the Kaggle overlay, the deployment rules the engine
cannot enforce -- is each consumer's own (`vllm_fork.md` in arc3
and in arc2).

## The repo

This repository is upstream vLLM at tag `v0.29.0` plus one commit, so
`git diff v0.29.0..main` is the whole fork. Keep the tag with any copy of this
repository: setuptools-scm reads the version off it, and without it the package
installs as `0.1.dev<N>` and every check that reads the engine's line refuses it.

**Carrying the stack across a tag.** Cherry-picking the fork's history one
commit at a time does not survive a base that has moved: the blanking stack
rewrites the same files several times, so its early commits conflict against
a tree they were never applied to. Fold it first, on its OWN base, into one
commit per logical change, and check the fold by `git diff` against the
branch it came from — empty, or the fold lost something. Then cherry-pick
the folded commits with `-x`, so each new commit names the one it came from.

Review the resolutions with

    git range-diff <old tag>..<folded> <new tag>..<port>

which pairs each ported commit with its original and shows only what the
resolution changed. For that to mean anything, the cherry-picks must contain
ONLY what was needed to make the old commit apply; anything the new base
demands on its own merits goes in its own commit on top. Otherwise the
range-diff stops distinguishing "what the upgrade forced" from "what we
decided", which is the one question it exists to answer.

One trap the 0.29 port hit: a conflict can resolve cleanly onto the wrong
line. Upstream had added a second, textually identical `if num_new_tokens ==
0: break`, and git attached the fork's replacement to the new one. It passed
review by eye and was caught by the fork's own scheduler test. A resolution
that lands somewhere plausible is not the same as one that lands where it
belongs.

**The fork's CI runs lint and mypy on the delta**, not on the tree: the
files a PR changed against its base branch, with `tufa-ci.yml`'s pinned mypy
and stub set. The scope is what makes that gate readable. Upstream's own
mypy does not pass on the whole tree — 285 errors at the v0.29.0 base, all
of them in vendored code (`vllm/third_party/`, the kimi_k3 `third_party/kda`
ops). Upstream's pre-commit leaves that code out of every hook (its top-level
`exclude`), and the jobs read the same pattern and leave it out too, though
the line edits two vendored files. On the delta the base is clean, so the job
inherits nothing and a red mypy is the PR's own. Check a claim about
it by running the job's own command; a wider run answers a different
question.

## What this line carries

`tufa/v0.29.0` is upstream's v0.29.0 tag plus these merged PRs, oldest
first: #5 (seven upstream cherry-picks and ten of ours: the PLE layer and the
QSA indexer as upstream has them, the Mamba state seed, the connector over a
ring group, the align split, the MTP block's own expert count); #6 (the
memory log, the draft loader's MoE backend, the FP8 KV branch); #7 (the
pipeline stages, the host-born PLE table, the device memory ceiling, the
multimodal block hash, the stage-local tables); #8 (the b12x routing-id
bound); #9 (`spec-decode-recoverssm`: speculative decoding for
Flash-Next and the prefix cache a parked host tier needs, nine upstream
cherry-picks and thirty-five of ours); #10 (`competitor-tricks`);
`spec-decode-4xl4` (#11: drafting under pipeline stages, the four-L4
round trip, the engine traces; five upstream cherry-picks and fourteen of
ours); `vllm031-picks` (#12: four upstream v0.31.0 cherry-picks);
`flash-int4` (#13: the INT4 KV cache, the FP8 recurrent state, the
RecoverSSM commit's stops, the transient KV pool and admission on a spent step
budget; twelve of ours); `b200-throughput` (#14: an expert count
per decoder layer, the CuteDSL MoE's W4A16 mode, and the B200's GDN and QSA
kernel shapes; nine of ours); and `host-upgrades` (#15: one PLE table
a host, an autotune file the caller names, and the one-pass tool-call parse;
seven of ours).
Every changed line in the line belongs to one of the groups below; a change
none of them describes is one this file has not caught up with.

**Experimental Flash-Next inference stack, opt-in.** The `XKB_*` and
`GDN_FUSED_COMMIT` switches carry the `conf_best` kernels: fused
sampling/top-k statistics, compact draft-vocabulary rejection sampling,
RecoverSSM commits folded into the following verify, strided GDN projections,
mixed-batch slicing, fused HC up/mix, last-row draft prefill, and SM120 MoE,
router, HC, L2 prefetch and GDN CUDA kernels. Off, the existing implementations
run. CUDA images are capability-12.0 only, with shape/dtype fallbacks;
`csrc/xkb` contains their sources and build provenance. The B200's existing
W4A16 backend, GDN launch shapes and per-layer expert counts are preserved.

The stack changes floating-point reduction order in sampling, routing and
state kernels. Last-row draft prefill retains every cache write, but changes
the attention/GEMM shapes. Neither seeded token identity nor model-quality
parity follows from mathematical equivalence. The consuming repository owns
the opt-in and benchmark/evaluation gate; no engine default enables it.

**Barriers in the Triton top-k/top-p kernel**, always on. `_topk_topp_kernel`
hands its per-program scratch row between warps: a gather stores outliers at
cumsum positions that other warps' tiles then read, and the row is reused for
the program's next batch row. Upstream has no barrier at those handoffs, so a
warp can read scratch another warp has yet to store, and the pivot search
then counts stale values. A `tl.debug_barrier` sits at each handoff; every
condition around them is a per-row scalar. Stock sampling runs this kernel
for any batch of eight rows or more, so the fix reaches every column, with or
without the stack, and changes no arithmetic.

**The recurrent-state blank** (`SamplingParams.zero_recurrent_state`), the
line's first change: a request's recurrent (Mamba/GDN) state deliberately
blanked at a chosen prompt position while its attention KV stays intact.
Why an agent wants that is the agent's (arc3
`duck.DuckAgent.zero_recurrent`). The V2 model runner implements it, and the
OpenAI-compatible server declares its fields on `ChatCompletionRequest` and
`CompletionRequest` and forwards them to `SamplingParams`; the scheduler cuts
a prefill chunk at the target. How the engine's limits on it are checked is
the last section's.

**Where the host memory went** (`vllm/utils/memory_log.py`, called from
`vllm/v1/engine/core.py` and `vllm/v1/worker/gpu_worker.py`). A cgroup OOM
kill leaves no traceback, so the only account of what filled the cgroup is
one printed before it did; `VLLM_MEMORY_LOG_S` starts a sampler in the
engine core and the worker, brackets `load_model`,
`determine_available_memory`, `initialize_from_config` and
`compile_or_warm_up_model`, and stops sampling once start-up is done --
unless `VLLM_MEMORY_LOG_SERVING` keeps a worker's sampler on through
serving, with torch's peak counters reset where start-up ends, since an
engine near its card's edge fails in serving. Its GPU figures are in MiB:
free and total, torch's allocated and reserved, and their peaks since the
reset. Unset, nothing runs. It reports the split the kernel acts on -- `anon` and
`unevictable` unreclaimable, `shmem` charged as cache but unreclaimable
without swap, `file` reclaimable except `file_mapped` -- because a total
says nothing about whether a cgroup at its limit is healthy or doomed.

It reports the process list with the totals, because the cgroup fields are
container-wide while ``VmRSS`` is one process: a cgroup that grows while
``VmRSS`` does not means somebody else, and only a name says who (on
Kaggle, `cicc`: FlashInfer building its kernels on a box with no network
to download them over). `VLLM_MEMORY_LOG_PAUSE_S`
holds at each marker, for a step too short for the sampling interval to
bracket on its own. All three variables are declared in `vllm/envs.py`, since
vLLM warns on any `VLLM_` name it does not recognise.

It lives in the fork rather than in a plugin because a plugin cannot reach
the engine core: vLLM discovers general plugins through installed entry
points, and the source checkout a Kaggle kernel runs has no dist-info to be
discovered by. A `sitecustomize` shim was the other way around that and
fails too: a spawned child receives the
parent's `sys.path` after `site` has already looked for one. No cell turns
it on; a run that needs it sets the variables in its engine environment.

**The PLE layer as upstream has it.** v0.29.0's PLE layer is the
pre-fusion one, and its FP8 loader selects only for a whole-checkpoint FP8
config. Cherry-picked in upstream order: `[Qwen4] validate FP8 PLE weight
scale after loading (#54722)`, `[Qwen3.8-Flash-Next] Fuse Qwen4Exp PLE
kernels (#54517)`, `[Bugfix][Model] Fix FP8 PLE loading in mixed ModelOpt
checkpoints (#54882)`, `[Bugfix][Qwen4Exp] fix state index strides in fused
PLE conv (#55375)`. Files: `models/qwen4_exp/nvidia/ple_layer.py`,
`nvidia/ops/ple.py` (new), `nvidia/model.py`, `nvidia/mtp.py` (the fused
path's call sites), `v1/attention/backends/short_conv_attn.py` (six lines
the fusion retires), `tests/models/qwen4_exp/test_ple.py`. #9's picks of
#55272, #54371 and #55309 (below) take the layer on to upstream's later
version, whose loader selects the FP8 table for an NVFP4 body with the table
excluded, RadixArk's checkpoint, by itself.

**The QSA indexer as the ladder ran it.** Three upstream commits after the
0.29 branch point, without which the sparse path is 0.29's earlier version
of the same design: `Separate prefill and decode paths for QSA indexer
(#54513)`, `Compact indexer logits workspace (#54915)`, `Improve QSA sparse
GQA for prefill and short-ctx decode (#54873)`. Files: `nvidia/qsa.py`,
`nvidia/indexer_qsa.py`, `nvidia/ops/qsa.py`, `nvidia/ops/qsa_indexer.py`
(new), `common/qsa_cache.py`, `model_executor/warmup/kernel_warmup.py`
(the warmup list keeps 0.29's sparse-MLA warmup and gains the QSA one),
`warmup/qwen4_exp_qsa_warmup.py` (new), `tests/models/qwen4_exp/
test_qsa_reference.py`. Between the PLE and QSA groups, one adaptation of
ours: `The compile-only warmup tensor carries explicit strides and answers
stride(dim)` (`warmup/jit_warmup_triton_helper.py`), the two lines of API
the QSA kernels use to describe their paged caches to warmup, which
upstream added inside a DeepSeek warmup migration this line does not carry.

**The Mamba state seed.** `The align-mode state seed is counted in the
mamba group's block size` (`v1/worker/gpu/model_states/mamba_hybrid.py`,
`tests/v1/worker/test_mamba_hybrid_state_seed.py`): a resumed request's
running-state column was seeded from its computed length divided by the
attention block size where the column is counted in the mamba group's;
on a hybrid the two can differ. Upstream tracks it as vllm#53142.

**The offloading connector passes over a ring group.** Flash-Next's
sparse layers keep one block per request of the raw keys of the open
compression group, a `CircularBufferSpec` smaller than a hash block that
the core never prefix-caches; the stock connector asserted on it at
start. It is a transient kind now: `is_transient_spec` in
`v1/kv_cache_interface.py`; `is_transient` on the group config, skipped by
the divisibility check in `kv_connector/v1/offloading/config.py` and by
every keying, store, load and lookup walk in `offloading/scheduler.py`,
whose partial-tail block is the persistent groups' block wherever the ring
sorts among them; `tests/v1/kv_connector/unit/offloading_connector/
test_scheduler.py`, with the ring first, middle and last.

**The align split cuts at the Mamba block.** The engine core sets
`cache_config.block_size` to the minimum over KV groups, which on
Flash-Next is the ring's 4 tokens, and the scheduler's align-mode chunk
split aligned prefill chunks to it instead of the 1568-token Mamba block:
a whole-prompt chunk ended past the last Mamba boundary and left every
full block null, so a repeated prefix never hit; a 2048-token budget ended
chunks mid-block and later hits restored a state short of its boundary.
The split now takes the Mamba groups' own block size
(`v1/core/sched/scheduler.py`); the stubs that call it carry that value
(`tests/v1/core/test_mamba_align_chunk_split.py`, which also gains the
ring-beside-Mamba reproduction, `tests/v1/core/test_prefix_caching.py`,
`tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`).

**The MTP block keeps an expert count of its own.** A checkpoint whose
decoder experts were pruned but whose MTP block was left whole declares
the block's count as `text_config.mtp_num_experts`; the draft model's
config is a fresh parse of the same file, so the override reaches the
draft's layers and nothing of the target's (`nvidia/mtp.py`,
`tests/models/qwen4_exp/test_config.py`).

**A drafter's own MoE backend under the V2 runner.**
`speculative_config.moe_backend` was applied only by the V1 proposer; the V2
draft loader now applies it too, so a target served on Marlin can run its
MTP head's unquantized experts on triton
(`vllm/v1/worker/gpu/spec_decode/eagle/utils.py`).

**The QSA main KV cache in FP8.** The sparse-attention owner accepts
`kv_cache_dtype="fp8_per_token_head"`: the per-token-head cache-update
kernel vLLM already has writes each (token, head) row as e4m3 with a
float32 scale inline in the page, and the sparse paged GQA kernel casts the
gathered tiles back and folds the key scale into the score column and the
value scale into the probability column. The cell is padded so a token's
bytes across heads stay a multiple of the compressed-key cache's, which the
page unifier can only match by an integer ratio. The indexer's caches and
the query stay bf16. FlashAttention's own FP8 check is bypassed because
these kernels, not FlashAttention's, read the cache
(`nvidia/qsa.py`, `nvidia/ops/qsa.py`, `model_executor/warmup/qwen4_exp_qsa_warmup.py`,
`tests/models/qwen4_exp/test_qsa_reference.py`, whose FP8 additions are
the two CPU tests of the page layout and its views: the fork carries no GPU
tests of its own).

**Pipeline stages.** Two changes let this architecture run over pipeline
stages, which every upstream check refused. The PLE layer gathers its
n-gram rows by the raw token ids, and only the first stage receives those,
so the refusal is now exactly a PLE layer placed after the first stage,
judged by the same layer split the model uses
(`ple_layers_beyond_first_stage` in `models/qwen4_exp/config.py`, applied
at config time in `model_executor/models/config.py` and again in
`models/qwen4_exp/nvidia/model_state.py`); this checkpoint's one PLE layer
is its second decoder layer, so any split with two or more layers on the
first stage serves, and later stages take no ids, as the model's layers
without a PLE already allow. And a stage allocates KV tensors for its own
layers only (`v1/core/kv_cache_utils.py`): upstream's per-worker projection
keeps the whole-model uniform spec inside a group a stage holds no layer of
-- the PLE's state group on every stage but the first -- and the tensor
builder read that spec's layers, so those stages built a tensor they could
not place and died allocating. The builder reads the projected names.
Measured on four B200 stages: 14.75 GiB of weights per stage, the
checkpoint's quarter, and the host tier sizing itself per stage.
The memory log's GPU line waits for CUDA to be initialised in its process
(`vllm/utils/memory_log.py`).

**A stage builds only the vocabulary tables and tower it runs.** Every
stage built the embedding, the LM head and the vision tower, though the
runner embeds tokens and encodes images on the first stage only and computes
logits on the last. The embedding is now built on the first stage and the
head on the last, the rest holding `PPMissingLayer`, as upstream's Qwen3.5
and Llama do (this checkpoint's head is untied); a later stage builds the
tower on the meta device behind a `StageMissingLayer`, as upstream builds a
tower with no modality allowed, which only the V2 runner allows -- V1's
`profile_run` encodes on every stage -- so a later stage under V1 is refused
by name at build. The deepstack placement check reads the config, since a
later stage no longer holds a tower to ask
(`models/qwen4_exp/nvidia/model.py`; the AMD model is untouched). Tests in
`tests/models/qwen4_exp/test_pipeline_stages.py`. On Flash-Next this frees
1.18 GiB on the first stage, 3.2 on each middle stage and 2.0 on the last;
four B200 stages emulating the four-L4 cell grow the pool from 1,033,284
tokens to 1,288,874. A tensor split of stages (2x2) takes the same code, the
tables vocab-parallel within each stage.

**The PLE table on the host is upstream's Engram offload** (#54371, below):
`engram_config.cpu_offload` creates the table in pinned host memory, so no
stage materialises it on the device, and reads its rows through UVA on a
side stream, a layer ahead.

**A per-process device memory ceiling** (`VLLM_DEVICE_MEMORY_LIMIT_GIB`,
`vllm/envs.py`, applied in `v1/worker/gpu_worker.py` when a worker takes
its device): the caching allocator refuses past it, so a large card stands
in for a small one. `gpu_memory_utilization` cannot do that -- it sizes
the pool, it caps nothing -- which is why an L4's construction failure
could not be reproduced on a B200 without it.

**A prompt tail completed by decode hashes all its images.** The block
hasher hashes a prompt's partial last block only once generated tokens
complete it, and it started such a block from the request's last mm input
alone, as if only generated tokens could follow the prompt's full blocks.
When that tail holds more than one image, the earlier ones were left out of
its hash; the next call, whose longer prompt hashes the same tokens with all
of them, missed the block and every one after, and with Mamba states on
block boundaries a miss there is a miss of the whole request. It starts from
the first mm input ending past the block now (`get_request_block_hasher` in
`v1/core/kv_cache_utils.py`; upstream's, unchanged on its main). Test in
`tests/v1/core/test_kv_cache_utils.py`.

**Bounded routing ids for b12x** (`_run_b12x_moe_plan` in
`vllm/model_executor/layers/fused_moe/b12x.py`). The direct b12x backend
indexes its expert tables with the routing ids it is handed and does not
bound them; vLLM's own b12x warm-up bounds the ids it invents
(`topk_ids.remainder_`) and the serving path did not, so CUDA graph capture
-- which runs the model over buffers no router has written -- read past the
expert tables and the engine died at start-up with an illegal address. The
serving path now applies the same in-place `remainder_`, which allocates
nothing, as capture demands. The backend is measured and not shipped
(arc3 `packages/ullm/flash_optimization.md`, "The MoE backend"); the fix is
what made the measurement possible. A second branch,
`sm120-gdn-prefill`, admits capability 12.0 to
`_resolve_gdn_prefill_backend` (head dim 128, CUDA 13); measured a wash on
the RTX and left unpinned.

**GDN RecoverSSM** (`model_executor/layers/mamba/ops/gdn_recoverssm.py`,
new; `v1/attention/backends/gdn_attn.py`, `short_conv_attn.py` and
`recoverssm_metadata.py`; `speculative_config.gdn_recoverssm`). A verify
runs a window from the step's starting state, writing no per-draft state,
and records per position what the accepted state needs; the commit replays
the accepted positions onto the checkpoint, as Kimi-K3's KDA RecoverSSM
does. No recurrent group then holds a state block per draft token: +17.6%
pool on the B200 (arc3 `packages/ullm/flash_optimization.md`, "Speculative
decoding"). The conv-only PLE group commits alongside, since every group
reads its window at an accepted count of one after it. The commit also
writes the state at a `prefix_match_unit` boundary a verify window crosses,
which a decode tail needs under drafting. The align-column fix it shares
with Kimi's RecoverSSM keeps the state in the last computed token's column.
Launch shapes from `benchmarks/kernels/benchmark_gdn_recoverssm.py`; GPU
tests in `tests/models/qwen4_exp/test_gdn_recoverssm.py`.

**A reduced draft vocabulary** (`models/qwen4_exp/nvidia/mtp.py`): EAGLE-3's
`draft_vocab_size` and `d2t`. A head whose own config sets
`draft_vocab_size` keeps an LM head over those ids, loads `d2t` from its
checkpoint and scatters its logits into the target vocabulary, where the
draft is sampled and verified. The size is read from the draft model
config, since the drafter class is built with the target's.

**What a draft config takes from the target** (`config/speculative.py`): the
cumem allocator, without which a draft VllmConfig revalidated beside a KV
connector and expandable segments is refused, and the resolved multimodal
settings.

**`logit_bias` with a drafter** (`sampling_params.py`,
`v1/engine/input_processor.py`): allowed under the V2 runner, which biases
the target's logits at every verified position before rejection sampling,
so a banned token's draft is rejected and never resampled; refused under
V1. ARC2's summary token ban rests on it.

**The PLE conv state is replicated only across ranks** (`ple_layer.py`,
`nvidia/model.py`): at one rank it shares a KV cache group with the GDN
states instead of taking a group of its own.

**Decode tails** (`CacheConfig.prefix_cache_decode_tail`;
`v1/core/single_type_kv_cache_manager.py`, the hybrid model state). A
decoding request keeps a prefix-cache entry at the last `prefix_match_unit`
boundary its decode passed -- the attention tail block and a copy of each
Mamba "align" group's state -- so a conversation's next turn resumes within
a unit of the reply's end. Under drafting the scheduler names a target block
the RecoverSSM commit writes the state into; the targets reach the GPU
through pinned, non-blocking copies.

**Context threads** (`CacheConfig.prefix_cache_context_threads`;
`v1/core/context_threads.py`, new). A client names each request's
conversation and history epoch in `SamplingParams.extra_args`; the engine
keeps each context's live thread and sends its dead variants to the
eviction end, still cached, when the thread is replaced, a later epoch
arrives, or `close_prefix_cache_context` closes it. A close also reaches a
request not yet freed, whose thread dies at its free: park mode holds a
finished request's blocks until its host copy lands, after the reply has
reached the client. A host tier reads the same bookkeeping
(`BlockPool.context_threads`: `take_dead_keys`, `request_decode_tails`) to
park threads whole.

**Block copies in chunks** (`v1/worker/utils.py`): a step's copy-on-write
copies go in chunks of at most 64 MiB, so a wave of partial-hit admissions
needs no more free memory than one chunk's sources.

**A divergent hit leaves transient groups out** (`v1/core/sched/scheduler.py`):
a per-request ring is never prefix-cached, so it is no lagging group to
fill.

**Opt-in, off:** drafts under the request's own top_k and top_p
(`speculative_config.draft_top_k_top_p`), online FP8 for the linears
ModelOpt leaves unquantized (`additional_config["online_fp8_linear"]`), the
FP8 indexer cache (upstream #54890), superseded-first eviction
(`prefix_cache_evict_superseded_first`) and relaxed draft acceptance (below).
The two FP8 changes were declined for the score
(arc3 `packages/ullm/flash_optimization.md`, "What ships").

**Upstream cherry-picks in #9:** #55054 (the PLE MTP metadata's copies made
asynchronous), #53388 (`disable_eagle_block_drop`: a hit keeps the block a
drafter would drop), #52807 (a recurrent group's unhashed block no longer
truncates the load boundary), #55450 (Mamba states retired across null
gaps), #54890 (above), #55272 (torch.compile removed from the Qwen4Exp
NVIDIA model), #54687 (the HC combine-norm reused for the MTP input),
#54371 (UVA PLE offload and Engram tensor parallelism) and #55309 (the PLE
residual add and the QSA output gate fused). Each commit names its PR and
what its conflicts took. #55272's ROCm rows in `tests/test_config.py` follow
the base's rule, which drops only the DeepseekV32 names there.

**Above #9: the competitor tricks** (vllm-tufa-arc #10, `competitor-tricks`;
arc3 `packages/ullm/flash_optimization.md`,
"Competitor tricks"). Six upstream cherry-picks: #58961 (the QSA key views
no longer keep the profiling KV cache alive), #58430 (allocator splits
during the profile run no longer count as consumed; ported by hand), #58720
(the expert-mapping lookups in `RoutedExperts.load_weights` indexed rather
than scanned), #58489 (the pinned PLE prefetch reads its n-gram ids from a
persistent buffer instead of graph-pool memory later segments reuse),
#58114 (the PLE short-conv metadata built without causal-conv1d metadata or
a device sync in mixed speculative batches; merged by hand over the fork's
decode-tail fields and RecoverSSM override, with `spec_req_idx` defined on
the all-spec path) and #54223 (the MXFP8 linear kernel chosen per layer by
its shape, so a layer below mm_mxfp8's 128-wide minimum falls to Marlin; the
layer config carries `weight_shape` without upstream's `bmm_batch_size`, the
FlashInfer kernels keep the fork's activation quantization, and the fork's
ModelOpt MXFP8 method builds its kernel in `create_weights`, where the shape
is known). Five of ours, from the competitor study: the input embedding in
pinned host memory (a public notebook's Qwen4Exp method, a Triton row gather
through UVA shared with the drafter, which `cpu_offload_params={"embed_tokens"}`
selects when no `cpu_offload_gb` is set; `tests/models/qwen4_exp/test_pinned_embedding.py`);
the graph-memory profile skipped when its estimate is not applied, so
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` saves the profiling capture as
well as the estimate; a compressed-tensors checkpoint's
PLE table kept unquantized (a public notebook's safeguard, for a W4A16 checkpoint
whose config declares FP8 over BF16 shards); large host-to-host PLE shard
copies over threads (sirikilohit's); and relaxed draft acceptance, off by
default (`speculative_config.accept_threshold_single` and
`accept_threshold_acc`, SGLang's two thresholds: a draft token is accepted
when the target gives it at least the first, or by the ratio test against
the second times its draft probability; Model Runner V2's standard rejection
sampling only, greedy requests exact;
`tests/v1/spec_decode/test_rejection_sampler_utils.py`). One test follows
the embedding (the stage test's stand-in config carries an
`offload_config`), and CI adds the worker and expert-loading tests the picks
reach and the MXFP8 selection test.

**Drafting under pipeline stages** (`spec-decode-4xl4`). Five
upstream cherry-picks, in upstream order: #54436 (a decoding request is never
dropped from the sampled-token broadcast), #50514 (EAGLE-3 under pipeline
parallelism in the V2 runner), #55745 (`record_stream` of `idx_mapping` in
the draft broadcast), #46994 (MTP under pipeline parallelism) and #54416 (no
fastsafetensors deadlock loading a draft model on a later stage); #46994's
test file leaves out two imports the fork's older copy never uses, which
tufa-ci's ruff refuses, and #54416 leaves out its changes to two test files
the fork does not carry. Four of ours. Qwen4Exp's MTP branches its input
path on whether it was handed intermediate tensors, not on the pipeline rank
(`models/qwen4_exp/nvidia/mtp.py`, `amd/mtp.py`). A stage before the last
commits GDN RecoverSSM with that step's own record, which rides the
pipeline receive's entry to the deferred commit at the micro-batch's next
step and keeps every tensor the commit reads, decode-tail targets included
(`v1/attention/backends/recoverssm_metadata.py`, `gdn_attn.py`,
`short_conv_attn.py`, `v1/worker/gpu/model_runner.py`, `model_states/`, and
`models/diffusion_gemma.py`'s model state, which takes the hook's new
keyword; `tests/v1/worker/test_recoverssm_deferred_commit.py`, which also
reads every model state's override against the hook). The offloading
connector's shared host region is sized by the largest stage's block, so
stages of different blocks -- the drafter on the last stage, or moved
layers -- agree on its layout, where they had corrupted host KV into an
illegal memory access (`v1/core/kv_cache_utils.py`, `v1/engine/core.py`,
`v1/kv_cache_interface.py`, `kv_connector/v1/offloading/config.py`; tests in
`tests/v1/core/test_kv_cache_utils.py` and
`offloading_connector/test_config.py`). And the EPLB test's runner carries
the model state the receive now consults
(`tests/v1/worker/test_gpu_model_runner_v2_eplb.py`).

**The pipeline's round trip** (arc3
`packages/ullm/flash_optimization_arc2.md`, "Drafting on the four L4s"). With
two micro-batches in flight, each waiting on its own sampled tokens and
drafts, a step is both stages plus the hand-offs, halved. Three changes
shorten it, the first two on by default behind a switch (`vllm/envs.py`):

- `VLLM_TP_LOCAL_TOPK_LOGITS`: under tensor parallelism a step whose
  requests all sample from a top-k that nothing reorders first gathers each
  rank's largest top-k + 128 logits (`LogitsProcessor.get_top_k_tokens`)
  instead of the vocabulary, and the sampler reads them at the vocabulary's
  width, -inf elsewhere: the logits the top-k keeps are exact, since the
  sampler and the rejection sampler both mask below the k-th before they
  normalize (which of several tokens tied at the k-th are kept can differ,
  as between the top-k kernels). Logit bias that only
  lowers (bans, `min_tokens` holding the stop tokens back) stays on the
  path while it touches at most the margin (`LogitBiasState.raises_logits`);
  raising bias, an allowed-token list, penalties, bad words, a thinking
  budget, a grammar or logprobs take the all-gather
  (`v1/worker/gpu/model_runner.py`, `sample/logit_bias.py`,
  `models/qwen4_exp/nvidia/model.py`;
  `tests/v1/worker/test_gpu_local_top_k_logits.py`).
- `VLLM_PP_BALANCED_MICRO_BATCHES`: under pipeline stages (the V2 runner,
  async scheduling) a request decodes in the micro-batch of the step its
  prefill ended, and nothing evened them out; a step now decodes at most an
  even share of its eligible decodes and those in flight, the rest a step
  later (`v1/core/sched/scheduler.py`, `async_scheduler.py`;
  `tests/v1/core/test_scheduler.py`).
- The pipeline consume filters a freed row's drafts with host indices and
  an asynchronous copy, where a device boolean mask blocked the host twice
  behind the stream (`v1/worker/gpu/pp_utils.py`).

On the four L4s' replay: the last stage's all-gathers 6.3 to 1.0 ms a step
(~4.5 of the 6.3 the logits'), the micro-batches 14/17-19 to 16/17
requests, +5.3% generated.

**Engine traces, off by default** (`vllm/envs.py`; JSONL, each named by a
path, unset nothing recorded). `VLLM_BATCH_QUEUE_TRACE` (`v1/engine/core.py`):
per batch-queue step, its depth on entry, the batch scheduled and when it was
sent, the batch waited on and for how long -- how many batches a pipeline
keeps in flight. `VLLM_WORKER_RPC_TRACE` (`v1/executor/multiproc_executor.py`):
per worker RPC, its method and when it was dequeued and finished, on the
node's monotonic clock as the batch-queue trace. `VLLM_KV_BLOCKS_TRACE`
(`v1/core/sched/scheduler.py`): about once a second, the pool's free and
deferred-free blocks and, per request holding blocks, its status, tokens and
blocks in each KV cache group -- who holds the pool.

**Above #11: four v0.31.0 picks** (vllm-tufa-arc #12, `vllm031-picks`; arc3
`packages/ullm/flash_optimization.md`, "vLLM v0.31.0 picks"). #57105 (the
QSA indexer's prefill logits workspace allocated at its worst case, so the
allocator reuses one block rather than one per size), #57104 (the scheduler
deadlock with a KV connector and MTP under KV pressure: a request promoted
after its async load could find no block for its draft lookahead), #57102
(one token-to-request mapping shared across a step's KV cache groups;
without upstream's `req_idx`, which the fork does not carry) and #58762 (GDN
metadata built once and reused across KV cache groups of the same spec).
#58762 is adapted: the fork's GDN builder carries RecoverSSM, whose commit
metadata names the group's spec state slots, block table and decode-tail
targets and whose context names its layers, so `update_block_table`
rebuilds both as `build` does; upstream's version would commit a later
group's accepted state into the first group's slots. Its test hunk for a
PCP drafter test the fork lacks is left out, and
`tests/v1/attention/test_gdn_metadata_builder.py` gains the RecoverSSM case,
which fails against upstream's version by the block-table offset it sets.
#57104 is completed: its margin for the promotion step counted the step's
slots in the scheduler's block size, one group's worth, while promotion
allocates the step in every group, so with several groups an admitted load
could still find too few blocks to be promoted. The margin is the
coordinator's count of the step over all groups, for the load being admitted
and for those in flight (`Scheduler._spec_decode_step_blocks`); upstream's
main has the single-group count. The full sequence keeps upstream's
recycling-aware reservation, but the step is counted as its allocation
counts it, without the admission cap: a sliding window past its cap still
allocates the step's blocks before it recycles any. That count records
nothing (the predictor's `read_only`), since the cap was also what kept a
Mamba group from recording a step that is only being reserved. Without it
the three-group test in
`tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py` fails below
the windows' caps and past them, admitting both of its loads.
Left out: #58582 (parallel Triton warmup), which sits on upstream's warmup
rework.

**Above #12: fewer bytes a token and a request** (vllm-tufa-arc #13,
`flash-int4`; arc3 `packages/ullm/flash_optimization.md`,
"INT4 KV cache", "The recurrent state in bf16", "The fixed blocks" and
"Admission on a spent budget"). Twelve of ours, no upstream picks.

**The QSA main KV cache in INT4.** The sparse-attention owner also accepts
`kv_cache_dtype="int4_per_token_head"`. A row is written as vLLM's
per-token-head INT4 cache update writes it -- a randomized Hadamard rotation,
asymmetric 4-bit values packed two to a byte, the zero point in the float32
scale's low mantissa bits -- with the rotation one Triton kernel ahead of the
shared packing kernel (`qsa_store_quantized_rows`). The query is rotated the
same way before the sparse kernel, divided by the head dim so every score is
kept, and the output rotated back after it with the gate applied then
(`qsa_int4_rotate`, one kernel each; the warmup builds the matrices before
graphs are captured). INT4 reads run a split-dot kernel of their own
(`_qsa_sparse_paged_gqa_splitk_int4_kernel`): each gathered row is loaded
once and split into its two nibbles, dotted against the query's even and odd
elements, and the zero points come off once per score column and value row.
Compute stays bf16; the gain is the pool (the kernel's time and the cell's
bytes: arc3's "INT4 KV cache"). The page layout
takes its data bytes (`qsa_quant_page_layout`) and its views say which format
they hold (`nvidia/ops/qsa.py`, `nvidia/qsa.py`; the layout's and the views'
CPU tests in `tests/models/qwen4_exp/test_qsa_reference.py`).

**A token's bytes as its layer stores them.** The hybrid block alignment
sized a token from the first backend it found, on Qwen4Exp the QSA side
cache's, which describes neither cache a QSA layer adds to a block, so every
recurrent page was padded to an undercount and a Mamba block left part of
itself empty (arc3's "The fixed blocks"). Attention layers now say what a token takes where their
backend's spec does not (`AttentionLayerBase.kv_bytes_per_token`, None by
default), and the alignment takes the largest (`platforms/interface.py`).
The QSA layer answers for INT4 only: its cell padded to the compressed-key
cache's multiple, plus that cache's row share. FP8 cells keep the backend's
count, from which the shipped cells derive their blocks.

**The GDN recurrent state in FP8 rows** (`mamba_ssm_cache_dtype="fp8_e4m3"`,
off by default; Qwen4Exp's NVIDIA model only, every other model refuses it;
`model_executor/layers/mamba/ops/gdn_fp8_state.py`, new). A head holds its V
rows of K e4m3 values, then their V float32 scales, each the row's absolute
maximum over 448 at its last write: the same bytes as a row with its scale
inline, so the state keeps its `[..., V, K + 4]` shape, pages and strides, and
every row starts 16-byte aligned. Writes round stochastically, from a hash of
the element's offset under a seed every launch advances. The packed decode
and sigmoid-gating updates, the RecoverSSM verify and commit, and prefill's
gather and scatter read and write the rows. The two updates are vendored
(`third_party/flash_linear_attention/ops/fused_recurrent.py`,
`fused_sigmoid_gating.py`), the line's first edits to vendored files; the
precompiled build extracts only its own trees from the base wheel, so these
are live. Declined for the defaults on score: seal sweep 2 read it out on its
own play, though it adds throughput on the RTX. The state's
shapes and dtypes have a CPU test (`tests/models/qwen4_exp/test_gdn_fp8_state.py`).

**The RecoverSSM commit replays to each stop.** In align mode the commit
stored the block-boundary and decode-tail states from inside its per-token
replay; with the FP8 state each store is the whole quantizing store, three
copies of it in the kernel. It now replays to each stop in order -- the
boundary and the tail, whichever the window reaches first, then its end --
with one store site after each, the stops unrolled (`tl.static_range`: a
runtime loop over them stored wrong states on a B200, though the Triton
interpreter ran it right). The same blocks are written, in the same order,
with the same rounding streams (`gdn_recoverssm.py`). On a B200 at
Flash-Next's decode shape (50 requests, k=4, 36 layers, align mode with
decode tails): the float32 commit 2.19 ms a step from 2.24, the FP8 one
4.09 from 23.8.

**The transient KV pool** (`CacheConfig.transient_kv_pool`,
`--transient-kv-pool`; off by default, V2 runner only, refused otherwise).
Every block id of the shared pool costs its widest group's bytes, and
Flash-Next's QSA compressor ring, a few kilobytes, took one such block per
running request (what that costs: arc3's "The fixed blocks"). The
option gives the transient groups a pool of their own. Grouping marks them
(`KVCacheGroupSpec.transient_pool`); the config builder sizes the shared pool
from the other groups and places the marked ones in an allocation of their
own -- four rings per `max_num_seqs` and transient group plus a null block,
each as wide as the widest ring group's packing -- whose count reaches the
workers as `KVCacheConfig.num_transient_blocks` and `KVCacheTensor.num_blocks`.
The transient groups' managers hand rings out of one free list and take them
back (`transient_ring_free_list`), since the groups' tensors overlay one
allocation, and a request that finds none free waits a step, as for KV
capacity. A pipeline stage holding none of the rings' layers keeps the count,
which the scheduler reads from one stage's config, and allocates nothing. The shared pool's zeroer, the offload connector and the warmup leave
the ring group out; the runner applies the scheduler's block copies to the
shared pool's tensors only (`shared_pool_kv_caches`), and the step-cost dummy
context takes each group's own ids (`v1/core/kv_cache_utils.py`,
`kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`,
`v1/kv_cache_interface.py`, `v1/worker/gpu/`, `v1/worker/utils.py`, the
offloading connector's `canonical_mapping.py` and `worker.py`; tests in
`tests/v1/core/test_contiguous_kv_packing.py`).

**Admission on a spent step budget** (`v1/core/sched/scheduler.py`). With a
KV connector the waiting loop no longer stops when the token or input budget
is spent: a request whose hit loads asynchronously (an external hit, a
divergent-hit fill) needs no budget this step and is still admitted, and one
that needs compute is set aside for the step, as an align-mode chunk that
cannot fit already is. Below a step budget smaller than a recurrent block, a
running prefill's chunk spends the whole budget every step, and the loop had
looked up about one waiting request a step: threads parked in the host tier
waited unread until the tier evicted them, then re-prefilled whole contexts.
Test in `tests/v1/core/test_scheduler.py`.

**Above #13: the B200's kernels** (vllm-tufa-arc #14,
`b200-throughput`; arc3 `packages/ullm/flash_optimization.md`,
"The MoE backend"). Nine of ours, no upstream picks: an expert count
per layer, then the three groups after it, which change kernels on the 10.x
family only or behind a variable.

**An expert count per decoder layer** (`models/qwen4_exp/nvidia/model.py`,
`mtp.py`, `model_executor/models/qwen3_next.py`). A checkpoint cut to a
different number of routed experts per layer declares the counts as
`text_config.num_experts_per_layer`, one per decoder layer, beside
`num_experts`, its largest; each decoder layer's MoE block builds its router
and expert stack at its own count. Without the key nothing changes. The MTP
block is whole and keeps `mtp_num_experts`; the draft config drops the list.
EPLB and AITER fused shared experts refuse the list, since both hold one count
for all layers. NVIDIA model only; tests in
`tests/models/qwen4_exp/test_config.py`.

**The CuteDSL NvFP4 MoE with BF16 activations**
(`VLLM_CUTEDSL_MOE_FP4_FORCE_A16`, off by default; `fused_moe/oracle/nvfp4.py`,
`fused_moe/experts/flashinfer_cutedsl_moe.py`, `envs.py`). The variable turns
`moe_backend="flashinfer_cutedsl"` from W4A4 into FlashInfer's W4A16 mode
(`CuteDslMoEWrapper(quant_mode="w4a16")`, FlashInfer #4048): the NvFP4 expert
weights are decoded to BF16 inside the kernel and no activation is quantized,
Marlin's arithmetic on the 10.x family's own tensor cores, which the backend's
support check keeps it to. The weights and scale factors are the CuteDSL
backend's own (gate/up interleaved, the scales in the MMA view); the alphas
are the weights' global scales with no activation scale folded in, and the
input-scale fold is skipped. One wrapper a layer, since its kernels are
compiled for the layer's expert count. arc3's B200 cell runs it.
The scheme it adds (NvFP4 weights, no activation key) is claimed only under
the variable and at BF16, the one output the W4A16 mode writes, so `auto`
leaves a native W4A16 checkpoint where it went before, on Marlin among the
rest. Its experts' weighted outputs are summed in the kernel's epilogue with
BF16 atomics, FlashInfer's fused finalize, whose order varies run to run;
`VLLM_CUTEDSL_MOE_W4A16_FUSED_FINALIZE=0` sums them in a separate reduction
in a fixed order instead.

**The GDN RecoverSSM kernels several value tiles a program, on the 10.x
family** (`model_executor/layers/mamba/ops/gdn_recoverssm.py`). The verify and
the commit take `TILES` consecutive value tiles of a request's head in turn,
so what the head shares across its tiles -- the request's metadata, the
per-layer base addresses, the window's q/k norms, gate and beta -- is loaded
and computed once a program rather than once a tile. On the 10.x family the
verify runs 32-row tiles, four a program, and the commit a whole head a
program (the `SM100_` constants, read at the B200 cell's own shape): the
commit stores the same states bit for bit, and the verify's wider tile sums
each row in another order. Every other card takes one tile a program at the
shapes it had (`_verify_launch`, `_commit_tiles`): the kernels it ran before,
bit for bit. `VLLM_SM100_KERNEL_SHAPES=0` puts the 10.x
family on those shapes too, and the INT4 configs below with them. The benchmark
(`benchmarks/kernels/benchmark_gdn_recoverssm.py`) sweeps the launch constant
its card reads, and takes its request count from `BENCH_REQUESTS`.

**The INT4 QSA kernel's own decode configs, on the 10.x family**
(`models/qwen4_exp/nvidia/ops/qsa.py`). `_select_config` takes
`int4_configs`: 128-column tiles and four warps for a drafted verify (decode
above 2,048 base programs) and the drafter's single-row passes (257-512),
where the table was tuned on a GB300 for the bf16 and FP8 kernels and an INT4
row is half their bytes. Read on a B200, so the 10.x family alone takes them
(`_int4_configs`); the outputs differ from the table's configs' by bf16
rounding, the sums split otherwise. The warmup asks for the configs of the
cache it warms. What the three select, on any card:
`tests/models/qwen4_exp/test_b200_kernel_selection.py`.

**Above #14: several engines a host, and the parse a reply waits on**
(vllm-tufa-arc #15, `host-upgrades`). Seven of ours, no upstream picks, in three groups:
the first two act only under a variable each, and the third returns what it
returned before, sooner.

**One PLE table a host** (`VLLM_QWEN4EXP_PLE_SHARED_TABLE`, unset by default;
`models/qwen4_exp/common/ple_shared.py`, `nvidia/ngram_embedding.py`). An
engine keeping Flash-Next's n-gram PLE table in pinned host memory holds a
copy of its own, tens of GiB, and engines on one host serving checkpoints
with the same table hold the same bytes each. With the variable naming a
file in `/dev/shm`, a holder builds the table there once from a checkpoint's
shards, in the engine's layout, and keeps it (`ple_shared.py hold`); each
engine maps it, registers the mapping with CUDA so its GPU reads the pages in
place, and checks each shard's first and last rows against it, so a table
built from another checkpoint fails the load. The mapping is writable and
registered without the read-only flag, which the B200 hosts' driver refuses;
nothing writes it. The holder is a job of its own, or, with
`VLLM_QWEN4EXP_PLE_SHARED_TABLE_HOLD_FROM` naming the checkpoint, a process
the one engine of a host starts in its own job, whose memory then carries the
table; an `flock` on a file beside the table keeps one holder a table, taken
and, when a holder dies, let go by the kernel, so no window lets a second
holder in and no stale lock needs taking over. A table is one
engine's whole table, so tensor parallelism is refused. Unset, each engine
allocates its own as before. Both variables are declared in `vllm/envs.py`,
since vLLM warns on any `VLLM_` name it does not recognise, and neither is a
compile factor: where the table lives keys no compiled graph. `ple_shared`
names them itself, since the holder runs it without vLLM. Tests in
`tests/models/qwen4_exp/test_ple_shared.py`.

**A FlashInfer autotune file the caller names**
(`VLLM_FLASHINFER_AUTOTUNE_CACHE_FILE`, unset by default; `envs.py`,
`model_executor/warmup/flashinfer_autotune_cache.py`). The variable names the
autotune config file itself, in place of one under the cache directory named
by the config's hash. That hash takes in state an engine fills in as it
starts, so no two engines of one configuration found each other's file and
each tuned again (the W4A16 experts, about 19 minutes on a B200); a caller
that keys the file by the settings the tuning depends on shares it across
jobs and hosts. A path, so no compile factor. Tests in
`tests/model_executor/test_flashinfer_autotune_cache_file.py`.

**A one-pass tool-call parse converts each call once**
(`parser/engine/parser_engine.py`). `extract_tool_calls` parses a whole reply
in one pass through the streaming machinery, which converted every argument
so far again for each argument chunk, to compute a delta the pass then
discards: quadratic in a long call, 4 s for a 40,000-character one, on the
caller's event loop. The pass now converts a call's arguments at its end
only; a stream still gets its deltas, and the result is the same. Test in
`tests/tool_parsers/test_qwen3coder_tool_parser.py`.

**CI.** `.github/workflows/tufa-ci.yml` whitelists upstream's PLE tests,
the QSA reference tests, the seed test, the fork's pipeline stage test and
upstream's `tests/v1/core/test_kv_cache_utils.py`. #9 adds upstream's
`tests/engine/test_arg_utils.py`, its own tests in `tests/test_config.py`, the
hybrid model state's CPU test and the block-copy chunk test.
`spec-decode-4xl4` adds the RecoverSSM deferred-commit test, the
local top-k test, the memory log's test, upstream's offloading connector
config test and the V2 runner's EPLB test. `vllm031-picks` adds
upstream's GDN metadata builder test with the RecoverSSM case, the KDA
metadata test and #57104's remote-prefill lifecycle test. Those builders
stage host tensors in pinned memory, which needs a driver, so
`tests/tufa_cpu_platform.py` turns pinning off along with the platform; two
cases of #58114's PLE metadata test that were deselected for the same
reason run again. `flash-int4` adds the FP8 state's CPU test and
upstream's contiguous KV packing test, which the transient pool reaches, and
leaves the vendored paths upstream's pre-commit excludes out of lint and mypy
("The repo", above). Its kernels' own tests need a GPU, so the line carries
none; runs are their evidence. `b200-throughput` adds the B200
kernels' selection test. `host-upgrades` adds the shared PLE table's
and the autotune file's tests, and upstream's Qwen3-Coder tool parser tests,
which reach the one-pass parse.

On the 27B this line also serves, the PLE and QSA groups are inert (no
table, no sparse layers), the seed change computes the same number, the
connector meets no ring group, and the split's Mamba block equals the
minimum it used before. Decode tails and context threads are off there, and
its Mamba groups, outside align mode, take no decode-tail targets. The
round trip's two switches act only under tensor parallelism (the logits) and
pipeline stages (the micro-batches). The QSA cache's INT4 path and the
transient pool find no sparse layers or ring group there, and the FP8 state
refuses any model but Qwen4Exp; the admission change acts wherever a KV
connector is configured. The per-layer expert count needs its key, the W4A16
mode its variable and an NvFP4 MoE, and the B200's GDN and QSA shapes a 10.x
card with RecoverSSM drafting or an INT4 sparse cache: the 27B has none of
them. The shared PLE table and the autotune file need their variables, and
the one-pass parse returns what it did before.

## What the engine cannot tell you

The blank is decided and applied inside a GPU kernel, because a
host-side decision costs a device sync per armed request per step and
async scheduling exists precisely so the CPU does not wait on the GPU.
The engine therefore **cannot log** whether a blank landed, was inherited
from cache, or was skipped; a blank that never lands looks exactly like
one that landed and did nothing. Validation is therefore external, and
splits by layer:

- **That the blank lands, and where** — the offline logprob replay
  (`arc3-experiments/logprob_repro/`). The blanked arm must
  collapse to the noise floor AND a position sweep must show its minimum
  exactly at the requested position, degrading either side. A single
  matching number is not evidence; the sweep shape is.
- **The control plane** — CPU tests in the fork, which need no GPU and
  run in seconds: the position's construction rules
  (`tests/test_sampling_params.py`), the scheduler cutting a prefill
  chunk at the target (`tests/v1/core/test_mamba_align_chunk_split.py`),
  and both OpenAI conversions carrying the fields
  (`tests/entrypoints/openai/test_zero_recurrent_conversion.py`).

What neither reaches is the kernel itself under mixed traffic — armed
beside unarmed requests, reused slots, the three state layouts. Those
need a GPU and do not exist yet.
