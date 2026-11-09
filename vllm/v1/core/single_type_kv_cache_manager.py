# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    resolve_block_hashes,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request

logger = init_logger(__name__)


def committed_decode_boundary(
    request: Request, unit: int, block_size: int
) -> int | None:
    """The last ``unit`` boundary of the request's committed tokens past its
    prompt, inside a block, whose key can be computed; or None. Under
    speculative decoding the scheduler's computed count runs ahead of what
    has been verified (``num_in_flight_tokens``), and only a committed token
    has final KV and state."""
    committed = request.num_computed_tokens - request.num_in_flight_tokens
    boundary = committed // unit * unit
    if boundary <= request.num_prompt_tokens or boundary % block_size == 0:
        return None
    if len(request.block_hashes) < boundary // unit:
        return None
    return boundary


def decode_tail_boundary(request: Request, unit: int, block_size: int) -> int | None:
    """The decode boundary ``request`` stands on as its next step is scheduled
    (``CacheConfig.prefix_cache_decode_tail``), or None.

    Its computed tokens -- what the previous steps wrote, and so what the
    running state holds until this step's forward -- end on a ``unit``
    boundary past the prompt, inside a block (a block boundary has its own
    entry), and every token up to it is known, so its key can be computed.
    """
    boundary = request.num_computed_tokens
    if boundary <= request.num_prompt_tokens:
        return None  # prefill: the prompt's last boundary has its own entry
    if boundary % unit != 0 or boundary % block_size == 0:
        return None
    if len(request.block_hashes) < boundary // unit:
        return None
    return boundary


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            scheduler_block_size: The scheduling granularity (LCM of all group
                block sizes); a multiple of this manager's ``block_size``.
            needs_kv_cache_zeroing: Whether worker-side KV cache zeroing needs
                newly allocated block IDs from this manager.
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every
                block until the request finishes.
        """
        self.scheduler_block_size = scheduler_block_size
        # Hybrid fine-grained lookup may lower this after all participating
        # managers have been validated by the coordinator.
        self.cache_hit_alignment_tokens = scheduler_block_size
        # The block size for this manager; used for actual block allocation.
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        self._record_new_block_ids = (
            needs_kv_cache_zeroing
            and isinstance(kv_cache_spec, AttentionSpec)
            and not isinstance(kv_cache_spec, CircularBufferSpec)
        )
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

        # CacheConfig.prefix_cache_decode_tail: the hash unit decode tails land
        # on, set by the coordinator when they are kept; None otherwise. And
        # how they are taken (KVCacheConfig.decode_tail_mode).
        self.decode_tail_unit: int | None = None
        self.decode_tail_mode: str | None = None
        # CacheConfig.prefix_cache_evict_superseded_first, set by the
        # coordinator; and per request, whether it resumed a conversation of
        # its own (its hit ends on a block no other request has taken).
        self.evict_superseded_first = False
        self._resumed_own: dict[str, bool] = {}
        # The full-attention manager whose blocks tell whether other running
        # requests share a prefix (set by the coordinator for recurrent
        # groups when superseded blocks go first).
        self.attention_probe: SingleTypeKVCacheManager | None = None

        # Whether this group's prefix-cache hits drop the EAGLE/MTP lookahead
        # block. Only consulted by managers whose hit logic is sparse within an
        # aligned segment (SWA). Initialized lazily by the coordinator after
        # determining the attention groups.
        self.use_eagle = False

        # Partial-hit copy-on-write bookkeeping. Populated only by fine-grained
        # managers (full attention, mamba "align"); harmlessly empty elsewhere.
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        # Boundary-state offload hand-off for external KV connectors. A mamba
        # "align" block table is not append-only (interior states are
        # nulled/freed and speculative blocks relocate in place), so a
        # connector cannot resolve its state blocks positionally. Record
        # (request, group, block, exact token boundary) for each committed
        # boundary state so a connector can offload the right block under the
        # right hash. Populated only by mamba "align".
        self._pending_boundary_state_offloads: list[
            tuple[str, int, KVCacheBlock, int]
        ] = []

    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def _has_partial_local_hit(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> bool:
        # The local prefix-cache hit ends inside one of this manager's
        # blocks: the shared tail block needs CoW.
        return (
            len(new_computed_blocks) > 0
            and num_local_computed_tokens % self.block_size != 0
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        read_only: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, clamp by `num_required_blocks` by
                `_max_admission_blocks_per_request`for recycling-aware specs
                (SWA, chunked-local).
            read_only: If True, record nothing for the step the count predicts,
                as for a reservation no allocation follows. An admission-capped
                count records nothing either way.

        Returns:
            The number of blocks to allocate.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Reserve the extra block that allocate_new_blocks pulls for the
            # partial-hit CoW redirect.
            num_new_blocks += 1
        return num_new_blocks + num_evictable_blocks

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the locally cached (prefix-hit) blocks to the request:
        1. Touch the computed blocks (paired with adding them to `req_blocks`)
           so their ref_cnt exactly tracks the referencing requests.
        1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
        2. Add the remaining computed blocks.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
            if self.evict_superseded_first:
                for block in new_computed_blocks:
                    if not block.is_null:
                        block.num_hits += 1
                last_hit = next(
                    (b for b in reversed(new_computed_blocks) if not b.is_null), None
                )
                self._resumed_own[request_id] = (
                    last_hit is not None and last_hit.num_hits == 1
                )
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Record the partial tail for the CoW redirect in
            # allocate_new_blocks; cap the cached count at the full blocks so
            # cache_blocks() re-caches the private copy once full.
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
            self.num_cached_block[request_id] = block_idx

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Allocate new blocks for external (KV-connector) computed tokens.

        Must run only after every group's local blocks have been touched via
        `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
        evict another group's cache-hit blocks (issue #33775).

        Args:
            request_id: The request ID.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = max(
            0, cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_fill_block(
        self, request_id: str, num_local_computed_tokens: int
    ) -> None:
        """
        Allocate the block(s) a connector fills for a divergent hit: what an
        external hit ending exactly at `num_local_computed_tokens` would get,
        after `add_local_computed_blocks` (called with no local blocks for
        this group) padded the skipped ones with nulls. For a single-state
        group that is one block, at the last chunk of the local hit.

        Must run after every group's local blocks have been touched, for the
        same reason as `allocate_external_computed_blocks`.
        """
        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_local_computed_tokens, self.block_size) - len(req_blocks)
        )
        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        cow_blocks: list[KVCacheBlock] = []
        if request_id in self._partial_hit_reqs:
            # Partial hit: redirect the shared tail to a private CoW block.
            # Replacing in place keeps the length-based allocation below
            # correct; the extra block was reserved by
            # get_num_blocks_to_allocate.
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return cow_blocks
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return cow_blocks + new_blocks

    @property
    def records_new_block_ids(self) -> bool:
        """Whether this manager's new blocks are zeroed by the worker."""
        return self._record_new_block_ids

    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def take_pending_cow_copies(
        self,
    ) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Drain pending CoW source and destination block pairs."""
        pending_copies = self._pending_cow_copies
        self._pending_cow_copies = []
        return pending_copies

    def take_decode_tail_updates(self) -> dict[str, tuple[int, int]]:
        """Drain decode-tail targets for the worker: request -> (boundary,
        block id), boundary 0 to clear. Only mamba "align" in "recover" mode
        populates this."""
        return {}

    def context_thread_blocks(
        self, request_id: str
    ) -> list[tuple[KVCacheBlock, BlockHashWithGroupId]]:
        """Cached blocks of the request's thread that its block list does not
        hold (``CacheConfig.prefix_cache_context_threads``): a mamba decode
        tail."""
        return []

    def take_pending_boundary_state_offloads(
        self,
    ) -> list[tuple[str, int, KVCacheBlock, int]]:
        """Drain producer boundary-state hand-offs.

        Entries are ``(req_id, group_id, block, boundary_tokens)``.

        Only mamba "align" populates this. The blocks are not kept alive by
        the request block table for the whole request, so a caller that reads
        them asynchronously must pin them first.
        """
        pending = self._pending_boundary_state_offloads
        self._pending_boundary_state_offloads = []
        return pending

    def _prefix_shared(self, request_id: str, num_tokens: int) -> bool:
        """Whether another running request holds the attention for this
        request's full blocks up to ``num_tokens``: a prefix conversations
        share (a system prompt), whose recurrent states are not the request's
        to supersede. A boundary inside the first block counts as shared.
        (The block the boundary falls in is left out: after a partial hit it
        is the request's own copy, with a reference held until the copy has
        run.) Without a probe, never."""
        attention = self.attention_probe
        if attention is None:
            return False
        block_idx = num_tokens // attention.block_size - 1
        if block_idx < 0:
            return True
        blocks = attention.req_to_blocks.get(request_id)
        return (
            blocks is not None
            and block_idx < len(blocks)
            and blocks[block_idx].ref_cnt > 1
        )

    def _apply_cow(
        self,
        request_id: str,
        block_idx: int,
        source_block: KVCacheBlock,
        cow_block: KVCacheBlock,
    ) -> None:
        """Redirect a partial prefix-cache hit to a private CoW block.

        Both copy endpoints stay retained until the copy has run on the worker,
        so a same-step free cannot recycle them: ``source_block`` keeps its
        hit-ref, ``cow_block`` takes an extra ref beyond the one handed to the
        request.
        """
        req_blocks = self.req_to_blocks[request_id]
        assert block_idx < len(req_blocks)
        assert req_blocks[block_idx] is source_block
        assert not source_block.is_null and source_block.ref_cnt > 0
        req_blocks[block_idx] = cow_block
        self._pending_cow_copies.append((source_block, cow_block))
        cow_block.ref_cnt += 1

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
            retention_interval: Sparse local-checkpoint granularity. ``None``
                keeps dense checkpointing; ``0`` keeps only the latest replay
                boundary; a positive multiple of ``scheduler_block_size`` keeps
                a tail once per that-sized segment. Only SWA acts on it.
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        # Token boundaries whose reachable tail must be retained under sparse
        # retention: the replay boundary (``num_prompt - 1``, capped by
        # ``get_computed_blocks``) and any detected shared-prefix junction.
        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)

        block_mask = self.reachable_block_mask(
            start_block=num_cached_blocks,
            end_block=num_full_blocks,
            alignment_tokens=self.cache_hit_alignment_tokens,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
        )
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
            block_mask=block_mask,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
    ) -> list[bool] | None:
        """Per-block mask for ``cache_full_blocks``. ``None`` means cache
        every (non-null) block — the default for full attention.

        Subclasses with sparse hit semantics (SWA / Mamba) override this to skip
        blocks that can never serve a hit at any alignment-aligned prefix length.
        ``reachable_boundaries`` are token positions whose reachable tail must be
        retained; the base (dense) policy ignores them.
        """
        return None

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        Pop the request's bookkeeping and return its blocks without yet
        returning them to the block pool. The caller is responsible for
        eventually passing the returned blocks to `block_pool.free_blocks`,
        freeing them in reverse order (so that tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        self._partial_hit_reqs.pop(request_id, None)
        self._resumed_own.pop(request_id, None)
        return req_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Free blocks in reverse order so that the tail blocks are freed first.
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache.

        Args:
            running_request_id: The request ID.

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.
        If eagle is enabled, drop the last matched block to force recompute the
        last block to get the required hidden states for eagle drafting head.
        For multi-module MTP, this recompute also rewrites the dropped block's
        draft-layer KVs, which depend on up to num_speculative_tokens - 1
        tokens past the matched prefix (i.e. on the cache writer's
        continuation, which the block hash does not cover); the coordinator
        asserts the block size covers that window.
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
                Always False for non-EAGLE/MTP groups, but can be False for EAGLE/MTP
                groups too if the last block is already dropped (e.g., in a
                convergence loop in `find_longest_cache_hit`).
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens). By default, it should
                be set to the block_size.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.

        Returns:
            A tuple containing cached blocks and the exact cache-hit length in
            tokens. The cached block tuple has skipped blocks replaced by null
            blocks for each kv cache group in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """

        raise NotImplementedError

    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        """Free blocks in ``[first_block, last_block)`` and replace with null_block.

        Iterates backward so newly-evictable tail blocks are reached even after
        earlier blocks in the range were nulled in a prior call.
        """
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for attention types (e.g.
                R-SWA) that evict a middle gap rather than a head prefix. Ignored
                by the default implementation.
        """
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        return 0

    def new_step_starts(self) -> None:
        return None


class FullAttentionManager(SingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size > 1:
            # DCP shards each block's KV across ranks; hashes must be viewed at
            # the sharded block size.
            block_size *= dcp_world_size
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        # Fine-grained mode (alignment_tokens == hash_block_size <
        # block_size): resolve_block_hashes kept the raw hash-granularity
        # list so interior boundaries can be probed.
        fine_grained = (
            alignment_tokens < block_size and block_size % alignment_tokens == 0
        )
        if fine_grained:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            full_block_hashes: BlockHashList = BlockHashListWithBlockSize(
                block_hashes, alignment_tokens, block_size
            )
        else:
            full_block_hashes = block_hashes

        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        # Phase 1: longest run of cached full blocks from the start. A missing
        # block implies every later block misses too (chained hashes).
        for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
            cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached_block:
                break
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        hit_length = len(computed_blocks[0]) * block_size

        # Phase 2 (fine-grained only): extend into the first non-full block by
        # probing its interior hash boundaries high-to-low (longest hit first).
        if fine_grained:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            scale_factor = block_size // alignment_tokens
            first_partial_idx = len(computed_blocks[0]) * scale_factor
            max_partial_idx = min(
                first_partial_idx + scale_factor - 1,
                max_length // alignment_tokens,
                len(block_hashes),
            )
            for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
                cached_tail = block_pool.get_cached_block(
                    block_hashes[fine_idx], kv_cache_group_ids
                )
                if not cached_tail:
                    continue
                for computed, cached in zip(computed_blocks, cached_tail):
                    computed.append(cached)
                hit_length = (fine_idx + 1) * alignment_tokens
                break

        # Eagle needs the tokens right before the generation point recomputed:
        # drop one hash unit when fine-grained (the tail block's KV is
        # append-only, so it still covers the reduced length), else one cache
        # block.
        if drop_eagle_block and hit_length > 0:
            hit_length -= min(alignment_tokens, block_size)
        # Round down to the alignment; a no-op when fine-grained (hits land on
        # hash boundaries by construction) and when alignment_tokens ==
        # block_size. Then trim blocks past the new tail.
        hit_length -= hit_length % alignment_tokens
        num_blocks = cdiv(hit_length, block_size)
        for computed in computed_blocks:
            del computed[num_blocks:]
        return computed_blocks, hit_length

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
        hash_block_size = self.block_pool.hash_block_size
        if self.block_size == hash_block_size:
            return
        self._cache_partial_tail_block(request, num_tokens)
        if self.decode_tail_unit is not None:
            self._cache_decode_tail_block(request)

    def _cache_decode_tail_block(self, request: Request) -> None:
        """Key the tail block at the decode boundary the request stands on
        (``CacheConfig.prefix_cache_decode_tail``). Attention KV is
        append-only, so the block itself serves the entry: a hit copies it
        before writing past the boundary. The entry replaces the block's
        shorter partial one, as the prompt's last boundary is replaced."""
        assert self.decode_tail_unit is not None
        if self.decode_tail_mode == "recover":
            boundary = committed_decode_boundary(
                request, self.decode_tail_unit, self.block_size
            )
        else:
            boundary = decode_tail_boundary(
                request, self.decode_tail_unit, self.block_size
            )
        if boundary is None:
            return
        blocks = self.req_to_blocks[request.request_id]
        block_idx = boundary // self.block_size
        if block_idx >= len(blocks):
            return
        self.block_pool.cache_partial_block(
            request=request,
            block=blocks[block_idx],
            num_tokens=boundary,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )

    def _cache_partial_tail_block(
        self,
        request: Request,
        num_tokens: int,
    ) -> None:
        """Cache the prompt tail when it ends inside a cache block.

        Only the final prompt hash boundary is registered as a partial
        prefix-cache entry; intermediate hash boundaries inside the same cache
        block are intentionally skipped.
        """
        hash_block_size = self.block_pool.hash_block_size
        boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size
        if boundary_tokens == 0 or boundary_tokens > num_tokens:
            return
        if boundary_tokens % self.block_size == 0:
            return

        blocks = self.req_to_blocks[request.request_id]
        block_idx = boundary_tokens // self.block_size
        if block_idx >= len(blocks):
            return
        self.block_pool.cache_partial_block(
            request=request,
            block=blocks[block_idx],
            num_tokens=boundary_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class RSWAManager(FullAttentionManager):
    """KV cache manager for Reference Sliding Window Attention (R-SWA).

    When ``num_prompt_tokens`` is supplied to ``remove_skipped_blocks``, frees
    gap blocks between the prefill tail and the current decode window.  This
    bounds per-request KV memory at O(prefix_len + rswa_window) instead of
    growing linearly with decode length.
    """

    def __init__(self, kv_cache_spec: RSWASpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.rswa_window: int = kv_cache_spec.rswa_window

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Free gap blocks that are no longer needed for attention.

        Gap = blocks entirely within
            [ceil(prefix_len / block_size) * block_size,
             max(prefix_len, processed_computed_tokens - rswa_window))

        Freed blocks are replaced with null_block in req_to_blocks so the
        block_table passed to FA4 is valid (null_block KV is all-zero;
        rswa_mask_mod marks gap positions as non-visible so FA4 skips them).
        """
        if num_prompt_tokens is None:
            super().remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )
            return

        bs = self.block_size
        # First block fully after the prefill boundary.
        first_gap_block = cdiv(num_prompt_tokens, bs)
        # Decode window start position; blocks before this are evictable.
        window_start = max(
            num_prompt_tokens, processed_computed_tokens - self.rswa_window
        )
        last_gap_block = window_start // bs  # exclusive upper bound
        self._remove_blocks_in_range(request_id, first_gap_block, last_gap_block)


class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        # Extra trailing tokens to retain below the window (never attended) so a
        # multi-module MTP store-side lag can still reconstruct the window from
        # cached blocks.
        self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens

    @classmethod
    def _contiguous_blocks_for_hit(
        cls, window_size: int, block_size: int, use_eagle: bool
    ) -> int:
        blocks = cdiv(window_size - 1, block_size)
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            blocks += 1
        return blocks

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."
        # Sliding-window cache hits must stay at the group's physical block
        # granularity. resolve_block_hashes() converts finer-grained hashes to
        # that view when the hybrid-cache alignment is smaller than block_size.
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        # The number of contiguous blocks needed for a prefix cache hit.
        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block
        )

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                    post_pop_blocks = i if drop_eagle_block else i + 1
                    if (post_pop_blocks * block_size) % alignment_tokens != 0:
                        continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if drop_eagle_block and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
            # Re-align after eagle pop: the pop may break the alignment
            # when block_size != alignment_tokens (hybrid models with
            # different page sizes, e.g. Gemma4).
            while (
                block_size != alignment_tokens
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        hit_length = len(computed_blocks[0]) * block_size
        return computed_blocks, hit_length

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
    ) -> list[bool] | None:
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        if alignment_tokens is None:
            # Fast path: when the coordinator imposes no alignment constraint.
            return None
        assert alignment_tokens % kv_cache_spec.block_size == 0

        block_size = kv_cache_spec.block_size
        # Contiguous blocks a hit needs at a boundary (incl. the EAGLE peek).
        need = cls._contiguous_blocks_for_hit(
            window_size=kv_cache_spec.sliding_window,
            block_size=block_size,
            use_eagle=use_eagle,
        )
        # The matched run's right edge sits on the aligned boundary block when
        # EAGLE peeks one block past it (shift=1), otherwise on the last block
        # before the boundary (shift=0).
        shift = 1 if use_eagle else 0

        mask = [False] * (end_block - start_block)

        # (1) Segment-boundary tails. ``retention_interval``:
        #   None -> dense (a tail at every ``alignment_tokens`` boundary);
        #   0    -> no dense tails (only the replay boundary below);
        #   >0   -> a tail once per ``retention_interval``-sized segment.
        segment_tokens = (
            alignment_tokens
            if retention_interval is None
            else (None if retention_interval == 0 else retention_interval)
        )
        if segment_tokens is not None:
            per_segment = segment_tokens // block_size
            if need >= per_segment:
                # Every block is reachable; cache them all.
                return None
            for i in range(start_block, end_block):
                if i >= shift and (i - shift) % per_segment >= per_segment - need:
                    mask[i - start_block] = True

        # (2) Reachable-boundary tails: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction. Both
        # land before segments would cover them under sparse retention, so keep
        # the ``need``-block tail ending on each boundary explicitly.
        if retention_interval is not None:
            for boundary_tokens in reachable_boundaries:
                aligned = boundary_tokens // alignment_tokens * alignment_tokens
                end = aligned // block_size + shift
                for j in range(max(start_block, end - need), min(end_block, end)):
                    mask[j - start_block] = True

        return mask

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        The trailing edge of the window is extended by ``extra_retained_tokens``
        so that those extra trailing tokens' blocks are retained (but not
        attended). This is needed for multi-module spec decoding which can
        re-prefill the last num_spec_prefill_tokens - 1 tokens from the end
        of the sequence, and thus needs to delay freeing/caching of blocks.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        return max(
            0,
            num_computed_tokens - self.sliding_window + 1 - self.extra_retained_tokens,
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        return 0


class CircularBufferManager(FullAttentionManager):
    """Claims the ring's single block per request; prefix caching disabled.

    In the transient pool (``CacheConfig.transient_kv_pool``) the rings come
    from the pool's free list, which every transient group's manager shares:
    the groups' tensors overlay one allocation, so a ring ID may name one
    group's ring at a time, as a shared-pool block ID does. The IDs index the
    transient pool's allocation, never the shared pool's, so they count
    against no shared capacity, are never zeroed with the shared pool's new
    blocks, and go back to the list when their request's blocks are popped
    for freeing.
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        transient_free: list[KVCacheBlock] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        # The transient pool's free rings (transient_ring_free_list), shared
        # with the other transient groups' managers; None without the pool.
        self._transient_free = transient_free

    def _claim_ring_block(self, request_id: str) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        if self._transient_free is not None:
            ring = self._transient_free.pop()
            ring.ref_cnt = 1
            req_blocks.append(ring)
            return [ring]
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        read_only: bool = False,
    ) -> int:
        if self.req_to_blocks.get(request_id):
            return 0
        if self._transient_free is None:
            return 1
        if self._transient_free:
            # From the transient pool: none of the shared pool's blocks.
            return 0
        # No ring free: more than the shared pool holds, so the request waits
        # a step, as Mamba's does for a state another request is writing.
        logger.warning_once(
            "The transient pool has no free ring; requests wait for one. "
            "Rings held: %d.",
            len(self.req_to_blocks),
        )
        return self.block_pool.num_gpu_blocks + 1

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        blocks = super().pop_blocks_for_free(request_id)
        if self._transient_free is None:
            return blocks
        for ring in blocks:
            ring.ref_cnt = 0
            self._transient_free.append(ring)
        return []

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        return self._claim_ring_block(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        self._claim_ring_block(request_id)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        return

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        return

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        return

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens).

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            "chunked local attention groups"
        )
        assert drop_eagle_block is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        hit_length = len(computed_blocks[0]) * kv_cache_spec.block_size
        return computed_blocks, hit_length

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        # Mamba layers use TP instead of DCP, so each rank holds the full
        # recurrent state. Undo the DCP/PCP block_size scaling that the base
        # class applies for attention groups whose KV cache is partitioned.
        self.block_size = kv_cache_spec.block_size
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()
        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            self._num_retired_blocks: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
            # Number of internal checkpoint blocks required by each request's
            # current allocation.
            self._num_checkpoint_blocks: dict[str, int] = {}
            # Requests that registered their own last-prompt-boundary partial
            # tail (producers). On the next step's CoW the boundary state moves
            # into a private cow_block; we record that block for connector
            # offload (see _pending_boundary_state_offloads).
            self._producer_partial_tail_reqs: dict[str, int] = {}
            # CacheConfig.prefix_cache_decode_tail. Per decoding request: the
            # tokens it was (re)scheduled with, known from its first
            # cache_blocks (its prompt; on resuming from preemption, also the
            # outputs it recomputes, whose steps are prefill), the boundary
            # of its latest snapshot, the snapshot due this step, the snapshot
            # allocated this step awaiting its key, and the keyed entry it
            # keeps (retired when the next one is keyed).
            self._decode_tail_prompt: dict[str, int] = {}
            self._decode_tail_last: dict[str, int] = {}
            self._decode_tail_due: dict[str, int] = {}
            self._pending_decode_tails: dict[str, tuple[KVCacheBlock, int]] = {}
            self._decode_tails: dict[
                str, tuple[KVCacheBlock, BlockHashWithGroupId]
            ] = {}
            # decode_tail_mode "recover": per request, the snapshot block the
            # worker writes the state into when a verify window crosses its
            # boundary, and that boundary; the boundary due this step; and the
            # targets to hand the worker (boundary 0 clears one). A target
            # outstanding over a step without drafts is spoiled: that step runs
            # no commit, so the boundary it may cross is never written; the
            # block goes back unkeyed.
            self._recover_tails: dict[str, tuple[KVCacheBlock, int]] = {}
            self._recover_due: dict[str, int] = {}
            self._recover_spoiled: set[str] = set()
            self._tail_updates: dict[str, tuple[int, int]] = {}
            # CacheConfig.prefix_cache_evict_superseded_first. Per request: the
            # boundary up to which its states are never superseded (0 when it
            # resumed a conversation of its own, else its prompt: a new
            # conversation's prompt states may be its siblings'), and its
            # latest cached state -- block, key, boundary -- which is where its
            # conversation resumes until a later one is cached.
            self._superseded_floor: dict[str, int] = {}
            self._latest_states: dict[
                str, tuple[KVCacheBlock, BlockHashWithGroupId, int]
            ] = {}

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        hit_length = 0

        block_size = kv_cache_spec.block_size
        if alignment_tokens < block_size and block_size % alignment_tokens == 0:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            hash_block_size = alignment_tokens
            scale_factor = block_size // hash_block_size
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
                num_tokens = (fine_idx + 1) * hash_block_size
                block_hash = block_hashes[fine_idx]
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    block_idx = fine_idx // scale_factor
                    for computed, cached in zip(computed_blocks, cached_block):
                        computed.extend([block_pool.null_block] * block_idx)
                        computed.append(cached)
                    hit_length = num_tokens
                    break
            return computed_blocks, hit_length

        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                hit_length = (i + 1) * block_size
                break  # we just need the last match - early stopping

        return computed_blocks, hit_length

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
    ) -> list[bool] | None:
        """Sparse Mamba state-snapshot retention.

        ``retention_interval``:

          ``None`` -> dense (cache every block; default, unchanged behavior)
          ``0``    -> keep only the ``reachable_boundaries`` states
          ``> 0``  -> keep one state per ``retention_interval``-sized segment

        ``reachable_boundaries`` are proven reuse points (the replay boundary and
        any cross-request shared-prefix junction, Marconi-style APC); their
        boundary state is always kept so sparse retention does not defeat reuse.
        """
        if retention_interval is None or alignment_tokens is None:
            # Dense caching (default) or no alignment constraint imposed.
            return None
        assert isinstance(kv_cache_spec, MambaSpec)
        block_size = kv_cache_spec.block_size
        mask = [False] * (end_block - start_block)

        # (1) Segment-boundary states. A Mamba hit needs exactly the single
        # state block ending on the boundary (no window, and draft models have
        # no mamba layers, so no eagle shift). Block ``i`` ends at token
        # ``(i + 1) * block_size``.
        segment_tokens = None if retention_interval == 0 else retention_interval
        if segment_tokens is not None:
            per_segment = segment_tokens // block_size
            if per_segment <= 1:
                # Interval at/below the block size: every block is a boundary.
                return None
            first_boundary = (
                start_block + per_segment
            ) // per_segment * per_segment - 1
            for i in range(first_boundary - start_block, len(mask), per_segment):
                mask[i] = True

        # (2) Reachable-boundary states: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction, both
        # of which segments would otherwise skip under sparse retention. A Mamba
        # hit needs exactly the single state block ending on the boundary.
        for boundary_tokens in reachable_boundaries:
            aligned = boundary_tokens // alignment_tokens * alignment_tokens
            boundary_block = aligned // block_size - 1
            if start_block <= boundary_block < end_block:
                mask[boundary_block - start_block] = True

        return mask

    def _remove_blocks_in_range(
        self, request_id: str, first_block: int, last_block: int
    ) -> None:
        if self.mamba_cache_mode != "align":
            return super()._remove_blocks_in_range(request_id, first_block, last_block)
        blocks = self.req_to_blocks.get(request_id, [])
        first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
        last_block = min(last_block, len(blocks))
        if first_block >= last_block:
            return
        freed: list[KVCacheBlock] = []
        # Mamba prefill leaves null gaps between states awaiting retirement.
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i].is_null:
                continue
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)
        self._num_retired_blocks[request_id] = last_block

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        assert isinstance(self.kv_cache_spec, MambaSpec)

        super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
            # The block allocated in the previous step is used to copy Mamba states
            # into the block allocated in the current step; the earlier block is
            # no longer needed and should be freed here.
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # Blocks allocated during prefill may be non-contiguous. Use
            # `last_state_block_idx` to free the appropriate block and replace it
            # with a null block.
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(processed_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        return 0

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        super().add_local_computed_blocks(
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
            num_external_computed_tokens,
        )
        if self.evict_superseded_first and self.mamba_cache_mode == "align":
            # The state the request resumes from is its conversation's latest
            # until the request caches a later one.
            hit = next(
                (b for b in reversed(new_computed_blocks) if not b.is_null), None
            )
            if hit is not None and hit.block_hash is not None:
                self._latest_states[request_id] = (
                    hit,
                    hit.block_hash,
                    num_local_computed_tokens,
                )

    def _note_cached_state(
        self,
        request_id: str,
        block: KVCacheBlock,
        block_hash: BlockHashWithGroupId | None,
        boundary: int,
    ) -> None:
        """A state of the request is cached at ``boundary``
        (``CacheConfig.prefix_cache_evict_superseded_first``). The later of it
        and the request's latest so far is where its conversation now
        resumes; the earlier is superseded."""
        if not self.evict_superseded_first or block_hash is None:
            return
        entry = (block, block_hash, boundary)
        latest = self._latest_states.get(request_id)
        if latest is None or boundary > latest[2]:
            self._latest_states[request_id] = entry
            if latest is not None:
                self._supersede_state(request_id, latest)
        elif boundary < latest[2]:
            self._supersede_state(request_id, entry)

    def _supersede_state(
        self,
        request_id: str,
        state: tuple[KVCacheBlock, BlockHashWithGroupId, int],
    ) -> None:
        """Send a superseded state to the eviction end -- unless it is in a new
        conversation's prompt (its siblings' too, perhaps) or ends a prefix
        another running request holds."""
        block, block_hash, boundary = state
        floor = self._superseded_floor.get(request_id)
        if floor is None or boundary <= floor:
            return
        if self._prefix_shared(request_id, boundary):
            return
        self.block_pool.supersede_cached_block(block, block_hash)

    def _needs_internal_checkpoint(
        self,
        request_id: str,
        num_tokens: int,
        num_computed_tokens: int,
    ) -> bool:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        checkpoint_idx = cdiv(num_tokens, self.block_size) - 2
        blocks = self.req_to_blocks[request_id]
        return (
            self.kv_cache_spec.num_prefill_checkpoint_blocks > 0
            and num_tokens % self.block_size != 0
            and num_computed_tokens % self.block_size == 0
            and checkpoint_idx >= 0
            and (checkpoint_idx >= len(blocks) or blocks[checkpoint_idx].is_null)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        read_only: bool = False,
    ) -> int:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        record = not (apply_admission_cap or read_only)
        if (
            len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            # Mamba can't rely on blocks generated by other requests in the current step
            # To put it in the next step, we return num_gpu_blocks + 1 so
            # that kv_cache_manager will think there is no enough blocks to allocate now
            # and don't schedule it in the current step.
            return self.block_pool.num_gpu_blocks + 1
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += (
                    self.kv_cache_spec.block_size * self.num_speculative_blocks
                )
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
                read_only=read_only,
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model

            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            num_new_blocks = (
                num_required_blocks
                - len(new_computed_blocks)
                - len(self.req_to_blocks[request_id])
            )
            has_partial_hit = (
                self._has_partial_local_hit(
                    new_computed_blocks, num_local_computed_tokens
                )
                or request_id in self._partial_hit_reqs
            )
            if has_partial_hit:
                num_new_blocks = max(num_new_blocks, 0) + 1
            checkpoint_block = int(
                self._needs_internal_checkpoint(
                    request_id, num_tokens, total_computed_tokens
                )
            )
            if record:
                self._num_checkpoint_blocks[request_id] = checkpoint_block
            if num_new_blocks > 0:
                num_new_blocks = 1 + int(has_partial_hit) + checkpoint_block
                if request_id not in self._allocated_block_reqs:
                    num_new_blocks += self.num_speculative_blocks

            num_evictable_computed_blocks = self._get_num_evictable_blocks(
                new_computed_blocks
            )
            decode_tail = self._decode_tail_due_at(
                request_id, total_computed_tokens, num_tokens
            )
            recover_tail = self._recover_tail_due_at(
                request_id, total_computed_tokens, num_tokens
            )
            if record:
                if decode_tail is not None:
                    self._decode_tail_due[request_id] = decode_tail
                else:
                    self._decode_tail_due.pop(request_id, None)
                if recover_tail is not None:
                    self._recover_due[request_id] = recover_tail
                else:
                    self._recover_due.pop(request_id, None)
                if (
                    request_id in self._recover_tails
                    and num_tokens - total_computed_tokens < 2
                ):
                    self._recover_spoiled.add(request_id)
            return (
                num_new_blocks
                + num_evictable_computed_blocks
                + int(decode_tail is not None)
                + int(recover_tail is not None)
            )

    def _decode_tail_due_at(
        self, request_id: str, num_computed_tokens: int, num_tokens: int
    ) -> int | None:
        """The boundary whose running state this step must snapshot before
        its forward moves it on (``CacheConfig.prefix_cache_decode_tail``),
        or None. Only a request already decoding: one allocated before, with
        no copy-on-write of its own pending this step (the snapshot copies the
        running block, which such a copy is still filling), and no lookahead
        (under speculative decoding the running state after a step sits at
        the accepted position, which the scheduler cannot name in advance)."""
        if self.decode_tail_unit is None or self.decode_tail_mode != "copy":
            return None
        prompt = self._decode_tail_prompt.get(request_id)
        if (
            prompt is None
            or num_computed_tokens <= prompt
            or num_computed_tokens % self.decode_tail_unit != 0
            or num_computed_tokens % self.block_size == 0
            or num_tokens != num_computed_tokens + 1
            or self.num_speculative_blocks > 0
            or request_id not in self._allocated_block_reqs
            or request_id in self._partial_hit_reqs
            or self._decode_tail_last.get(request_id) == num_computed_tokens
        ):
            return None
        return num_computed_tokens

    def _recover_tail_due_at(
        self, request_id: str, num_computed_tokens: int, num_tokens: int
    ) -> int | None:
        """Under decode_tail_mode "recover", the boundary this step's verify
        window may cross and which needs a snapshot block, or None: the next
        unit boundary past the tokens already scheduled (so no step in flight
        can cross it before the target reaches the worker), past the tokens
        the request was (re)scheduled with, inside a block (a block
        boundary's state the commit already keeps), and no further than this
        step's last token. Only a verify step, with drafts: past those tokens
        every step is a decode, one token and its drafts, and only a verify
        runs the RecoverSSM commit that writes the target. A request with a
        target outstanding keeps it until its committed tokens pass it."""
        if self.decode_tail_unit is None or self.decode_tail_mode != "recover":
            return None
        prompt = self._decode_tail_prompt.get(request_id)
        if (
            prompt is None
            or num_computed_tokens < prompt
            or num_tokens - num_computed_tokens < 2
            or request_id in self._recover_tails
            or request_id not in self._allocated_block_reqs
            or request_id in self._partial_hit_reqs
        ):
            return None
        unit = self.decode_tail_unit
        boundary = (num_computed_tokens // unit + 1) * unit
        if boundary % self.block_size == 0:
            boundary += unit
        if boundary <= prompt or boundary > num_tokens:
            return None
        return boundary

    def _target_recover_tail(self, request_id: str, boundary: int) -> None:
        """Take a snapshot block for ``boundary`` and hand it to the worker,
        whose RecoverSSM commit writes the state there when the step that
        crosses the boundary commits. Held until the committed tokens pass
        the boundary, then keyed; or freed with the request."""
        snapshot = self.block_pool.get_new_blocks(1)[0]
        self._recover_tails[request_id] = (snapshot, boundary)
        self._tail_updates[request_id] = (boundary, snapshot.block_id)

    def _key_recover_tail(self, request: Request) -> None:
        """Once the request's committed tokens have passed its target, the
        commit that crossed it has written the snapshot: key it at the
        boundary, retire the previous decode tail, and let the block go idle
        (cached); a spoiled one goes back unkeyed. The worker's target is
        cleared."""
        target = self._recover_tails.get(request.request_id)
        if target is None:
            return
        snapshot, boundary = target
        committed = request.num_computed_tokens - request.num_in_flight_tokens
        if committed < boundary:
            return
        del self._recover_tails[request.request_id]
        self._tail_updates[request.request_id] = (0, 0)
        assert self.decode_tail_unit is not None
        key = None
        spoiled = request.request_id in self._recover_spoiled
        self._recover_spoiled.discard(request.request_id)
        if (
            not spoiled
            and len(request.block_hashes) >= boundary // self.decode_tail_unit
        ):
            key = self.block_pool.cache_partial_block(
                request=request,
                block=snapshot,
                num_tokens=boundary,
                kv_cache_group_id=self.kv_cache_group_id,
                block_size=self.block_size,
            )
        if key is not None:
            previous = self._decode_tails.get(request.request_id)
            if previous is not None:
                self.block_pool.retire_cached_block(*previous)
            self._decode_tails[request.request_id] = (snapshot, key)
            self._note_cached_state(request.request_id, snapshot, key, boundary)
        self.block_pool.free_blocks([snapshot])

    def take_decode_tail_updates(self) -> dict[str, tuple[int, int]]:
        if self.mamba_cache_mode != "align":
            return {}
        updates = self._tail_updates
        self._tail_updates = {}
        return updates

    def context_thread_blocks(
        self, request_id: str
    ) -> list[tuple[KVCacheBlock, BlockHashWithGroupId]]:
        if self.mamba_cache_mode != "align":
            return []
        tail = self._decode_tails.get(request_id)
        return [tail] if tail is not None else []

    def _snapshot_decode_tail(self, request_id: str, boundary: int) -> None:
        """Queue a copy of the running state at ``boundary`` into a block of
        its own. The copy runs before this step's forward; ``cache_blocks``
        keys the block right after this allocation. Both copy endpoints stay
        retained until the copy has run -- the snapshot by the ref it is
        allocated with, the running block by one more -- after which the
        snapshot is an idle cached block like any other."""
        blocks = self.req_to_blocks[request_id]
        block_idx = boundary // self.block_size
        if block_idx != len(blocks) - 1 or blocks[block_idx].is_null:
            return  # not where a decoding request keeps its running state
        running = blocks[block_idx]
        snapshot = self.block_pool.get_new_blocks(1)[0]
        running.ref_cnt += 1
        self._pending_cow_copies.append((running, snapshot))
        self._pending_decode_tails[request_id] = (snapshot, boundary)
        self._decode_tail_last[request_id] = boundary

    def _key_decode_tail(self, request: Request) -> None:
        """Key this step's snapshot and retire the request's previous one:
        a follow-up request extends the whole reply, so only the latest
        boundary is worth a block."""
        pending = self._pending_decode_tails.pop(request.request_id, None)
        if pending is None:
            return
        snapshot, boundary = pending
        assert self.decode_tail_unit is not None
        if len(request.block_hashes) < boundary // self.decode_tail_unit:
            # Not keyable yet; the snapshot goes back unkeyed after its copy.
            return
        key = self.block_pool.cache_partial_block(
            request=request,
            block=snapshot,
            num_tokens=boundary,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )
        if key is None:
            return
        # The copy that fills it runs with this step: defer same-step hits.
        self.cached_blocks_this_step.add(key)
        previous = self._decode_tails.get(request.request_id)
        if previous is not None:
            self.block_pool.retire_cached_block(*previous)
        self._decode_tails[request.request_id] = (snapshot, key)
        self._note_cached_state(request.request_id, snapshot, key, boundary)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += self.block_size * self.num_speculative_blocks
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model
            decode_tail = self._decode_tail_due.pop(request_id, None)
            if decode_tail is not None:
                self._snapshot_decode_tail(request_id, decode_tail)
            recover_tail = self._recover_due.pop(request_id, None)
            if recover_tail is not None:
                self._target_recover_tail(request_id, recover_tail)
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]
            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            checkpoint_block = self._num_checkpoint_blocks.get(request_id, 0)
            partial_hit = self._partial_hit_reqs.get(request_id)
            has_partial_hit = partial_hit is not None
            # `num_required_blocks` might be less than `len(req_blocks)` if blocks are
            # over-allocated at last round.
            if num_required_blocks <= len(req_blocks) and not has_partial_hit:
                self._allocated_block_reqs.add(request_id)
                return []
            else:
                prev_block_len = len(req_blocks)
                blocks_allocated = request_id in self._allocated_block_reqs
                # Record the last state block
                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    self.last_state_block_idx[request_id] = prev_block_len - 1

                num_skipped_blocks = (
                    num_required_blocks - self.num_speculative_blocks - 1
                )
                # null blocks
                if prev_block_len < num_skipped_blocks:
                    # minus the internal checkpoint block
                    # so we don't set null for that block
                    null_end = num_skipped_blocks - checkpoint_block
                    req_blocks.extend(
                        [self._null_block for _ in range(prev_block_len, null_end)]
                    )

                if blocks_allocated:
                    # Relocate exclusively owned speculative scratch blocks.
                    for block_idx in range(
                        prev_block_len - self.num_speculative_blocks, prev_block_len
                    ):
                        if block_idx < num_skipped_blocks:
                            self._relocate_speculative_block(req_blocks, block_idx)
                        else:
                            break
                num_new_blocks = num_required_blocks - len(req_blocks)
                if has_partial_hit:
                    num_new_blocks = max(num_new_blocks, 0) + 1
                max_new_blocks = 1 + int(has_partial_hit) + checkpoint_block
                if not blocks_allocated:
                    max_new_blocks += self.num_speculative_blocks
                assert num_new_blocks <= max_new_blocks
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                returned_blocks = req_blocks[prev_block_len:]
                if partial_hit is not None:
                    block_idx, source_block = partial_hit
                    cow_block = new_blocks[0]
                    new_blocks = new_blocks[1:]
                    if blocks_allocated:
                        # The worker block table of a running request is
                        # append-only, so the request must stay on
                        # source_block. Move the cache entry to cow_block
                        # instead; the queued copy fills it before forward
                        # overwrites source_block.
                        assert req_blocks[block_idx] is source_block
                        self.block_pool.move_block_hashes(source_block, cow_block)
                        self._pending_cow_copies.append((source_block, cow_block))
                        source_block.ref_cnt += 1
                        boundary_tokens = self._producer_partial_tail_reqs.pop(
                            request_id, None
                        )
                        if boundary_tokens is not None:
                            self._note_cached_state(
                                request_id,
                                cow_block,
                                cow_block.block_hash,
                                boundary_tokens,
                            )
                        if boundary_tokens is not None:
                            # This CoW preserved a producer's own boundary
                            # state in cow_block; hand it to the connector for
                            # partial-tail offload once the copy has run.
                            self._pending_boundary_state_offloads.append(
                                (
                                    request_id,
                                    self.kv_cache_group_id,
                                    cow_block,
                                    boundary_tokens,
                                )
                            )
                        if cow_block.block_hash is not None:
                            # The moved entry is only filled by this step's
                            # copy, so defer same-step hits on it.
                            self.cached_blocks_this_step.add(cow_block.block_hash)
                    else:
                        self._apply_cow(request_id, block_idx, source_block, cow_block)
                        returned_blocks = [cow_block] + returned_blocks
                req_blocks.extend(new_blocks)
                self._allocated_block_reqs.add(request_id)
                self._partial_hit_reqs.pop(request_id, None)
                returned_blocks.extend(new_blocks)
                return returned_blocks

    def _relocate_speculative_block(
        self, req_blocks: list[KVCacheBlock], block_idx: int
    ) -> None:
        block = req_blocks[block_idx]
        assert self.block_pool.is_block_writable(block), (
            "Speculative Mamba blocks must be exclusively owned and unhashed "
            "before relocation"
        )
        req_blocks.append(block)
        req_blocks[block_idx] = self._null_block

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        recover_block: KVCacheBlock | None = None
        if self.mamba_cache_mode == "align":
            # A "recover" target not yet crossed: its block goes back with the
            # request's own, unkeyed, after any step still in flight (which
            # may yet write into it) has run. The worker drops the target with
            # the request.
            target = self._recover_tails.pop(request_id, None)
            if target is not None:
                recover_block = target[0]
            self._recover_due.pop(request_id, None)
            self._recover_spoiled.discard(request_id)
            self._tail_updates.pop(request_id, None)
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
            self._num_retired_blocks.pop(request_id, None)
            self._num_checkpoint_blocks.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
            # The request's decode tail stays cached for its follow-up; a
            # snapshot not yet keyed goes back unkeyed once its copy has run.
            self._decode_tail_prompt.pop(request_id, None)
            self._decode_tail_last.pop(request_id, None)
            self._decode_tail_due.pop(request_id, None)
            self._pending_decode_tails.pop(request_id, None)
            self._decode_tails.pop(request_id, None)
            # The request's latest state stays cached in its place: its
            # conversation resumes there.
            self._superseded_floor.pop(request_id, None)
            self._latest_states.pop(request_id, None)
            # An offer is only guaranteed to hold committed bytes until the end
            # of the pass that made it. This request's blocks are going back to
            # the pool now, so drop its not-yet-offered hand-offs rather than
            # let a connector claim a block another request may already have
            # been handed.
            self._pending_boundary_state_offloads = [
                entry
                for entry in self._pending_boundary_state_offloads
                if entry[0] != request_id
            ]
        blocks = super().pop_blocks_for_free(request_id)
        if recover_block is not None:
            blocks = [*blocks, recover_block]
        return blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens whose mamba state are not needed anymore. Mamba only
        need to keep the state of the last computed token, so we return
        num_computed_tokens - 1.
        """
        return num_computed_tokens - 1

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
        if self.mamba_cache_mode == "align":
            if self.evict_superseded_first:
                self._superseded_floor.setdefault(
                    request.request_id,
                    0
                    if self._resumed_own.get(request.request_id)
                    else request.num_prompt_tokens,
                )
                blocks = self.req_to_blocks[request.request_id]
                for idx in range(num_cached_blocks_before, num_cached_blocks_after):
                    if not blocks[idx].is_null:
                        self._note_cached_state(
                            request.request_id,
                            blocks[idx],
                            blocks[idx].block_hash,
                            (idx + 1) * self.block_size,
                        )
            partial_hash = self._cache_partial_tail_block(request, num_tokens)
            if partial_hash is not None:
                self.cached_blocks_this_step.add(partial_hash)
            if self.decode_tail_unit is not None:
                self._decode_tail_prompt.setdefault(
                    request.request_id, request.num_tokens
                )
                self._key_decode_tail(request)
                self._key_recover_tail(request)
        if num_cached_blocks_after > num_cached_blocks_before:
            blocks = self.req_to_blocks[request.request_id]
            for idx in range(num_cached_blocks_before, num_cached_blocks_after):
                block = blocks[idx]
                # Skip null blocks (align-mode skipped states) and blocks that
                # were not cached this step — with sparse retention
                # (reachable_block_mask) the intermediate state snapshots carry
                # no hash and must not be recorded as cached-this-step.
                if block.is_null or block.block_hash is None:
                    continue
                self.cached_blocks_this_step.add(block.block_hash)
                if self.mamba_cache_mode == "align":
                    # Offer every retained boundary with its exact block.
                    # The connector filters against its save window, which may
                    # extend past the original prompt during resumed prefill.
                    self._pending_boundary_state_offloads.append(
                        (
                            request.request_id,
                            self.kv_cache_group_id,
                            block,
                            (idx + 1) * self.block_size,
                        )
                    )

    def new_step_starts(self) -> None:
        self.cached_blocks_this_step.clear()

    def _cache_partial_tail_block(
        self,
        request: Request,
        num_tokens: int,
    ) -> BlockHashWithGroupId | None:
        hash_block_size = self.block_pool.hash_block_size
        if self.block_size == hash_block_size:
            return None
        if num_tokens % self.block_size == 0:
            return None
        if num_tokens % hash_block_size != 0:
            return None
        latest_prompt_hash_boundary = (
            request.num_prompt_tokens // hash_block_size
        ) * hash_block_size
        if num_tokens != latest_prompt_hash_boundary:
            return None

        block_idx = num_tokens // self.block_size
        blocks = self.req_to_blocks[request.request_id]
        if block_idx >= len(blocks):
            return None
        source_block = blocks[block_idx]
        if source_block.is_null:
            return None

        partial_hash = self.block_pool.cache_partial_block(
            request=request,
            block=source_block,
            num_tokens=num_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )
        if partial_hash is not None:
            self._partial_hit_reqs[request.request_id] = (block_idx, source_block)
            self.num_cached_block[request.request_id] = block_idx
            # Producer of this partial tail: the boundary state currently lives
            # in ``source_block`` but the next step's forward overwrites it. The
            # upcoming CoW copies it into a durable cow_block; record the req so
            # allocate_new_blocks hands that block to the connector for offload.
            self._producer_partial_tail_reqs[request.request_id] = num_tokens
        return partial_hash


class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # Cross-attention does not use prefix caching / external KV loads.
        return

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        raise NotImplementedError("CrossAttentionManager does not support caching")


class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)


def transient_ring_free_list(num_transient_blocks: int) -> list[KVCacheBlock]:
    """The transient pool's free rings, for the managers of every transient
    group to share; block 0 is held back as the pool's null block (a padded
    batch row's block table reads 0)."""
    return [
        KVCacheBlock(block_id) for block_id in range(num_transient_blocks - 1, 0, -1)
    ]


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_in_flight_tokens: int,
    max_model_len: int,
    transient_free: list[KVCacheBlock] | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """
    Get the appropriate manager for a given KVCacheSpec.

    Uses the KVCacheSpecRegistry to look up the manager class, supporting
    both built-in and custom specs registered via @register_kv_cache_spec
    and KVCacheSpecRegistry.register.

    Args:
        kv_cache_spec: The KVCacheSpec instance
        max_in_flight_tokens: The max tokens scheduled but not yet settled
            (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`
        max_model_len: The maximum context length the model could serve
        transient_free: For a group in the transient pool, the pool's free
            rings, one list for every transient group
            (`transient_ring_free_list`); None otherwise.
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass
    """
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"
    )
    if transient_free is not None:
        assert issubclass(manager_class, CircularBufferManager), (
            f"{manager_class.__name__} cannot take a transient pool"
        )
        kwargs["transient_free"] = transient_free
    # SlidingWindow / ChunkedLocalAttention managers recycle blocks;
    # the runtime admission cap must match the recycling-aware bound the
    # startup pool sizer uses (single source of truth: the spec method).
    # R-SWA also recycles gap blocks but peak physical KV still fits the
    # full-attention bound (prefix + window <= max_model_len), so it inherits
    # FullAttentionSpec sizing without a separate admission cap.
    if isinstance(
        kv_cache_spec,
        (SlidingWindowSpec, ChunkedLocalAttentionSpec),
    ):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


def register_all_kvcache_specs(vllm_config):
    """Built-in spec registration"""
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    KVCacheSpecRegistry.register(
        SlidingWindowSpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )
    KVCacheSpecRegistry.register(
        CircularBufferSpec,
        CircularBufferManager,
        uniform_type_base_spec=CircularBufferSpec,
    )
    KVCacheSpecRegistry.register(
        SlidingWindowMLASpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )

    KVCacheSpecRegistry.register(
        MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec
    )
    KVCacheSpecRegistry.register(
        ChunkedLocalAttentionSpec,
        ChunkedLocalAttentionManager,
        uniform_type_base_spec=ChunkedLocalAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        CrossAttentionSpec,
        CrossAttentionManager,
        uniform_type_base_spec=CrossAttentionSpec,
    )

    # FullAttentionSpec subclasses — grouped with FullAttentionSpec
    KVCacheSpecRegistry.register(
        MLAAttentionSpec, FullAttentionManager, uniform_type_base_spec=FullAttentionSpec
    )
    KVCacheSpecRegistry.register(
        RSWASpec, RSWAManager, uniform_type_base_spec=FullAttentionSpec
    )
    # NOTE(Mengqing): HiddenStateCacheSpec won't take part in
    # grouping, thus the uniform_type_base_spec is just a
    # placeholder.
    KVCacheSpecRegistry.register(
        HiddenStateCacheSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        SinkFullAttentionSpec,
        SinkFullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    from vllm.platforms import current_platform

    current_platform.register_custom_kv_cache_specs(vllm_config)
