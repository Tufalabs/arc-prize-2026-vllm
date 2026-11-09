# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client contexts' live threads (``CacheConfig.prefix_cache_context_threads``).

A client names a request's context and the context's history epoch in
``SamplingParams.extra_args``. Per context, the engine keeps the cached blocks
its latest request held: its live thread. What a context no longer needs goes
to the eviction end of the free queue, still cached:

- when a later request of the context frees, the earlier thread's blocks it
  does not hold (the previous decode tail and prompt-end state, the states it
  copied from);
- when a request with a later epoch is scheduled (the client rewrote the
  history), the whole earlier thread, except what the new request took as a
  hit;
- when the client closes the context, or every context under a prefix; a
  request of it the close finds scheduled or still held (a host tier holds a
  finished request's blocks until its copy lands) dies whole at its free.

A block another context took as a hit, or one in use, keeps its place.

A host tier reads the same bookkeeping (``BlockPool.context_threads``): a
finishing request's decode tails, to park its thread whole, and the keys of
every dead thread, to drop them, whether or not the GPU still holds them.
"""

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    request_context,
)

if TYPE_CHECKING:
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    from vllm.v1.request import Request

logger = init_logger(__name__)

Thread = list[tuple[KVCacheBlock, BlockHashWithGroupId]]


class ContextThreads:
    def __init__(
        self,
        block_pool: BlockPool,
        managers: Sequence["SingleTypeKVCacheManager"],
    ) -> None:
        self.block_pool = block_pool
        self.managers = managers
        # Per context: the epoch and the thread of its latest freed request.
        self._threads: dict[str, tuple[int, Thread]] = {}
        # Per scheduled request naming a context: that context and epoch.
        self._requests: dict[str, tuple[str, int]] = {}
        # Scheduled requests whose context a close ended before they freed.
        self._closed: set[str] = set()
        # Blocks sent to the eviction end, by cause.
        self.num_dead: Counter[str] = Counter()
        # Keys of dead threads not yet taken by the host tier (take_dead_keys),
        # each with its context and what ended it.
        self._dead_keys: list[tuple[BlockHashWithGroupId, str, str]] = []

    def on_allocated(
        self,
        request: "Request",
        new_computed_blocks: Sequence[Sequence[KVCacheBlock]],
    ) -> None:
        """After an allocation: mark the blocks the request took as a hit
        that another context cached as shared, and on the request's first
        allocation, end its context's thread if the request's epoch is
        later."""
        tag = request_context(request)
        context = tag[0] if tag is not None else None
        for blocks in new_computed_blocks:
            for block in blocks:
                if block.context is not None and block.context != context:
                    block.context_shared = True
        if tag is None or request.request_id in self._requests:
            return
        self._requests[request.request_id] = tag
        thread = self._threads.get(tag[0])
        if thread is not None and thread[0] < tag[1]:
            hit = {
                block.block_hash
                for blocks in new_computed_blocks
                for block in blocks
                if block.block_hash is not None
            }
            self._kill(
                tag[0], [(b, key) for b, key in thread[1] if key not in hit], "epoch"
            )
            self._threads[tag[0]] = (tag[1], [])

    def before_free(self, request: "Request") -> tuple[str, int, Thread, bool] | None:
        """The context, epoch and thread of a request about to be freed (the
        cached blocks it holds that its context cached), and whether a close
        ended its context while it held them."""
        tag = self._requests.pop(request.request_id, None)
        if tag is None:
            return None
        closed = request.request_id in self._closed
        self._closed.discard(request.request_id)
        context, epoch = tag
        thread: Thread = []
        for manager in self.managers:
            for block in manager.req_to_blocks.get(request.request_id, ()):
                if (
                    not block.is_null
                    and block.context == context
                    and block.block_hash is not None
                ):
                    thread.append((block, block.block_hash))
            thread.extend(manager.context_thread_blocks(request.request_id))
        return context, epoch, thread, closed

    def after_free(self, freed: tuple[str, int, Thread, bool]) -> None:
        """Keep a freed request's thread as its context's; the earlier
        thread's blocks it does not hold are dead. A request a close reached
        before its free is dead whole, and leaves the context to whatever
        request opened it since."""
        context, epoch, thread, closed = freed
        if closed:
            self._kill(context, thread, "close")
            return
        previous = self._threads.get(context)
        if previous is not None and previous[0] > epoch:
            # A request of an epoch its context has left.
            self._kill(context, thread, "epoch")
            return
        if previous is not None and previous[0] == epoch:
            held = {key for _, key in thread}
            self._kill(
                context,
                [(b, key) for b, key in previous[1] if key not in held],
                "replaced",
            )
        self._threads[context] = (epoch, thread)
        # The thread is one unit of recency: its blocks go to the young end of
        # the free queue together, the decode tails freed when they were keyed
        # included.
        for block, key in thread:
            if block.ref_cnt == 0 and block.block_hash == key and not block.dead:
                self.block_pool.free_block_queue.remove(block)
                self.block_pool.free_block_queue.append_n([block])

    def take_dead_keys(self) -> list[tuple[BlockHashWithGroupId, str, str]]:
        """Drain the keys of dead threads for a host tier to drop, each with
        its context and cause ("replaced", "epoch" or "close"): every key of a
        dead thread but the ones another context took as a hit, whether the
        GPU still holds the block or not."""
        keys = self._dead_keys
        self._dead_keys = []
        return keys

    def request_decode_tails(
        self, request_id: str
    ) -> list[tuple[KVCacheBlock, BlockHashWithGroupId]]:
        """The request's keyed decode tails, one per recurrent group, which
        its block list does not hold: what a host tier parking the request's
        thread stores beside its blocks."""
        return [
            tail
            for manager in self.managers
            for tail in manager.context_thread_blocks(request_id)
        ]

    def close(self, prefix: str) -> int:
        """End the context ``prefix``, and every context under it
        (``prefix/...``). Returns the blocks sent to the eviction end; a
        request of them not yet freed sends its blocks there at its free."""

        def ended(context: str) -> bool:
            return context == prefix or context.startswith(prefix + "/")

        contexts = [c for c in self._threads if ended(c)]
        held = [r for r, (c, _) in self._requests.items() if ended(c)]
        self._closed.update(held)
        before = self.num_dead["close"]
        for context in contexts:
            self._kill(context, self._threads.pop(context)[1], "close")
        num_dead = self.num_dead["close"] - before
        logger.info(
            "Closed prefix-cache context %s (%d threads, %d requests not yet "
            "freed): %d blocks to the eviction end. Totals: %s, %d contexts "
            "tracked.",
            prefix,
            len(contexts),
            len(held),
            num_dead,
            dict(self.num_dead),
            len(self._threads),
        )
        return num_dead

    def _kill(self, context: str, thread: Thread, cause: str) -> None:
        for block, key in thread:
            if not (block.block_hash == key and block.context_shared):
                self._dead_keys.append((key, context, cause))
            if self.block_pool.evict_first_context_block(block, key, context):
                self.num_dead[cause] += 1
