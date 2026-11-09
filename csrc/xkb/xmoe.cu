// xmoe: NVFP4 MoE expert kernels for decode-sized batches (Qwen3.8 e307: hidden 2560, intermediate 640, top-10).
// Weights are the A operand of mma.m16n8k16 (rows = output features) and tokens are the 8-wide N operand.
// A work unit is (expert, 256 output rows, up to TMAX tokens) over the full K, so each touched expert's weights stream from DRAM once.
// A persistent CTA per SM runs one producer warp (TMA bulk copies into a ring of stages) and eight consumer warps (dequant + mma).
// Weights use a private layout made by xmoe_rt.repack(): per (expert, unit, k-chunk) 16 KB of A words in the code array and 2 KB of scale bytes in the scale array.
// The two arrays have the byte sizes of the checkpoint's codes and block scales, so the repack can run in place; a stage gets both with two bulk copies.
// Build: nvcc -O3 -arch=sm_120a -cubin -o xmoe.cubin xmoe.cu
#include <cuda_bf16.h>
#include <stdint.h>

#define TOPK 10
#ifndef TMAX
#define TMAX 32
#endif
#define KC 128
#ifndef NSTAGE
#define NSTAGE 3
#endif
#define NCW 8
#ifndef MODE
#define MODE 0  // 0 real; 1 no dequant/mma (memory pipeline only); 2 fp16-accumulate mma (timing proxy for full-rate tensor cores); 3 as 1 without token loads
#endif
#define XSTRIDE (KC * 2 + 16)

constexpr int A_BYTES = 256 * KC / 2;
constexpr int S_BYTES = 256 * KC / 16;
constexpr int W_BYTES = A_BYTES + S_BYTES;
constexpr int X_BYTES = TMAX * XSTRIDE;
constexpr int ST_BYTES = W_BYTES + X_BYTES;
static_assert(ST_BYTES % 16 == 0, "stage alignment");

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(n));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    asm volatile(
        "{\n .reg .pred p;\n"
        "W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra W%=;\n}\n" ::"r"(sa(b)),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(sa(dst)),
        "l"(src), "r"(bytes), "r"(sa(bar)), "l"(pol)
        : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(sa(p)));
    return r;
}
__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if MODE == 2
    uint32_t c0 = __float_as_uint(c[0]), c1 = __float_as_uint(c[1]);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    c[0] = __uint_as_float(c0);
    c[1] = __uint_as_float(c1);
    return;
#endif
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// Scale byte (E5M3, value s * 2^20 after this conversion) -> bf16 pair {s, s}.
template <int J>
__device__ __forceinline__ uint32_t scale2(uint32_t sw) {
    uint32_t d = __byte_perm(sw, 0u, J | (4 << 4) | (J << 8) | (4 << 12));
    return d * 16u + 0x40004000u;
}
// One A word (8 fp4 nibbles) -> four bf16x2 fragment registers, scaled. fp4 v maps to bf16 v * 2^-126.
__device__ __forceinline__ void deq(uint32_t q, uint32_t slo, uint32_t shi, uint32_t* a) {
    const uint32_t S = 0x80008000u, M = 0x01C001C0u;
    uint32_t o0 = (q & S) | ((q >> 6) & M);
    uint32_t o1 = ((q << 4) & S) | ((q >> 2) & M);
    uint32_t o2 = ((q << 8) & S) | ((q << 2) & M);
    uint32_t o3 = ((q << 12) & S) | ((q << 6) & M);
    a[0] = hmul2(o0, slo);
    a[1] = hmul2(o1, shi);
    a[2] = hmul2(o2, slo);
    a[3] = hmul2(o3, shi);
}

struct Pipe {
    uint8_t* smem;
    uint64_t* full;
    uint64_t* empty;
    int* hdr;  // per slot: unit index starting in that slot, or -1 = no more units
};

// GATED: w13 (rows: gate 0..639, up 640..1279), writes act_sorted[pos][640] = silu(gate) * up with vLLM's roundings.
// !GATED: w2 (rows 0..2559), writes out[pair][2560] = bf16(bf16(acc) * bf16(topk_w * oscale)) as Marlin does.
template <bool GATED, int NT>
__device__ __forceinline__ void consume_unit(const Pipe& P, uint32_t& it, int nsk, int e, int rt, int pos0, int cnt,
                                             const float* __restrict__ oscale, const int* __restrict__ sorted_pair,
                                             const float* __restrict__ topk_w, __nv_bfloat16* __restrict__ out) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    float acc[2][NT][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][n][c] = 0.f;

    for (int s = 0; s < nsk; ++s, ++it) {
        const int slot = it % NSTAGE;
        mbar_wait(&P.full[slot], (it / NSTAGE) & 1);
        const uint8_t* st = P.smem + slot * ST_BYTES;
#if MODE != 1 && MODE != 3
        uint4 A[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) A[i] = lds128(st + w * 2048 + i * 512 + lane * 16);
        uint4 Sc[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) Sc[i] = lds128(st + A_BYTES + w * 256 + i * 128 + g * 16);
        const uint32_t xb = sa(st + W_BYTES) + (lane & 7) * XSTRIDE + (lane >> 3) * 16;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t b[NT][4];
#pragma unroll
            for (int n = 0; n < NT; ++n) ldsm4(b[n], xb + n * 8 * XSTRIDE + kk * 64);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int ks = kk * 2 + h;
                const uint32_t sw = word(Sc[ks >> 2], ks & 3);
                uint32_t a0[4], a1[4];
                deq(word(A[kk], h * 2 + 0), scale2<0>(sw), scale2<1>(sw), a0);
                deq(word(A[kk], h * 2 + 1), scale2<2>(sw), scale2<3>(sw), a1);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    mma(acc[0][n], a0, b[n][h * 2], b[n][h * 2 + 1]);
                    mma(acc[1][n], a1, b[n][h * 2], b[n][h * 2 + 1]);
                }
            }
        }
#endif
        __syncwarp();
        if (lane == 0) mbar_arrive(&P.empty[slot]);
    }

    const float osc = oscale[e];
    if constexpr (GATED) {
        const int r0 = rt * 128 + w * 16 + g;
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int col = n * 8 + 2 * t + (c & 1);
                if (col < cnt) {
                    float gv = __bfloat162float(__float2bfloat16(acc[0][n][c] * osc));
                    float uv = __bfloat162float(__float2bfloat16(acc[1][n][c] * osc));
                    float sv = __bfloat162float(__float2bfloat16(gv / (1.0f + expf(-gv))));
                    out[(size_t)(pos0 + col) * 640 + r0 + (c >> 1) * 8] = __float2bfloat16(sv * uv);
                }
            }
    } else {
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int col = n * 8 + 2 * t + (c & 1);
                if (col < cnt) {
                    const int pair = sorted_pair[pos0 + col];
                    const __nv_bfloat16 tw = __float2bfloat16(topk_w[pair] * osc);
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        const int r = rt * 256 + w * 32 + m * 16 + g + (c >> 1) * 8;
                        out[(size_t)pair * 2560 + r] = __hmul(__float2bfloat16(acc[m][n][c]), tw);
                    }
                }
            }
    }
}

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// One persistent kernel for both expert GEMMs. Units come from an atomic counter: first the w13 units (group-major, 5 row tiles of 256),
// then the w2 units (group-major, 10 row tiles of 256). phases: 1 = w13 only, 2 = w2 only, 3 = both, where a w2 unit of group g first
// waits until done[g] counts g's five w13 units, so w2 work fills the w13 tail instead of waiting for a kernel boundary.
extern "C" __global__ void __launch_bounds__((NCW + 1) * 32, 1)
xmoe_mlp(const uint8_t* __restrict__ W13, const uint8_t* __restrict__ S13, const float* __restrict__ osc13, const uint8_t* __restrict__ W2,
         const uint8_t* __restrict__ S2, const float* __restrict__ osc2,
         const __nv_bfloat16* __restrict__ X, const int* __restrict__ sorted_pair, const int4* __restrict__ groups, int* ngroups,
         const float* __restrict__ topk_w, __nv_bfloat16* act, __nv_bfloat16* __restrict__ y, int* done, int phases) {
    extern __shared__ __align__(128) uint8_t smem[];
    Pipe P{smem, reinterpret_cast<uint64_t*>(smem + NSTAGE * ST_BYTES), reinterpret_cast<uint64_t*>(smem + NSTAGE * ST_BYTES) + NSTAGE,
           reinterpret_cast<int*>(smem + NSTAGE * ST_BYTES + 16 * NSTAGE)};
    if (threadIdx.x == 0) {
        for (int i = 0; i < NSTAGE; ++i) {
            mbar_init(&P.full[i], 1);
            mbar_init(&P.empty[i], NCW);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int ng = ngroups[0];
    const int n13 = (phases & 1) ? ng * 5 : 0, nall = n13 + ((phases & 2) ? ng * 10 : 0);
    int* ctr = ngroups + (phases == 2 ? 2 : 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t it = 0;
    if (warp == NCW) {
        const uint64_t pw = policy_evict_first(), px = policy_evict_last();
        for (;;) {
            int u = 0;
            if (lane == 0) u = atomicAdd(ctr, 1);
            u = __shfl_sync(0xffffffffu, u, 0);
            if (u >= nall) break;
            const bool gated = u < n13;
            const int v = gated ? u : u - n13, RT = gated ? 5 : 10, NSK = gated ? 20 : 5, K = gated ? 2560 : 640;
            const int gi = v / RT, rt = v % RT;
            const int4 gp = groups[gi];
            const int e = gp.x, pos0 = gp.y, cnt = gp.z;
            if (!gated && (phases & 1)) {
                if (lane == 0) {
                    while (ld_acquire(&done[gi]) < 5) __nanosleep(128);
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                __syncwarp();
            }
            const size_t chunk0 = ((size_t)e * RT + rt) * NSK;  // first k-chunk of the unit in the code and scale arrays
            const uint8_t* wsrc = (gated ? W13 : W2) + chunk0 * A_BYTES;
            const uint8_t* ssrc = (gated ? S13 : S2) + chunk0 * S_BYTES;
            const __nv_bfloat16* xrow = nullptr;
            if (lane < cnt) {
                const int pos = pos0 + lane;
                xrow = gated ? X + (size_t)(sorted_pair[pos] / TOPK) * K : act + (size_t)pos * K;
            }
            for (int s = 0; s < NSK; ++s, ++it) {
                const int slot = it % NSTAGE;
                if (lane == 0) {
                    mbar_wait(&P.empty[slot], ((it / NSTAGE) & 1) ^ 1);
                    if (s == 0) P.hdr[slot] = u;
                    mbar_expect_tx(&P.full[slot], W_BYTES + (MODE == 3 ? 0 : cnt * KC * 2));
                    bulk_g2s(smem + slot * ST_BYTES, wsrc + (size_t)s * A_BYTES, A_BYTES, &P.full[slot], pw);
                    bulk_g2s(smem + slot * ST_BYTES + A_BYTES, ssrc + (size_t)s * S_BYTES, S_BYTES, &P.full[slot], pw);
                }
                __syncwarp();
                if (MODE != 3 && lane < cnt) bulk_g2s(smem + slot * ST_BYTES + W_BYTES + lane * XSTRIDE, xrow + s * KC, KC * 2, &P.full[slot], px);
            }
        }
        if (lane == 0) {
            const int slot = it % NSTAGE;
            mbar_wait(&P.empty[slot], ((it / NSTAGE) & 1) ^ 1);
            P.hdr[slot] = -1;
            mbar_arrive(&P.full[slot]);
        }
    } else {
        for (;;) {
            const int slot = it % NSTAGE;
            mbar_wait(&P.full[slot], (it / NSTAGE) & 1);
            const int u = P.hdr[slot];
            if (u < 0) break;
            if (u < n13) {
                const int gi = u / 5, rt = u % 5;
                const int4 gp = groups[gi];
                switch ((gp.z + 7) >> 3) {
                    case 1: consume_unit<true, 1>(P, it, 20, gp.x, rt, gp.y, gp.z, osc13, sorted_pair, topk_w, act); break;
                    case 2: consume_unit<true, 2>(P, it, 20, gp.x, rt, gp.y, gp.z, osc13, sorted_pair, topk_w, act); break;
                    case 3: consume_unit<true, 3>(P, it, 20, gp.x, rt, gp.y, gp.z, osc13, sorted_pair, topk_w, act); break;
                    default: consume_unit<true, 4>(P, it, 20, gp.x, rt, gp.y, gp.z, osc13, sorted_pair, topk_w, act); break;
                }
                if (phases & 2) {
                    // publish: all consumer warps' act stores, then one release increment of the group's counter
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    asm volatile("bar.sync 1, %0;" ::"n"(NCW * 32) : "memory");
                    if (threadIdx.x == 0) {
                        __threadfence();
                        atomicAdd(&done[gi], 1);
                    }
                }
            } else {
                const int v = u - n13, gi = v / 10, rt = v % 10;
                const int4 gp = groups[gi];
                switch ((gp.z + 7) >> 3) {
                    case 1: consume_unit<false, 1>(P, it, 5, gp.x, rt, gp.y, gp.z, osc2, sorted_pair, topk_w, y); break;
                    case 2: consume_unit<false, 2>(P, it, 5, gp.x, rt, gp.y, gp.z, osc2, sorted_pair, topk_w, y); break;
                    case 3: consume_unit<false, 3>(P, it, 5, gp.x, rt, gp.y, gp.z, osc2, sorted_pair, topk_w, y); break;
                    default: consume_unit<false, 4>(P, it, 5, gp.x, rt, gp.y, gp.z, osc2, sorted_pair, topk_w, y); break;
                }
            }
        }
    }
}

#define CMAX 2048
// Groups pairs by expert: sorted_pair[pos] lists pair ids (token * TOPK + slot) expert by expert.
// groups[] holds (expert, first pos, count <= TMAX) chunks; experts come in decreasing token count and an expert's chunks are adjacent,
// so the chunks of a busy expert run at the same time (their repeated weight reads meet in L2) and the work ends on single-token experts.
// ngroups[0] = number of groups; ngroups[1], ngroups[2] = unit counters for xmoe_mlp, and done[g] = 0, all reset here.
// Needs every expert's count <= CMAX (rows <= CMAX, as an expert appears once a token).
extern "C" __global__ void __launch_bounds__(1024)
xmoe_route(const int* __restrict__ topk_ids, int npairs, int E, int* __restrict__ sorted_pair, int4* __restrict__ groups,
           int* __restrict__ ngroups, int* __restrict__ done) {
    __shared__ int cnt[512], cur[512], off[512], bcnt[CMAX + 1], bgoff[CMAX + 1];
    const int tid = threadIdx.x;
    for (int i = tid; i < 512; i += blockDim.x) cnt[i] = 0, cur[i] = 0;
    for (int i = tid; i <= CMAX; i += blockDim.x) bcnt[i] = 0;
    __syncthreads();
    for (int p = tid; p < npairs; p += blockDim.x) {
        const int e = topk_ids[p];
        if ((unsigned)e < (unsigned)E) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    for (int e = tid; e < E; e += blockDim.x)
        if (cnt[e]) atomicAdd(&bcnt[min(cnt[e], CMAX)], 1);
    if (tid < 32) {  // pair offsets in expert order
        int c[16], ls = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) c[j] = cnt[tid * 16 + j], ls += c[j];
        int ps = ls;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int a = __shfl_up_sync(0xffffffffu, ps, d);
            if (tid >= d) ps += a;
        }
        ps -= ls;
#pragma unroll
        for (int j = 0; j < 16; ++j) off[tid * 16 + j] = ps, ps += c[j];
    }
    __syncthreads();
    if (tid >= 32 && tid < 64) {  // group offsets per count bucket, largest count first
        const int l = tid - 32, per = (CMAX + 32) / 32;
        int ls = 0;
        for (int j = 0; j < per; ++j) {
            const int b = CMAX - (l * per + j);
            if (b >= 1) ls += bcnt[b] * ((b + TMAX - 1) / TMAX);
        }
        int ps = ls;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int a = __shfl_up_sync(0xffffffffu, ps, d);
            if (l >= d) ps += a;
        }
        ps -= ls;
        for (int j = 0; j < per; ++j) {
            const int b = CMAX - (l * per + j);
            if (b >= 1) bgoff[b] = ps, ps += bcnt[b] * ((b + TMAX - 1) / TMAX);
        }
        if (l == 31) ngroups[0] = ps;
    }
    if (tid == 64) ngroups[1] = 0, ngroups[2] = 0;
    __syncthreads();
    for (int e = tid; e < E; e += blockDim.x) {
        const int c = cnt[e];
        if (!c) continue;
        const int b = min(c, CMAX), per = (b + TMAX - 1) / TMAX;
        int gi = atomicAdd(&bgoff[b], per);
        for (int k = 0; k < c; k += TMAX, ++gi) {
            groups[gi] = make_int4(e, off[e] + k, min(TMAX, c - k), 0);
            done[gi] = 0;
        }
    }
    for (int p = tid; p < npairs; p += blockDim.x) {
        const int e = topk_ids[p];
        if ((unsigned)e < (unsigned)E) sorted_pair[off[e] + atomicAdd(&cur[e], 1)] = p;
    }
}
