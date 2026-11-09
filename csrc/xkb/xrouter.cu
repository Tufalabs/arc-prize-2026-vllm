// xrouter: the MoE router of Qwen3.8 e307 in two kernels, replacing cuBLAS's router GEMM plus vLLM's moeSoftmax and moeTopK.
// xr_gemm: P[s][m][n] = sum over k in split s of x[m][k] * w[n][k], fp32 partials, n < NP = 320 (rows of w past E read as zeros).
// xr_topk: one warp a token: sums the S partials, rounds the logit to bf16 (as the bf16 router output does), softmax in fp32, top-10, renormalize.
// The softmax and top-k follow vLLM's moeSoftmax / moeTopK (csrc topk_softmax_kernels.cu): expf(v - max) * (1 / Z), argmax ties to the lower expert, renormalized as p * (1 / sum of the selected p).
// Differences: the GEMM sums in another order than cuBLAS (a logit can land one bf16 step away) and Z is summed in another order (last-bit change of the weights).
#include <stdint.h>
#include <cuda_bf16.h>

#ifndef MODE
#define MODE 0  // 0 real, 1 loads only (no mma), 2 fp16-accumulate mma (timing proxy for full-rate tensor cores)
#endif
#ifndef NST
#define NST 3  // cp.async ring stages of 64 k-values
#endif
#ifndef MT
#define MT 64  // rows a CTA (multiple of 32)
#endif
#ifndef RND
#define RND 1  // 1: round logits to bf16 before the softmax, as vLLM's bf16 router output; 0: keep fp32
#endif
#ifndef TK
#define TK 1  // top-k selection: 1 sorted lanes + redux.sync (two warp reductions a round), 0 ten-value scan + shuffle argmax a round
#endif
#ifndef PDL
#define PDL 0  // 1: griddepcontrol (programmatic dependent launch): xr_gemm prefetches its first w stages before waiting on the previous kernel
#endif

#define NP 320
#define NT 64
#define KCH 64
#define TOPK 10
#define NV (NP / 32)
#define WM (MT / 32)  // m16 blocks a warp: warps 2 (M) x 2 (N), warp tile (MT / 2) x 32

constexpr int A_ST = MT * KCH * 2;
constexpr int B_ST = NT * KCH * 2;
constexpr int ST = A_ST + B_ST;

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if MODE == 2
    uint32_t c0 = __float_as_uint(c[0]), c1 = __float_as_uint(c[1]);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(c0), "+r"(c1)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    c[0] = __uint_as_float(c0);
    c[1] = __uint_as_float(c1);
#elif MODE == 1
    c[0] += __uint_as_float(a[0] ^ b0);  // keeps the loads alive
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Tile rows of 64 bf16 (128 bytes, eight 16-byte chunks); chunk c of row r sits at chunk c ^ (r & 7).
__device__ __forceinline__ uint32_t swz(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

// grid (NP / NT, ceil(M / MT), S), 128 threads; kc = K / S, a multiple of 64.
extern "C" __global__ void __launch_bounds__(128) xr_gemm(const __nv_bfloat16* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ w, int E,
                                                          int M, int K, int kc, float* __restrict__ P) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n0 = blockIdx.x * NT, m0 = blockIdx.y * MT, s = blockIdx.z;
    const int k0 = s * kc, nch = kc / KCH;
    const uint32_t sbase = sa(smem);

    // Loads: thread tid copies chunk (tid & 7) of rows (tid >> 3) + 16 i.
    const int lc = tid & 7, lr = tid >> 3;
    auto load_w = [&](int ch, int st) {
        const uint32_t b = sbase + st * ST + A_ST;
#pragma unroll
        for (int i = 0; i < NT / 16; i++) {
            const int r = lr + 16 * i, n = n0 + r;
            const bool ok = n < E;
            cp16(b + swz(r, lc), w + (size_t)(ok ? n : 0) * K + k0 + ch * KCH + lc * 8, ok);
        }
    };
    auto load_x = [&](int ch, int st) {
        const uint32_t a = sbase + st * ST;
#pragma unroll
        for (int i = 0; i < MT / 16; i++) {
            const int r = lr + 16 * i, m = m0 + r;
            const bool ok = m < M;
            cp16(a + swz(r, lc), x + (size_t)(ok ? m : 0) * ldx + k0 + ch * KCH + lc * 8, ok);
        }
    };

#if PDL
    // w does not depend on the previous kernel: start its stages, then wait for x.
#pragma unroll
    for (int st = 0; st < NST - 1; st++)
        if (st < nch) load_w(st, st);
    asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
    for (int st = 0; st < NST - 1; st++) {
        if (st < nch) load_x(st, st);
        cp_commit();
    }
#else
#pragma unroll
    for (int st = 0; st < NST - 1; st++) {
        if (st < nch) {
            load_w(st, st);
            load_x(st, st);
        }
        cp_commit();
    }
#endif

#if PDL
    // The top-k kernel's griddepcontrol.wait returns only after this whole grid has finished, so its CTAs may start now.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const int wm = wid >> 1, wn = wid & 1;  // warp tile rows wm * MT / 2, cols wn * 32
    float acc[WM][4][4];
#pragma unroll
    for (int i = 0; i < WM; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    // Rows of the warp's m16 blocks that hold only padding skip their mma (warp-uniform).
    const int wrow0 = m0 + wm * (MT / 2);

    for (int ch = 0; ch < nch; ch++) {
        cp_wait<NST - 2>();
        __syncthreads();
        {
            const int nx = ch + NST - 1;
            if (nx < nch) {
                load_w(nx, nx % NST);
                load_x(nx, nx % NST);
            }
            cp_commit();
        }
        const uint32_t a = sbase + (ch % NST) * ST, b = a + A_ST;
#pragma unroll
        for (int kk = 0; kk < KCH / 16; kk++) {
            uint32_t bf[2][4];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int r = wn * 32 + j * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm4(bf[j], b + swz(r, kk * 2 + ((lane >> 3) & 1)));
            }
#pragma unroll
            for (int i = 0; i < WM; i++) {
                if (wrow0 + i * 16 >= M) break;
                uint32_t af[4];
                const int r = wm * (MT / 2) + i * 16 + (lane & 15);
                ldsm4(af, a + swz(r, kk * 2 + (lane >> 4)));
#pragma unroll
                for (int j = 0; j < 4; j++) mma(acc[i][j], af, bf[j >> 1][(j & 1) * 2], bf[j >> 1][(j & 1) * 2 + 1]);
            }
        }
    }

    // Epilogue: fp32 partials, rows < M only.
    float* Ps = P + (size_t)s * M * NP;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < WM; i++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int m = wrow0 + i * 16 + g + 8 * h;
            if (m < M) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = n0 + wn * 32 + j * 8 + 2 * t;
                    *reinterpret_cast<float2*>(Ps + (size_t)m * NP + n) = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                }
            }
        }
    }
}

// grid ceil(M / 4), 128 threads: warp q handles token blockIdx.x * 4 + q; lane l holds experts l + 32 j.
// renorm: vLLM's renormalize flag. pad (optional, may be null): vLLM's is_padding mask, rows flagged get ids -1 (weights as computed), as moeTopK.
// logits (optional, may be null): the bf16-rounded (RND) logits as fp32 [M, NP], for checks.
// S is a template parameter so every partial load of a lane is issued before the first add.
template <int S>
__device__ __forceinline__ void xr_topk_body(const float* __restrict__ P, int M, int E, int renorm, const unsigned char* __restrict__ pad,
                                             float* __restrict__ topk_w, int* __restrict__ topk_ids, float* __restrict__ logits) {
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (m >= M) return;
    float part[S][NV];
#pragma unroll
    for (int s = 0; s < S; s++) {
        const float* p = P + ((size_t)s * M + m) * NP + lane;
#pragma unroll
        for (int j = 0; j < NV; j++) part[s][j] = __ldcg(p + 32 * j);
    }
    const bool is_pad = pad != nullptr && pad[m];
    float v[NV];
#pragma unroll
    for (int j = 0; j < NV; j++) {
        v[j] = part[0][j];
#pragma unroll
        for (int s = 1; s < S; s++) v[j] += part[s][j];
    }
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < NV; j++) {
        const int e = lane + 32 * j;
#if RND
        v[j] = __bfloat162float(__float2bfloat16_rn(v[j]));
#endif
        if (logits) logits[(size_t)m * NP + e] = v[j];
        if (e >= E) v[j] = -INFINITY;
        mx = fmaxf(mx, v[j]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float z = 0.f;
#pragma unroll
    for (int j = 0; j < NV; j++) {
        v[j] = lane + 32 * j < E ? expf(v[j] - mx) : 0.f;
        z += v[j];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    const float rz = 1.f / z;
#if TK == 1
    // Keys: float bits of p + 1 (p >= 0, so integer order is float order), 0 for experts past E; a lane sorts its ten keys
    // (descending, ties to the lower j, which is the lower expert), then each round takes the warp's largest head key
    // (redux max) and, among lanes holding it, the lowest expert (redux min), and the winning lane pops its head.
    uint32_t key[NV];
    int jj[NV];
#pragma unroll
    for (int j = 0; j < NV; j++) {
        float p = v[j] * rz;
        if (isnan(p) || isinf(p)) p = 0.f;
        key[j] = lane + 32 * j < E ? __float_as_uint(p) + 1u : 0u;
        jj[j] = j;
    }
    // Sorting network for 10 (29 compare-exchanges, depth 8).
#define CE(a, b)                                                                               \
    {                                                                                          \
        const bool sw = key[b] > key[a] || (key[b] == key[a] && jj[b] < jj[a]);               \
        const uint32_t tk = sw ? key[b] : key[a];                                              \
        const uint32_t uk = sw ? key[a] : key[b];                                              \
        const int tj = sw ? jj[b] : jj[a];                                                     \
        const int uj = sw ? jj[a] : jj[b];                                                     \
        key[a] = tk; key[b] = uk; jj[a] = tj; jj[b] = uj;                                      \
    }
    CE(4, 9) CE(3, 8) CE(2, 7) CE(1, 6) CE(0, 5)
    CE(1, 4) CE(6, 9) CE(0, 3) CE(5, 8)
    CE(0, 2) CE(3, 6) CE(7, 9)
    CE(0, 1) CE(2, 4) CE(5, 7) CE(8, 9)
    CE(1, 2) CE(4, 6) CE(7, 8) CE(3, 5)
    CE(2, 5) CE(6, 8) CE(1, 3) CE(4, 7)
    CE(2, 3) CE(6, 7)
    CE(3, 4) CE(5, 6)
    CE(4, 5)
#undef CE
    float sel = 0.f, myw = 0.f;
    int myid = 0;
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
        const uint32_t best = __reduce_max_sync(0xffffffffu, key[0]);
        const uint32_t cand = key[0] == best ? (uint32_t)(lane + 32 * jj[0]) : 0xffffffffu;
        const uint32_t be = __reduce_min_sync(0xffffffffu, cand);
        const float bv = best ? __uint_as_float(best - 1u) : 0.f;
        if ((int)(be & 31) == lane) {
#pragma unroll
            for (int j = 0; j < NV - 1; j++) {
                key[j] = key[j + 1];
                jj[j] = jj[j + 1];
            }
            key[NV - 1] = 0u;
        }
        sel += bv;
        if (lane == k) {
            myw = bv;
            myid = (int)be;
        }
    }
#else
    uint32_t taken = 0;
#pragma unroll
    for (int j = 0; j < NV; j++) {
        float p = v[j] * rz;
        if (isnan(p) || isinf(p)) p = 0.f;
        v[j] = p;
        if (lane + 32 * j >= E) taken |= 1u << j;
    }
    float sel = 0.f, myw = 0.f;
    int myid = 0;
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
        float bv = -1.f;
        int be = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < NV; j++)
            if (!(taken >> j & 1) && v[j] > bv) {
                bv = v[j];
                be = lane + 32 * j;
            }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                be = oe;
            }
        }
        if ((be & 31) == lane) taken |= 1u << (be >> 5);
        sel += bv;
        if (lane == k) {
            myw = bv;
            myid = be;
        }
    }
#endif
    if (lane < TOPK) {
        const float scale = renorm ? 1.f / (sel > 0.f ? sel : 1.f) : 1.f;
        topk_w[(size_t)m * TOPK + lane] = myw * scale;
        topk_ids[(size_t)m * TOPK + lane] = is_pad ? -1 : myid;
    }
}

// S: 1, 2, 4, 5, 8 or 10 (other values return without writing).
extern "C" __global__ void __launch_bounds__(128) xr_topk(const float* __restrict__ P, int S, int M, int E, int renorm,
                                                          const unsigned char* __restrict__ pad, float* __restrict__ topk_w,
                                                          int* __restrict__ topk_ids, float* __restrict__ logits) {
#if PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    switch (S) {
        case 1: xr_topk_body<1>(P, M, E, renorm, pad, topk_w, topk_ids, logits); break;
        case 2: xr_topk_body<2>(P, M, E, renorm, pad, topk_w, topk_ids, logits); break;
        case 4: xr_topk_body<4>(P, M, E, renorm, pad, topk_w, topk_ids, logits); break;
        case 5: xr_topk_body<5>(P, M, E, renorm, pad, topk_w, topk_ids, logits); break;
        case 8: xr_topk_body<8>(P, M, E, renorm, pad, topk_w, topk_ids, logits); break;
        case 10: xr_topk_body<10>(P, M, E, renorm, pad, topk_w, topk_ids, logits); break;
        default: break;
    }
}
