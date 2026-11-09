// xhc: the HC sublayer of Qwen3.8 e307 after combine_norm, in three kernels chained with programmatic dependent launch.
// Stock chain it replaces: cuBLAS down GEMM [M, 10240] x [336, 10240]^T (plus cuBLAS's splitK reduce), _hc_silu_kernel, _hc_up_mix_kernel.
// hc_down: P[s][m][n] = sum over k in split s of xn[m][k] * w[n][k], fp32 partials, n < NP (rows of w past N read as zeros).
// hc_red: D = sum of the S partials in split order, rounded to bf16 (as the bf16 GEMM output); lora = bf16(silu(D / 4)) for n < R, inj = D for R <= n < R + ninj.
// hc_up: gate = bf16(lora @ w_up^T) for the 4 streams of each output column, out = bf16(sum over streams of sigmoid(gate) * xn / 4).
// Both GEMMs prefetch their weights (L2 bulk prefetch plus the first cp.async stages) before griddepcontrol.wait, so weight reads overlap the previous kernel.
// Numerics: hc_down sums K in another order than cuBLAS (D can land one bf16 step away); hc_up runs the same mma k-steps in the same order as the Triton kernel.
#include <stdint.h>
#include <cuda_bf16.h>

#ifndef MODE
#define MODE 0  // 0 real, 1 loads only (no mma), 2 fp16-accumulate mma (timing proxy for full-rate tensor cores)
#endif
#ifndef PDL
#define PDL 1
#endif
#ifndef DPF
#define DPF 0  // hc_down: L2 bulk prefetch of the CTA's weight slice before griddepcontrol.wait
#endif
#ifndef UPF
#define UPF 0  // hc_up: same for its weight tile
#endif
#ifndef WS
#define WS 0  // 1: warp-specialized hc_down and hc_up: DPW / UPW producer warps fill the stages with cp.async and signal mbarriers, the mma warps never meet at a block barrier
#endif
#ifndef DPW
#define DPW 2
#endif
#ifndef UPW
#define UPW 2
#endif
#ifndef TILED
#define TILED 0  // timing probe (wrong numbers): read each weight stage as one contiguous block, as a pre-tiled weight layout would
#endif
#ifndef DCL
#define DCL 1  // hc_down (WS builds): CTAs a thread-block cluster along the split dimension; > 1 sums the cluster's fp32 tiles through distributed shared memory, so P holds S / DCL partial sets
#endif
#ifndef DCLP
#define DCLP 1  // cluster reduction: 1 push (each CTA stores its rows of the tile into the owning CTA's shared memory, then sums locally), 0 pull (ld.shared::cluster from every CTA)
#endif
#ifndef DCLX
#define DCLX 0  // timing probe (wrong numbers): 1 skips the cluster sums and partial stores, keeping the barriers; 2 also skips the barriers and pushes
#endif
#ifndef DUPF
#define DUPF 0  // hc_down prefetches a slice of the up weights into L2 (pf, pf_bytes): 1 at its start, 2 halfway through its k-loop
#endif
// hc_down tile: DMT rows x DNT cols, DWM x DWN warps, DKCH k-values a stage, DNST stages.
#ifndef DMT
#define DMT 96
#endif
#ifndef DNT
#define DNT 112
#endif
#ifndef DWM
#define DWM 6
#endif
#ifndef DWN
#define DWN 2
#endif
#ifndef DKCH
#define DKCH 64
#endif
#ifndef DNST
#define DNST 3
#endif
// hc_up tile: UMT rows x UNS output cols (4 * UNS gate cols), UWM x UWN warps.
#ifndef UMT
#define UMT 144
#endif
#ifndef UNS
#define UNS 32
#endif
#ifndef UWM
#define UWM 3
#endif
#ifndef UWN
#define UWN 4
#endif
#ifndef UKCH
#define UKCH 32
#endif
#ifndef UNST
#define UNST 3
#endif

#define HCN 4
#define HD 2560
#define RK 320

typedef __nv_bfloat16 bf16;

__device__ int xhc_cfg[32] = {DMT, DNT, DWM, DWN, DKCH, DNST, UMT, UNS, UWM, UWN, UKCH, UNST, PDL, MODE, DPF, UPF, DUPF, WS, WS ? DPW : 0, WS ? UPW : 0, DCL};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
// Arrives on the mbarrier once all cp.async of this thread issued so far have landed (no pending-count increment: count it in the init).
__device__ __forceinline__ void cp_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void pf_l2(const void* p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm2(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
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
    c[0] += __uint_as_float(a[0] ^ b0);
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Triton's tl.sigmoid on fp32: 1 / (1 + exp(-x)) with exp as ex2.approx of x * log2(e) and the division as div.full.f32.
__device__ __forceinline__ float tri_sigmoid(float x) {
    float e, r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(-x * 1.44269504088896341f));
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(1.f), "f"(1.f + e));
    return r;
}

// Tile rows of KCH bf16; 16-byte chunk c of row r is swizzled so the 8 rows an ldmatrix reads hit distinct banks.
template <int KCH>
__device__ __forceinline__ uint32_t swz(int r, int c) {
    if constexpr (KCH == 64)
        return r * 128 + ((c ^ (r & 7)) << 4);
    else
        return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
}

// One stage of KCH k-values for a warp: IM m16 blocks from A rows arow0 + 16 i (only the first mvalid), NQ n8 tiles from B rows rowb(q).
template <int KCH, int IM, int NQ, class RowB>
__device__ __forceinline__ void mma_stage(float (&acc)[IM][NQ][4], uint32_t a, uint32_t b, int arow0, int mvalid, RowB rowb, int lane) {
#pragma unroll
    for (int kk = 0; kk < KCH / 16; kk++) {
        uint32_t bf[NQ][2];
#pragma unroll
        for (int q = 0; q + 1 < NQ; q += 2) {
            uint32_t r4[4];
            ldsm4(r4, b + swz<KCH>(rowb(q + (lane >> 4)) + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
            bf[q][0] = r4[0];
            bf[q][1] = r4[1];
            bf[q + 1][0] = r4[2];
            bf[q + 1][1] = r4[3];
        }
        if constexpr (NQ & 1) {
            uint32_t r2[2];
            ldsm2(r2, b + swz<KCH>(rowb(NQ - 1) + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
            bf[NQ - 1][0] = r2[0];
            bf[NQ - 1][1] = r2[1];
        }
#pragma unroll
        for (int i = 0; i < IM; i++) {
            if (i >= mvalid) break;
            uint32_t af[4];
            ldsm4(af, a + swz<KCH>(arow0 + i * 16 + (lane & 15), kk * 2 + (lane >> 4)));
#pragma unroll
            for (int q = 0; q < NQ; q++) mma(acc[i][q], af, bf[q][0], bf[q][1]);
        }
    }
}

// ---------------------------------------------------------------- hc_down
constexpr int D_IM = DMT / DWM / 16;
constexpr int D_WN = DNT / DWN;
constexpr int D_NQ = D_WN / 8;
constexpr int D_THREADS = 32 * DWM * DWN;
constexpr int D_A = DMT * DKCH * 2;
constexpr int D_ST = (DMT + DNT) * DKCH * 2;
constexpr int D_RED = DCL > 1 ? DMT * DNT * 4 : 0;                        // fp32 tile a CTA hands to its cluster (reuses the stages)
constexpr int D_DATA = DNST * D_ST > D_RED ? DNST * D_ST : D_RED;        // bytes before the mbarriers
static_assert(DMT % (16 * DWM) == 0 && DNT % (8 * DWN) == 0, "hc_down tile");
static_assert(DCL == 1 || (WS && DMT % DCL == 0 && DNT % 4 == 0), "hc_down cluster reduction");
static_assert(D_DATA + 16 * DNST <= 99 * 1024, "hc_down shared memory");
#if DCL > 1
#define D_CLUSTER __cluster_dims__(1, 1, DCL)
#else
#define D_CLUSTER
#endif
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}
__device__ __forceinline__ void st_cluster2(uint32_t a, float x, float y) { asm volatile("st.shared::cluster.v2.f32 [%0], {%1,%2};" ::"r"(a), "f"(x), "f"(y) : "memory"); }
__device__ __forceinline__ float4 ld_cluster4(uint32_t a) {
    float4 v;
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
    return v;
}

// grid (NP / DNT, ceil(M / DMT), S), D_THREADS threads, DNST * D_ST bytes of shared memory.
// x [M, K] (row stride ldx), w [N, K] row-major; split s covers 64-chunks [s * nch / S, (s + 1) * nch / S); P [S][M][NP] fp32.
// pf (16-byte aligned, pf_bytes a multiple of 16, may be null): bytes to prefetch into L2 for the next GEMM, spread over the CTAs (DUPF).
#if WS
extern "C" __global__ void D_CLUSTER __launch_bounds__(D_THREADS + 32 * DPW) hc_down(const bf16* __restrict__ x, int ldx, const bf16* __restrict__ w, int N, int M, int K,
                                                                          int S, float* __restrict__ P, int NP, const uint8_t* __restrict__ pf, int pf_bytes) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n0 = blockIdx.x * DNT, m0 = blockIdx.y * DMT, s = blockIdx.z;
    const int nchT = K / DKCH, c0 = s * nchT / S, nch = (s + 1) * nchT / S - c0;
    const uint32_t sbase = sa(smem), full = sbase + D_DATA, empty = full + 8 * DNST;
    constexpr int CW = DWM * DWN;
    if (tid == 0) {
        for (int st = 0; st < DNST; st++) {
            mbar_init(full + 8 * st, 32 * DPW);
            mbar_init(empty + 8 * st, CW);
        }
    }
    __syncthreads();
    if (wid >= CW) {
        // producer warps
        const int pt = tid - 32 * CW;
        constexpr int CH = DKCH / 8, RPI = 32 * DPW / CH;
        const int lc = pt % CH, lr = pt / CH;
        auto load_w = [&](int ch, int st) {
            const uint32_t b = sbase + st * D_ST + D_A;
#pragma unroll
            for (int i = 0; i < (DNT + RPI - 1) / RPI; i++) {
                const int r = lr + RPI * i, n = n0 + r;
                if (DNT % RPI == 0 || r < DNT) {
                    if (TILED)
                        cp16(b + swz<DKCH>(r, lc), w + ((size_t)(blockIdx.x * nchT + c0 + ch) * DNT + r) * DKCH + lc * 8, n < N);
                    else
                        cp16(b + swz<DKCH>(r, lc), w + (size_t)(n < N ? n : 0) * K + (c0 + ch) * DKCH + lc * 8, n < N);
                }
            }
        };
        auto load_x = [&](int ch, int st) {
            const uint32_t a = sbase + st * D_ST;
#pragma unroll
            for (int i = 0; i < (DMT + RPI - 1) / RPI; i++) {
                const int r = lr + RPI * i, m = m0 + r;
                if (DMT % RPI == 0 || r < DMT) cp16(a + swz<DKCH>(r, lc), x + (size_t)(m < M ? m : 0) * ldx + (c0 + ch) * DKCH + lc * 8, m < M);
            }
        };
        const int pre = nch < DNST ? nch : DNST;
#if PDL
        for (int st = 0; st < pre; st++) load_w(st, st);
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        for (int st = 0; st < pre; st++) {
            load_x(st, st);
            cp_arrive(full + 8 * st);
        }
#else
        for (int st = 0; st < pre; st++) {
            load_w(st, st);
            load_x(st, st);
            cp_arrive(full + 8 * st);
        }
#endif
        if (DUPF) {
            const int ncta = gridDim.x * gridDim.y * gridDim.z, cta = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
            const int pfsl = ((pf_bytes + ncta - 1) / ncta + 15) & ~15, pf0 = cta * pfsl, pfn = min(pfsl, pf_bytes - pf0);
            if (pf != nullptr && pt == 0 && pfn > 0) pf_l2(pf + pf0, pfn);
        }
        for (int it = DNST; it < nch; it++) {
            const int st = it % DNST;
            mbar_wait(empty + 8 * st, ((it / DNST) - 1) & 1);
            load_w(it, st);
            load_x(it, st);
            cp_arrive(full + 8 * st);
        }
        if (DCL == 1 || DCLX == 2) return;
#if DCLP
        cluster_sync();  // pairs with the consumers' sync before the pushes
#endif
    } else {
#if PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const int wm = wid / DWN, wn = wid % DWN;
    const int arow0 = wm * (DMT / DWM);
    const int rem = M - (m0 + arow0);
    const int mvalid = rem <= 0 ? 0 : (rem + 15) / 16;
    float acc[D_IM][D_NQ][4];
#pragma unroll
    for (int i = 0; i < D_IM; i++)
#pragma unroll
        for (int q = 0; q < D_NQ; q++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][q][e] = 0.f;
    auto rowb = [&](int q) { return wn * D_WN + q * 8; };
    for (int it = 0; it < nch; it++) {
        const int st = it % DNST;
        mbar_wait(full + 8 * st, (it / DNST) & 1);
        const uint32_t a = sbase + st * D_ST;
        mma_stage<DKCH, D_IM, D_NQ>(acc, a, a + D_A, arow0, mvalid, rowb, lane);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + 8 * st);
    }
    const int g = lane >> 2, t = lane & 3;
    if (DCL == 1) {
        float* Ps = P + (size_t)s * M * NP;
#pragma unroll
        for (int i = 0; i < D_IM; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + arow0 + i * 16 + g + 8 * h;
                if (m < M)
#pragma unroll
                    for (int q = 0; q < D_NQ; q++)
                        __stcg(reinterpret_cast<float2*>(Ps + (size_t)m * NP + n0 + wn * D_WN + q * 8 + 2 * t), make_float2(acc[i][q][2 * h], acc[i][q][2 * h + 1]));
            }
        return;
    }
    if (DCLX == 2) {  // probe: no epilogue at all
        float z = 0.f;
#pragma unroll
        for (int i = 0; i < D_IM; i++)
#pragma unroll
            for (int q = 0; q < D_NQ; q++) z += acc[i][q][0] + acc[i][q][1] + acc[i][q][2] + acc[i][q][3];
        if (z == 1.2345e-30f) P[tid] = z;
        return;
    }
    // Cluster reduction: every mma warp is past its last stage before the tile overwrites the stages.
#if DCLP && DCL > 1
    // Push: rows [j * RR, (j + 1) * RR) of this CTA's tile go to slot (own rank) of CTA j; every CTA of the cluster is past its stages
    // (the barrier below orders the remote stores before anyone sums).
    cluster_sync();
    {
        constexpr int RR = DMT / DCL;
        const uint32_t me = cluster_rank();
#pragma unroll
        for (int i = 0; i < D_IM; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = arow0 + i * 16 + g + 8 * h, j = r / RR;
                const uint32_t dst = mapa(sbase, j) + ((me * RR + (r - j * RR)) * DNT + wn * D_WN + 2 * t) * 4;
#pragma unroll
                for (int q = 0; q < D_NQ; q++) st_cluster2(dst + q * 32, acc[i][q][2 * h], acc[i][q][2 * h + 1]);
            }
    }
#else
    asm volatile("bar.sync 1, %0;" ::"n"(D_THREADS) : "memory");
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int i = 0; i < D_IM; i++)
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
            for (int q = 0; q < D_NQ; q++)
                *reinterpret_cast<float2*>(red + (arow0 + i * 16 + g + 8 * h) * DNT + wn * D_WN + q * 8 + 2 * t) = make_float2(acc[i][q][2 * h], acc[i][q][2 * h + 1]);
#endif
    }
#if DCL > 1
    // Rank r sums rows [r * DMT / DCL, (r + 1) * DMT / DCL) of the DCL tiles in rank order and writes them to partial set s / DCL.
    cluster_sync();
    if (!DCLX) {
        constexpr int RR = DMT / DCL, C4 = DNT / 4;
        const uint32_t rank = cluster_rank();
        float* Pg = P + (size_t)(s / DCL) * M * NP;
#if DCLP
        const float* red = reinterpret_cast<const float*>(smem);  // slot j: rows of CTA j
        for (int q = tid; q < RR * C4; q += blockDim.x) {
            const int rr = q / C4, c = (q % C4) * 4, m = m0 + rank * RR + rr;
            if (m >= M) continue;
            float4 v = *reinterpret_cast<const float4*>(red + rr * DNT + c);
#pragma unroll
            for (int j = 1; j < DCL; j++) {
                const float4 u = *reinterpret_cast<const float4*>(red + (j * RR + rr) * DNT + c);
                v.x += u.x;
                v.y += u.y;
                v.z += u.z;
                v.w += u.w;
            }
            __stcg(reinterpret_cast<float4*>(Pg + (size_t)m * NP + n0 + c), v);
        }
#else
        uint32_t rb[DCL];
#pragma unroll
        for (int j = 0; j < DCL; j++) rb[j] = mapa(sbase, j);
        for (int q = tid; q < RR * C4; q += blockDim.x) {
            const int r = rank * RR + q / C4, c = (q % C4) * 4, m = m0 + r;
            if (m >= M) continue;
            const uint32_t off = (r * DNT + c) * 4;
            float4 v = ld_cluster4(rb[0] + off);
#pragma unroll
            for (int j = 1; j < DCL; j++) {
                const float4 u = ld_cluster4(rb[j] + off);
                v.x += u.x;
                v.y += u.y;
                v.z += u.z;
                v.w += u.w;
            }
            __stcg(reinterpret_cast<float4*>(Pg + (size_t)m * NP + n0 + c), v);
        }
#endif
    }
#if !DCLP
    cluster_sync();  // no CTA leaves while the others read its tile
#endif
#endif
}

#else
extern "C" __global__ void __launch_bounds__(D_THREADS) hc_down(const bf16* __restrict__ x, int ldx, const bf16* __restrict__ w, int N, int M, int K,
                                                               int S, float* __restrict__ P, int NP, const uint8_t* __restrict__ pf, int pf_bytes) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n0 = blockIdx.x * DNT, m0 = blockIdx.y * DMT, s = blockIdx.z;
    const int nchT = K / DKCH, c0 = s * nchT / S, nch = (s + 1) * nchT / S - c0;
    // L2 prefetch of slice [pf0, pf0 + pfn) of the up weights, one slice a CTA in 16-byte units.
    const int ncta = gridDim.x * gridDim.y * gridDim.z, cta = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const int pfsl = ((pf_bytes + ncta - 1) / ncta + 15) & ~15, pf0 = cta * pfsl, pfn = min(pfsl, pf_bytes - pf0);
    auto prefetch_up = [&]() {
        if (pf != nullptr && tid == 0 && pfn > 0) pf_l2(pf + pf0, pfn);
    };
    if (DUPF == 1) prefetch_up();
    const uint32_t sbase = sa(smem);
    constexpr int CH = DKCH / 8, RPI = D_THREADS / CH;
    const int lc = tid % CH, lr = tid / CH;
    auto load_w = [&](int ch, int st) {
        const uint32_t b = sbase + st * D_ST + D_A;
#pragma unroll
        for (int i = 0; i < (DNT + RPI - 1) / RPI; i++) {
            const int r = lr + RPI * i, n = n0 + r;
            if (DNT % RPI == 0 || r < DNT) cp16(b + swz<DKCH>(r, lc), w + (size_t)(n < N ? n : 0) * K + (c0 + ch) * DKCH + lc * 8, n < N);
        }
    };
    auto load_x = [&](int ch, int st) {
        const uint32_t a = sbase + st * D_ST;
#pragma unroll
        for (int i = 0; i < (DMT + RPI - 1) / RPI; i++) {
            const int r = lr + RPI * i, m = m0 + r;
            if (DMT % RPI == 0 || r < DMT) cp16(a + swz<DKCH>(r, lc), x + (size_t)(m < M ? m : 0) * ldx + (c0 + ch) * DKCH + lc * 8, m < M);
        }
    };
#if PDL
#if DPF
    for (int r = tid; r < DNT; r += D_THREADS)
        if (n0 + r < N) pf_l2(w + (size_t)(n0 + r) * K + c0 * DKCH, nch * DKCH * 2);
#endif
#pragma unroll
    for (int st = 0; st < DNST - 1; st++)
        if (st < nch) load_w(st, st);
    asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
    for (int st = 0; st < DNST - 1; st++) {
        if (st < nch) load_x(st, st);
        cp_commit();
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#else
#pragma unroll
    for (int st = 0; st < DNST - 1; st++) {
        if (st < nch) {
            load_w(st, st);
            load_x(st, st);
        }
        cp_commit();
    }
#endif
    const int wm = wid / DWN, wn = wid % DWN;
    const int arow0 = wm * (DMT / DWM);
    const int rem = M - (m0 + arow0);
    const int mvalid = rem <= 0 ? 0 : (rem + 15) / 16;
    float acc[D_IM][D_NQ][4];
#pragma unroll
    for (int i = 0; i < D_IM; i++)
#pragma unroll
        for (int q = 0; q < D_NQ; q++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][q][e] = 0.f;
    auto rowb = [&](int q) { return wn * D_WN + q * 8; };
    for (int ch = 0; ch < nch; ch++) {
        cp_wait<DNST - 2>();
        __syncthreads();
        {
            const int nx = ch + DNST - 1;
            if (nx < nch) {
                load_w(nx, nx % DNST);
                load_x(nx, nx % DNST);
            }
            cp_commit();
        }
        if (DUPF == 2 && ch == nch / 2) prefetch_up();
        const uint32_t a = sbase + (ch % DNST) * D_ST;
        mma_stage<DKCH, D_IM, D_NQ>(acc, a, a + D_A, arow0, mvalid, rowb, lane);
    }
    float* Ps = P + (size_t)s * M * NP;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < D_IM; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int m = m0 + arow0 + i * 16 + g + 8 * h;
            if (m < M)
#pragma unroll
                for (int q = 0; q < D_NQ; q++)
                    __stcg(reinterpret_cast<float2*>(Ps + (size_t)m * NP + n0 + wn * D_WN + q * 8 + 2 * t), make_float2(acc[i][q][2 * h], acc[i][q][2 * h + 1]));
        }
}

#endif

// ---------------------------------------------------------------- hc_red
// grid ceil(M * (R + ninj) / 4 / 256), 256 threads: one thread a row and 4 consecutive columns.
// lora [M, R] contiguous; inj [M, ninj] with row stride ld_inj (ninj 0 or 4).
extern "C" __global__ void __launch_bounds__(256) hc_red(const float* __restrict__ P, int S, int M, int NP, int R, int ninj, bf16* __restrict__ lora,
                                                         bf16* __restrict__ inj, int ld_inj) {
#if PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const int nq = (R + ninj) >> 2;
    const int q = blockIdx.x * 256 + threadIdx.x;
    if (q >= M * nq) return;
    const int m = q / nq, c = (q - m * nq) * 4;
    const float* p = P + (size_t)m * NP + c;
    const size_t sst = (size_t)M * NP;
    float4 v = __ldcg(reinterpret_cast<const float4*>(p));
    int s = 1;
    for (; s + 8 <= S; s += 8) {
        float4 u[8];
#pragma unroll
        for (int j = 0; j < 8; j++) u[j] = __ldcg(reinterpret_cast<const float4*>(p + (s + j) * sst));
#pragma unroll
        for (int j = 0; j < 8; j++) {
            v.x += u[j].x;
            v.y += u[j].y;
            v.z += u[j].z;
            v.w += u[j].w;
        }
    }
    for (; s < S; s++) {
        const float4 u = __ldcg(reinterpret_cast<const float4*>(p + s * sst));
        v.x += u.x;
        v.y += u.y;
        v.z += u.z;
        v.w += u.w;
    }
    const bf16 d[4] = {__float2bfloat16_rn(v.x), __float2bfloat16_rn(v.y), __float2bfloat16_rn(v.z), __float2bfloat16_rn(v.w)};
    if (c < R) {
        bf16 o[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float xv = __bfloat162float(d[j]) / (float)HCN;
            o[j] = __float2bfloat16_rn(xv * tri_sigmoid(xv));
        }
        *reinterpret_cast<uint2*>(lora + (size_t)m * R + c) = *reinterpret_cast<const uint2*>(o);
    } else {
        *reinterpret_cast<uint2*>(inj + (size_t)m * ld_inj + (c - R)) = *reinterpret_cast<const uint2*>(d);
    }
}

// ---------------------------------------------------------------- hc_up
constexpr int U_IM = UMT / UWM / 16;
constexpr int U_WC = UNS / UWN;  // output columns a warp
constexpr int U_H = U_WC / 8;    // n8 tiles a stream a warp
constexpr int U_NQ = HCN * U_H;
constexpr int U_THREADS = 32 * UWM * UWN;
constexpr int U_BR = HCN * UNS;
constexpr int U_A = UMT * UKCH * 2;
constexpr int U_ST = (UMT + U_BR) * UKCH * 2;
constexpr int U_XROW = HCN * UNS * 2 + 16;  // xn tile row in shared memory: 4 streams x UNS columns, padded 16 bytes against bank conflicts
constexpr int U_X = UNST * U_ST;             // xn tile offset
constexpr int U_SMEM = U_X + UMT * U_XROW;
static_assert(UMT % (16 * UWM) == 0 && UNS % (8 * UWN) == 0 && RK % UKCH == 0, "hc_up tile");
static_assert(U_SMEM <= 99 * 1024, "hc_up shared memory");

// grid (HD / UNS, ceil(M / UMT)), U_THREADS threads, U_SMEM bytes of shared memory.
// lora [M, RK] contiguous; w [4 * HD, RK] row-major; xn [M, 4 * HD] (row stride ldxn, a multiple of 8); out [M, HD] (row stride ldo).
// The xn tile of the epilogue rides in the last k-chunk's cp.async group; weights of the first stages load before griddepcontrol.wait.
#if WS
extern "C" __global__ void __launch_bounds__(U_THREADS + 32 * UPW) hc_up(const bf16* __restrict__ lora, const bf16* __restrict__ w, const bf16* __restrict__ xn, int ldxn,
                                                                         bf16* __restrict__ out, int ldo, int M) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n0 = blockIdx.x * UNS, m0 = blockIdx.y * UMT;
    constexpr int nch = RK / UKCH, CW = UWM * UWN;
    const uint32_t sbase = sa(smem), full = sbase + U_SMEM, empty = full + 8 * UNST;
    if (tid == 0) {
        for (int st = 0; st < UNST; st++) {
            mbar_init(full + 8 * st, 32 * UPW);
            mbar_init(empty + 8 * st, CW);
        }
    }
    __syncthreads();
    if (wid >= CW) {
        const int pt = tid - 32 * CW;
        constexpr int PT = 32 * UPW, CH = UKCH / 8, RPI = PT / CH;
        const int lc = pt % CH, lr = pt / CH;
        auto load_w = [&](int ch, int st) {
            const uint32_t b = sbase + st * U_ST + U_A;
#pragma unroll
            for (int i = 0; i < (U_BR + RPI - 1) / RPI; i++) {
                const int r = lr + RPI * i;
                if (U_BR % RPI == 0 || r < U_BR) {
                    const int s = r / UNS, c = r - s * UNS;
                    if (TILED)
                        cp16(b + swz<UKCH>(r, lc), w + ((size_t)(blockIdx.x * nch + ch) * U_BR + r) * UKCH + lc * 8, true);
                    else
                        cp16(b + swz<UKCH>(r, lc), w + (size_t)(s * HD + n0 + c) * RK + ch * UKCH + lc * 8, true);
                }
            }
        };
        auto load_a = [&](int ch, int st) {
            const uint32_t a = sbase + st * U_ST;
#pragma unroll
            for (int i = 0; i < (UMT + RPI - 1) / RPI; i++) {
                const int r = lr + RPI * i, m = m0 + r;
                if (UMT % RPI == 0 || r < UMT) cp16(a + swz<UKCH>(r, lc), lora + (size_t)(m < M ? m : 0) * RK + ch * UKCH + lc * 8, m < M);
            }
        };
        auto load_x = [&]() {
            constexpr int XC = HCN * UNS / 8, SC = UNS / 8;
#pragma unroll 4
            for (int q = pt; q < UMT * XC; q += PT) {
                const int r = q / XC, c = q - r * XC, s = c / SC, m = m0 + r;
                if (m < M) cp16(sbase + U_X + r * U_XROW + c * 16, xn + (size_t)m * ldxn + s * HD + n0 + (c - s * SC) * 8, true);
            }
        };
        constexpr int pre = nch < UNST ? nch : UNST;
#if PDL
        for (int st = 0; st < pre; st++) load_w(st, st);
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        for (int st = 0; st < pre; st++) {
            load_a(st, st);
            if (st == nch - 1) load_x();
            cp_arrive(full + 8 * st);
        }
#else
        for (int st = 0; st < pre; st++) {
            load_w(st, st);
            load_a(st, st);
            if (st == nch - 1) load_x();
            cp_arrive(full + 8 * st);
        }
#endif
        for (int it = UNST; it < nch; it++) {
            const int st = it % UNST;
            mbar_wait(empty + 8 * st, ((it / UNST) - 1) & 1);
            load_w(it, st);
            load_a(it, st);
            if (it == nch - 1) load_x();
            cp_arrive(full + 8 * st);
        }
        return;
    }
#if PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    const int wm = wid / UWN, wn = wid % UWN;
    const int arow0 = wm * (UMT / UWM);
    const int rem = M - (m0 + arow0);
    const int mvalid = rem <= 0 ? 0 : (rem + 15) / 16;
    const int g = lane >> 2, t = lane & 3;
    float acc[U_IM][U_NQ][4];
#pragma unroll
    for (int i = 0; i < U_IM; i++)
#pragma unroll
        for (int q = 0; q < U_NQ; q++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][q][e] = 0.f;
    auto rowb = [&](int q) { return (q / U_H) * UNS + wn * U_WC + (q % U_H) * 8; };
#pragma unroll 1
    for (int it = 0; it < nch; it++) {
        const int st = it % UNST;
        mbar_wait(full + 8 * st, (it / UNST) & 1);
        const uint32_t a = sbase + st * U_ST;
        mma_stage<UKCH, U_IM, U_NQ>(acc, a, a + U_A, arow0, mvalid, rowb, lane);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + 8 * st);
    }
    // the last stage's full barrier also covered the xn tile
    const uint8_t* xs = smem + U_X;
#pragma unroll
    for (int i = 0; i < U_IM; i++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int r = arow0 + i * 16 + g + 8 * hh, m = m0 + r;
            if (m >= M) continue;
#pragma unroll
            for (int h = 0; h < U_H; h++) {
                const int col = wn * U_WC + h * 8 + 2 * t;
                float o0 = 0.f, o1 = 0.f;
#pragma unroll
                for (int s = 0; s < HCN; s++) {
                    const float* a2 = &acc[i][s * U_H + h][2 * hh];
                    const float g0 = __bfloat162float(__float2bfloat16_rn(a2[0])), g1 = __bfloat162float(__float2bfloat16_rn(a2[1]));
                    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(xs + r * U_XROW + (s * UNS + col) * 2);
                    o0 += tri_sigmoid(g0) * __bfloat162float(x2.x);
                    o1 += tri_sigmoid(g1) * __bfloat162float(x2.y);
                }
                *reinterpret_cast<__nv_bfloat162*>(out + (size_t)m * ldo + n0 + col) = __floats2bfloat162_rn(o0 / HCN, o1 / HCN);
            }
        }
}

#else
extern "C" __global__ void __launch_bounds__(U_THREADS) hc_up(const bf16* __restrict__ lora, const bf16* __restrict__ w, const bf16* __restrict__ xn, int ldxn,
                                                             bf16* __restrict__ out, int ldo, int M) {
    extern __shared__ __align__(128) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n0 = blockIdx.x * UNS, m0 = blockIdx.y * UMT;
    constexpr int nch = RK / UKCH;
    const uint32_t sbase = sa(smem);
    constexpr int CH = UKCH / 8, RPI = U_THREADS / CH;
    const int lc = tid % CH, lr = tid / CH;
    auto load_w = [&](int ch, int st) {
        const uint32_t b = sbase + st * U_ST + U_A;
#pragma unroll
        for (int i = 0; i < (U_BR + RPI - 1) / RPI; i++) {
            const int r = lr + RPI * i;
            if (U_BR % RPI == 0 || r < U_BR) {
                const int s = r / UNS, c = r - s * UNS;
                cp16(b + swz<UKCH>(r, lc), w + (size_t)(s * HD + n0 + c) * RK + ch * UKCH + lc * 8, true);
            }
        }
    };
    auto load_a = [&](int ch, int st) {
        const uint32_t a = sbase + st * U_ST;
#pragma unroll
        for (int i = 0; i < (UMT + RPI - 1) / RPI; i++) {
            const int r = lr + RPI * i, m = m0 + r;
            if (UMT % RPI == 0 || r < UMT) cp16(a + swz<UKCH>(r, lc), lora + (size_t)(m < M ? m : 0) * RK + ch * UKCH + lc * 8, m < M);
        }
    };
    auto load_x = [&]() {
        constexpr int XC = HCN * UNS / 8, SC = UNS / 8;  // 16-byte chunks a row, a stream
#pragma unroll
        for (int i = 0; i < (UMT * XC + U_THREADS - 1) / U_THREADS; i++) {
            const int q = tid + i * U_THREADS, r = q / XC, c = q - r * XC, s = c / SC, m = m0 + r;
            if ((UMT * XC % U_THREADS == 0 || q < UMT * XC) && m < M)
                cp16(sbase + U_X + r * U_XROW + c * 16, xn + (size_t)m * ldxn + s * HD + n0 + (c - s * SC) * 8, true);
        }
    };
#if PDL
#if UPF
    if (tid < HCN && blockIdx.y == 0) pf_l2(w + (size_t)(tid * HD + n0) * RK, UNS * RK * 2);
#endif
#pragma unroll
    for (int st = 0; st < UNST - 1; st++)
        if (st < nch) load_w(st, st);
    asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
    for (int st = 0; st < UNST - 1; st++) {
        if (st < nch) load_a(st, st);
        if (st == nch - 1) load_x();
        cp_commit();
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#else
#pragma unroll
    for (int st = 0; st < UNST - 1; st++) {
        if (st < nch) {
            load_w(st, st);
            load_a(st, st);
        }
        if (st == nch - 1) load_x();
        cp_commit();
    }
#endif
    const int wm = wid / UWN, wn = wid % UWN;
    const int arow0 = wm * (UMT / UWM);
    const int rem = M - (m0 + arow0);
    const int mvalid = rem <= 0 ? 0 : (rem + 15) / 16;
    const int g = lane >> 2, t = lane & 3;
    float acc[U_IM][U_NQ][4];
#pragma unroll
    for (int i = 0; i < U_IM; i++)
#pragma unroll
        for (int q = 0; q < U_NQ; q++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][q][e] = 0.f;
    auto rowb = [&](int q) { return (q / U_H) * UNS + wn * U_WC + (q % U_H) * 8; };
#pragma unroll 1
    for (int ch = 0; ch < nch; ch++) {
        cp_wait<UNST - 2>();
        __syncthreads();
        {
            const int nx = ch + UNST - 1;
            if (nx < nch) {
                load_w(nx, nx % UNST);
                load_a(nx, nx % UNST);
            }
            if (nx == nch - 1) load_x();
            cp_commit();
        }
        const uint32_t a = sbase + (ch % UNST) * U_ST;
        mma_stage<UKCH, U_IM, U_NQ>(acc, a, a + U_A, arow0, mvalid, rowb, lane);
    }
    cp_wait<0>();
    __syncthreads();
    const uint8_t* xs = smem + U_X;
#pragma unroll
    for (int i = 0; i < U_IM; i++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int r = arow0 + i * 16 + g + 8 * hh, m = m0 + r;
            if (m >= M) continue;
#pragma unroll
            for (int h = 0; h < U_H; h++) {
                const int col = wn * U_WC + h * 8 + 2 * t;
                float o0 = 0.f, o1 = 0.f;
#pragma unroll
                for (int s = 0; s < HCN; s++) {
                    const float* a2 = &acc[i][s * U_H + h][2 * hh];
                    const float g0 = __bfloat162float(__float2bfloat16_rn(a2[0])), g1 = __bfloat162float(__float2bfloat16_rn(a2[1]));
                    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(xs + r * U_XROW + (s * UNS + col) * 2);
                    o0 += tri_sigmoid(g0) * __bfloat162float(x2.x);
                    o1 += tri_sigmoid(g1) * __bfloat162float(x2.y);
                }
                *reinterpret_cast<__nv_bfloat162*>(out + (size_t)m * ldo + n0 + col) = __floats2bfloat162_rn(o0 / HCN, o1 / HCN);
            }
        }
}
#endif
