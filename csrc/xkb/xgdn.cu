// xgdn: the GDN RecoverSSM fused verify (GDN_FUSED_COMMIT) for sm_120, a CUDA port of the fork's _gdn_recoverssm_fused_verify_kernel.
// One CTA per (value-row tile, request, value head); K = V = 128, bf16 state, float32 records.
// Thread layout: 8 lanes share a value row (lane c holds keys c*8..c*8+7 and 64+c*8..64+c*8+7), so a warp's state load covers 4 full rows of 256 bytes.
// A thread holds RPT rows (rows oc + OCT*i of the tile, oc = tid / 8), RPT*16 state values in float32 registers.
// The window (q, k normalised, v, gates, betas) and the previous step's records sit in shared memory while the state load is in flight.
// Same reads and writes as the Triton kernel: the pending replay and committed-state store, outputs, correction records, key and gate stages, zeros for null-block rows.
// FUSECOPY=1 writes the new key and gate records into the page caches itself (the last CTA of a key head to finish reading writes the keys), replacing the copy kernel.
// Rounding: sums run in a different order than Triton's, so results differ in the last bits ("last-bit rounding").
#include <stdint.h>

#ifndef NT
#define NT 128
#endif
#ifndef RPT
#define RPT 8
#endif
#ifndef MINB
#define MINB 2
#endif
#ifndef TMAX
#define TMAX 4
#endif
#ifndef EXPMODE
#define EXPMODE 0  // 0: ex2.approx.f32 of x*log2(e), as Triton lowers tl.exp; 1: ex2.approx.ftz.f32; 2: expf
#endif
#ifndef DIVMODE
#define DIVMODE 0  // 0: div.full.f32, as Triton lowers fp32 '/'; 1: IEEE division
#endif
#ifndef QREG
#define QREG 1     // 1: q of a token in registers for all rows; 0: reloaded from shared memory a row
#endif
#ifndef FUSECOPY
#define FUSECOPY 0 // 1: write the new key and gate records into the caches directly, no stages and no copy kernel (needs one CTA a head)
#endif

#define KD 128
#define OCT (NT / 8)
#define ROWS (OCT * RPT)
#define NW (NT / 32)
#define VSPLIT (KD / ROWS)
static_assert(KD % ROWS == 0, "rows a CTA must divide 128");
static_assert(RPT <= 8, "lane c stores row c of its rows");
static_assert(!FUSECOPY || VSPLIT == 1, "FUSECOPY needs one CTA a value head");

struct Args {
  const uint16_t *q, *k, *v;
  const void *a, *b;
  const float *A_log;
  const void *dt_bias;
  uint16_t *state;
  float *corr;
  const float *key, *gate;
  float *key_stage, *gate_stage;
  const int *row_pending;
  uint16_t *out;
  const int *qsl, *sidx;
  int *cnt;  // FUSECOPY: arrivals a (row, key head), [rows, key heads] int32, zero between launches
  int64_t s_q, s_k, s_v, s_a, s_b;
  int64_t s_sb, s_sh, s_sv;
  int64_t s_cb, s_ch, s_cp;
  int64_t s_kb, s_kh, s_kp;
  int64_t s_gb, s_gh, s_gp;
  int64_t s_ksr, s_ksh, s_ksp;
  int64_t s_gsr, s_gsh;
  int64_t s_out, s_qsl, s_sidx;
  int64_t null_block, group, bt, flags;  // flags: 1 a/b float32, 2 dt_bias float32
  double scale;
};

extern "C" __device__ int xg_cfg[10] = {NT, RPT, ROWS, VSPLIT, TMAX, MINB, EXPMODE, QREG, FUSECOPY, 0};

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float bf2f(uint16_t x) { return __uint_as_float((uint32_t)x << 16); }
__device__ __forceinline__ uint32_t pack2(float l, float h) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(h), "f"(l));
  return r;
}
__device__ __forceinline__ uint16_t f2bf(float x) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float texp(float x) {
#if EXPMODE == 2
  return expf(x);
#else
  float y, s = x * 1.4426950408889634f;
#if EXPMODE == 1
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(s));
#else
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(s));
#endif
  return y;
#endif
}
__device__ __forceinline__ float tdiv(float a, float b) {
#if DIVMODE == 0
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
#else
  return __fdiv_rn(a, b);
#endif
}
__device__ __forceinline__ uint4 ld_state(const uint16_t* p) { return __ldcs(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ void st_state(uint16_t* p, uint4 v) { __stcs(reinterpret_cast<uint4*>(p), v); }
__device__ __forceinline__ void ld16(const float* s, float* d) {
  // a thread's 16 keys of a 128-key shared row: c*8..c*8+7, then 64+c*8..
  const float4* p = reinterpret_cast<const float4*>(s);
  float4 x0 = p[0], x1 = p[1], x2 = p[16], x3 = p[17];
  d[0] = x0.x; d[1] = x0.y; d[2] = x0.z; d[3] = x0.w; d[4] = x1.x; d[5] = x1.y; d[6] = x1.z; d[7] = x1.w;
  d[8] = x2.x; d[9] = x2.y; d[10] = x2.z; d[11] = x2.w; d[12] = x3.x; d[13] = x3.y; d[14] = x3.z; d[15] = x3.w;
}
__device__ __forceinline__ float oct_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 4);
  return x;
}

extern "C" __global__ void __launch_bounds__(NT, MINB) xgdn_verify(const __grid_constant__ Args p) {
  const int vt = blockIdx.x, rb = blockIdx.y, h = blockIdx.z;
  const int G = (int)p.group, kh = h / G;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, c = tid & 7, oc = tid >> 3;
  const int v0 = vt * ROWS;

  __shared__ __align__(16) float s_q[TMAX][KD];
  __shared__ __align__(16) float s_k[TMAX][KD];
  __shared__ __align__(16) float s_kr[TMAX][KD];
  __shared__ float s_cr[TMAX][ROWS];
  __shared__ float s_v[TMAX][ROWS];
  __shared__ float s_eg[TMAX], s_beta[TMAX], s_egr[TMAX];
#if FUSECOPY
  __shared__ int s_last;
  float gate_new = 0.0f;
#endif

  const int64_t bos = p.qsl[rb * p.s_qsl];
  const int qlen = max(0, min((int)(p.qsl[(rb + 1) * p.s_qsl] - bos), min((int)p.bt, TMAX)));
  const int64_t sidx = p.sidx[rb * p.s_sidx];
  if (sidx <= p.null_block) {
    for (int e = tid; e < qlen * ROWS; e += NT)
      p.out[(bos + e / ROWS) * p.s_out + h * KD + v0 + e % ROWS] = 0;
    return;
  }

  // 1. The state rows, raw bf16, in flight while the window and records load.
  uint16_t* const sbase = p.state + sidx * p.s_sb + (int64_t)h * p.s_sh + c * 8;
  uint4 raw[RPT][2];
#pragma unroll
  for (int i = 0; i < RPT; i++) {
    const uint16_t* row = sbase + (int64_t)(v0 + oc + OCT * i) * p.s_sv;
    raw[i][0] = ld_state(row);
    raw[i][1] = ld_state(row + 64);
  }
  const int pend = max(0, min(p.row_pending[rb], TMAX));

  // 2. The window: a warp normalises a token's q and k; one thread a token computes gate and beta.
  const float scale = (float)p.scale;
  for (int t = warp; t < qlen; t += NW) {
    const int64_t tok = bos + t;
    const uint2 qr = *reinterpret_cast<const uint2*>(p.q + tok * p.s_q + kh * KD + lane * 4);
    const uint2 kr = *reinterpret_cast<const uint2*>(p.k + tok * p.s_k + kh * KD + lane * 4);
    const float qf[4] = {lo(qr.x), hi(qr.x), lo(qr.y), hi(qr.y)};
    const float kf[4] = {lo(kr.x), hi(kr.x), lo(kr.y), hi(kr.y)};
    float sq = qf[0] * qf[0], sk = kf[0] * kf[0];
#pragma unroll
    for (int j = 1; j < 4; j++) { sq = fmaf(qf[j], qf[j], sq); sk = fmaf(kf[j], kf[j], sk); }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) { sq += __shfl_xor_sync(0xffffffffu, sq, m); sk += __shfl_xor_sync(0xffffffffu, sk, m); }
    const float rq = rsqrtf(sq + 1e-6f), rk = rsqrtf(sk + 1e-6f);
    float4 qo, ko;
    qo.x = (qf[0] * rq) * scale; qo.y = (qf[1] * rq) * scale; qo.z = (qf[2] * rq) * scale; qo.w = (qf[3] * rq) * scale;
    ko.x = kf[0] * rk; ko.y = kf[1] * rk; ko.z = kf[2] * rk; ko.w = kf[3] * rk;
    reinterpret_cast<float4*>(s_q[t])[lane] = qo;
    reinterpret_cast<float4*>(s_k[t])[lane] = ko;
  }
  if (tid < qlen) {
    const int64_t tok = bos + tid;
    float ra, rbv, dtb;
    if (p.flags & 1) {
      ra = static_cast<const float*>(p.a)[tok * p.s_a + h];
      rbv = static_cast<const float*>(p.b)[tok * p.s_b + h];
    } else {
      ra = bf2f(static_cast<const uint16_t*>(p.a)[tok * p.s_a + h]);
      rbv = bf2f(static_cast<const uint16_t*>(p.b)[tok * p.s_b + h]);
    }
    dtb = (p.flags & 2) ? static_cast<const float*>(p.dt_bias)[h] : bf2f(static_cast<const uint16_t*>(p.dt_bias)[h]);
    const float negA = -texp(p.A_log[h]);
    const float x = ra + dtb;
    const float sp = x <= 20.0f ? logf(1.0f + texp(x)) : x;
    const float gate = negA * sp;
    s_eg[tid] = texp(gate);
    s_beta[tid] = tdiv(1.0f, 1.0f + texp(-rbv));
#if FUSECOPY
    gate_new = gate;
#else
    if (vt == 0) p.gate_stage[rb * p.s_gsr + h * p.s_gsh + tid] = gate;
#endif
  }
  for (int e = tid; e < qlen * ROWS; e += NT)
    s_v[e / ROWS][e % ROWS] = bf2f(p.v[(bos + e / ROWS) * p.s_v + h * KD + v0 + e % ROWS]);
  // The previous step's records: keys of the key head, this tile's corrections, the head's gates.
  {
    const float* kb = p.key + sidx * p.s_kb + kh * p.s_kh;
    for (int e = tid; e < pend * KD; e += NT) s_kr[e / KD][e % KD] = kb[(e / KD) * p.s_kp + e % KD];
    const float* cb = p.corr + sidx * p.s_cb + h * p.s_ch + v0;
    for (int e = tid; e < pend * ROWS; e += NT) s_cr[e / ROWS][e % ROWS] = cb[(e / ROWS) * p.s_cp + e % ROWS];
    if (tid < pend) s_egr[tid] = texp(p.gate[sidx * p.s_gb + h * p.s_gh + tid * p.s_gp]);
  }
  __syncthreads();
#if FUSECOPY
  // This CTA has read its records. The gate records of head h are read by this CTA alone: overwrite them now.
  // The key records of key head kh are read by its G CTAs: the last of them to arrive writes the new keys, then clears the counter for the next launch.
  if (tid < qlen) const_cast<float*>(p.gate)[sidx * p.s_gb + h * p.s_gh + tid * p.s_gp] = gate_new;
  if (tid == 0) {
    int* cp = p.cnt + rb * (gridDim.z / G) + kh;
    __threadfence();
    const int last = atomicAdd(cp, 1) == G - 1;
    if (last) { *cp = 0; __threadfence(); }
    s_last = last;
  }
  __syncthreads();
  if (s_last) {
    float* kw = const_cast<float*>(p.key) + sidx * p.s_kb + kh * p.s_kh;
    for (int e = tid; e < qlen * KD; e += NT) kw[(e / KD) * p.s_kp + e % KD] = s_k[e / KD][e % KD];
  }
#else
  if (vt == 0 && h % G == 0) {
    float* ks = p.key_stage + rb * p.s_ksr + kh * p.s_ksh;
    for (int e = tid; e < qlen * KD; e += NT) ks[(e / KD) * p.s_ksp + e % KD] = s_k[e / KD][e % KD];
  }
#endif

  // 3. The state in float32.
  float S[RPT][16];
#pragma unroll
  for (int i = 0; i < RPT; i++) {
    const uint32_t w[8] = {raw[i][0].x, raw[i][0].y, raw[i][0].z, raw[i][0].w, raw[i][1].x, raw[i][1].y, raw[i][1].z, raw[i][1].w};
#pragma unroll
    for (int j = 0; j < 8; j++) { S[i][2 * j] = lo(w[j]); S[i][2 * j + 1] = hi(w[j]); }
  }

  // 4. The deferred commit: fma(correction, key, state * exp(gate)) a pending token, then the committed state stored and read back as bf16.
#pragma unroll 1
  for (int t = 0; t < pend; t++) {
    const float eg = s_egr[t];
    float kr[16];
    ld16(&s_kr[t][c * 8], kr);
#pragma unroll
    for (int i = 0; i < RPT; i++) {
      const float cr = s_cr[t][oc + OCT * i];
#pragma unroll
      for (int j = 0; j < 16; j++) S[i][j] = fmaf(cr, kr[j], S[i][j] * eg);
    }
  }
  if (pend > 0) {
#pragma unroll
    for (int i = 0; i < RPT; i++) {
      uint4 w0, w1;
      w0.x = pack2(S[i][0], S[i][1]); w0.y = pack2(S[i][2], S[i][3]); w0.z = pack2(S[i][4], S[i][5]); w0.w = pack2(S[i][6], S[i][7]);
      w1.x = pack2(S[i][8], S[i][9]); w1.y = pack2(S[i][10], S[i][11]); w1.z = pack2(S[i][12], S[i][13]); w1.w = pack2(S[i][14], S[i][15]);
      uint16_t* row = sbase + (int64_t)(v0 + oc + OCT * i) * p.s_sv;
      st_state(row, w0);
      st_state(row + 64, w1);
      const uint32_t w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int j = 0; j < 8; j++) { S[i][2 * j] = lo(w[j]); S[i][2 * j + 1] = hi(w[j]); }
    }
  }

  // 5. The window: decay, correction (v - S k) beta, rank-1 update, output S q; lane c stores row c of its rows.
  uint16_t* const ob = p.out + h * KD + v0;
  float* const cw = p.corr + sidx * p.s_cb + h * p.s_ch + v0;
#pragma unroll 1
  for (int t = 0; t < qlen; t++) {
    const float eg = s_eg[t], beta = s_beta[t];
    float kn[16];
    ld16(&s_k[t][c * 8], kn);
#if QREG
    float qn[16];
    ld16(&s_q[t][c * 8], qn);
#endif
    float o_sel = 0.0f, c_sel = 0.0f;
#pragma unroll
    for (int i = 0; i < RPT; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++) S[i][j] *= eg;
      float d = S[i][0] * kn[0];
#pragma unroll
      for (int j = 1; j < 16; j++) d = fmaf(S[i][j], kn[j], d);
      d = oct_sum(d);
      const float corr = (s_v[t][oc + OCT * i] - d) * beta;
#pragma unroll
      for (int j = 0; j < 16; j++) S[i][j] = fmaf(corr, kn[j], S[i][j]);
#if !QREG
      float qn[16];
      ld16(&s_q[t][c * 8], qn);
#endif
      float o = S[i][0] * qn[0];
#pragma unroll
      for (int j = 1; j < 16; j++) o = fmaf(S[i][j], qn[j], o);
      o = oct_sum(o);
      if (c == i) { o_sel = o; c_sel = corr; }
    }
    if (c < RPT) {
      const int r = oc + OCT * c;
      ob[(bos + t) * p.s_out + r] = f2bf(o_sel);
      cw[t * p.s_cp + r] = c_sel;
    }
  }
}
