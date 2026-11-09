// L2 prefetch kernel for MoE expert weights (TOKS idea 20): fire-and-forget `cp.async.bulk.prefetch.L2.global` over one byte range.
// Each thread issues prefetches of `ch` bytes (a multiple of 16, the range 16-byte aligned) grid-stride over [base, base + bytes); the kernel ends as soon as they are issued, so it holds no SM while the bytes arrive.
// hint 1 marks the lines evict_last so the window's own reads do not push them out before the expert kernel reads them.
// Build: nvcc -O3 -arch=sm_120a -cubin -o pf.cubin pf.cu
#include <cstdint>

extern "C" __global__ void l2pf(const char* base, unsigned long long bytes, unsigned ch, int hint) {
    unsigned long long n = (bytes + ch - 1) / ch;
    uint64_t pol = 0;
    if (hint) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    for (unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (unsigned long long)gridDim.x * blockDim.x) {
        const char* a = base + j * ch;
        unsigned b = (unsigned)(bytes - j * ch < ch ? bytes - j * ch : ch);
        if (hint)
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(a), "r"(b), "l"(pol) : "memory");
        else
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a), "r"(b) : "memory");
    }
}

// Demotes a range's lines back to evict_normal once their reader is done, so evict_last lines from one layer do not pile up in L2 over the next layers.
extern "C" __global__ void l2normal(const char* base, unsigned long long bytes) {
    unsigned long long n = bytes / 128;
    for (unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (unsigned long long)gridDim.x * blockDim.x)
        asm volatile("applypriority.global.L2::evict_normal [%0], 128;" ::"l"(base + j * 128) : "memory");
}

// Paced form: the grid's threads issue chunk j at t0 + j * ns_per_chunk (globaltimer, t0 read by each CTA at its start), so the range arrives at about ch / ns_per_chunk bytes a ns instead of in one burst that slows the kernels running beside it.
// One CTA of one warp issues at most about one 64 KB chunk every 240 ns (about 270 GB/s); CTAs on more SMs raise that limit.
extern "C" __global__ void l2pf_paced(const char* base, unsigned long long bytes, unsigned ch, unsigned ns_per_chunk) {
    unsigned long long n = (bytes + ch - 1) / ch, t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long due = t0 + j * ns_per_chunk, t;
        for (;;) {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            if (t >= due) break;
            __nanosleep(100);
        }
        const char* a = base + j * ch;
        unsigned b = (unsigned)(bytes - j * ch < ch ? bytes - j * ch : ch);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a), "r"(b) : "memory");
    }
}
