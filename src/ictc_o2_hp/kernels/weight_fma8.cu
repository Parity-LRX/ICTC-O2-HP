// Fixed 256x1024, K=10504 FP32 weight GEMM. CUDA 8-product SIMT chains.
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef K_UNROLL
#define K_UNROLL 8
#endif
enum : int { M = 256, N = 1024, K = 10504, BM = 32, BN = TILE_N,
    THREADS = 128, MR = 4, NC = BN / 16, SPAN = 2626 };
struct __align__(NC * 4) Vec { float x[NC]; };
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
__device__ __forceinline__ float add_rn(float a, float b) {
    float c;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}
extern "C" __global__ void weight_fma8(const float* __restrict__ A,
    const float* __restrict__ B, float* __restrict__ P) {
    __shared__ __align__(16) float sa[32 * BM];
    __shared__ __align__(16) float sb[32 * BN];
    const int tid = threadIdx.x, rm = tid / 16, cn = (tid % 16) * NC;
    const long long row0 = (long long)blockIdx.x * BM;
    const long long col0 = (long long)blockIdx.y * BN;
    const long long part = (long long)blockIdx.z, base = part * SPAN;
    float c[4][MR][NC] = {};
    for (int start = 0; start < SPAN; start += 32) {
            for (int idx = tid; idx < 32 * BM; idx += THREADS) {
                const int rr = idx % BM, kk = idx / BM;
                const long long local = (long long)start + kk;
                const long long k = base + local;
                sa[kk * BM + rr] = local < SPAN && k < K
                    ? A[k * M + row0 + rr] : 0.0f;
            }
            for (int idx = tid; idx < 32 * BN / NC; idx += THREADS) {
                const int kk = idx / (BN / NC), cc = (idx % (BN / NC)) * NC;
                const long long local = (long long)start + kk;
                const long long k = base + local;
                Vec v = {};
                if (local < SPAN && k < K)
                    v = *reinterpret_cast<const Vec*>(B + k * N + col0 + cc);
                *reinterpret_cast<Vec*>(sb + kk * BN + cc) = v;
            }
            __syncthreads();
        #pragma unroll
        for (int chain = 0; chain < 4; ++chain) {
            #pragma unroll K_UNROLL
            for (int kk = 0; kk < 8; ++kk) {
                float a[MR];
                #pragma unroll
                for (int r = 0; r < MR; ++r) a[r] = sa[(chain * 8 + kk) * BM + rm + r * 8];
                const Vec b = *reinterpret_cast<const Vec*>(sb + (chain * 8 + kk) * BN + cn);
                #pragma unroll
                for (int r = 0; r < MR; ++r) {
                    #pragma unroll
                    for (int n = 0; n < NC; ++n)
                        c[chain][r][n] = fma_rn(a[r], b.x[n], c[chain][r][n]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < MR; ++r) {
        Vec result;
        #pragma unroll
        for (int n = 0; n < NC; ++n)
            result.x[n] = add_rn(add_rn(c[0][r][n], c[1][r][n]),
                add_rn(c[2][r][n], c[3][r][n]));
        const long long row = row0 + rm + r * 8;
        *reinterpret_cast<Vec*>(P + (part * M + row) * N + col0 + cn) = result;
    }
}
