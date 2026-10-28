#define B 768
#define W 20
#define DEPTH 512
#define ADDRB 9
#define STEPS 2048
#define NDSP 128
#define PBITLO 8
#define ECONST 12758042340208592819ULL
#define K 1

typedef unsigned int u32; typedef unsigned long long u64;
__device__ __forceinline__ u64 cache_priority_lut_policy() {
    u64 policy;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}
__device__ __forceinline__ u32 cache_priority_lut16(const unsigned short* pointer, u64 policy) {
    u32 value;
    asm("ld.global.nc.L2::cache_hint.u16 %0, [%1], %2;" : "=r"(value) : "l"(pointer), "l"(policy));
    return value;
}

// Immutable RULE data; callers are 8-byte aligned and read four in-bounds u16 values.
__device__ __forceinline__ u64 metadata_rule_load64(const void* pointer) {
    u64 value;
    asm("ld.global.nc.u64 %0, [%1];" : "=l"(value) : "l"(pointer));
    return value;
}

// Every caller is 4-byte aligned and refers to four in-bounds TSH bytes.
__device__ __forceinline__ u32 metadata_load32(const void* pointer) {
    u32 value;
    asm("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(pointer));
    return value;
}
#define M18 ((1u<<18)-1)
#define BANK_TILE 32
// Physical [lane-group][row][lane-within-group], logically unchanged.
#define BANK_ROW(row, lane) ((row) ^ ((((lane) / BANK_TILE) * 17) & (DEPTH - 1)))
#define BANK_INDEX(row, lane) ((((long long)((lane) / BANK_TILE) * DEPTH + BANK_ROW(row, lane)) * BANK_TILE) + ((lane) % BANK_TILE))

// One block per nonce-group; blockDim.x == B (one thread per lane).
// K nonces in flight per block => K independent bank loads/step (MLP).
// 2-step product lag: at step s (parity cur == s&1) the CA taps read
// prodsh[cur] = products of state s-2; AFTER those reads sync, this macro
// overwrites prodsh[cur] with products of state s, to be read at s+2.
#define DSP_LAYER(cur, k)                                                        \
    for (int j = threadIdx.x; j < NDSP; j += B) {                                \
        u64 p = (u64)(sh[cur][k][MULA[j]] & M18)                                 \
              * (((u64)(sh[cur][k][MULB[j]] & M18) << 1) | 1ULL) + (u64)MULC[j]; \
        prodsh[cur][k][j] = (u32)(p >> PBITLO);                                  \
    }

// one CA step for lane b of nonce k -> v  (taps read the ring+product pool)
#define CA_STEP(cur, k, rdk, v) \
{ \
    u32 g = (sh[cur][k][b] ^ (rdk)) & wmask; \
    u32 neighborhood = (g << 1) | (g >> (W-1)) | ((g & 1u) << (W+1)); \
    u32 ng = 0; \
    { \
    for (int s = 0; s < 4; s++) { \
        u32 td = ti_[s]; int c = td >> 10; int ti = td & 1023u; \
        u32 lcr = (neighborhood >> c) & 7u; \
        u32 T = (prodsh[cur][k][ti] >> ts_[s]) & 1; \
        ng |= ((u32)(ru[s] >> ((T<<3)|lcr)) & 1) << c; \
    } \
    } \
    { \
    _Pragma("unroll 1") \
    for (int s = 4; s < 20; s += 4) { \
        u64 rule4 = metadata_rule_load64(ru + s); \
        u32 shift4 = metadata_load32(ts_ + s); \
        { \
        u32 rule = (u32)(rule4 >> 0) & 65535u; \
        u32 td = ti_[s + 0]; int c = td >> 10; int ti = td & 1023u; \
        u32 shift = (shift4 >> 0) & 255u; \
        u32 lcr = (neighborhood >> c) & 7u; \
        u32 T = (sh[cur][k][ti] >> shift) & 1; \
        ng |= ((u32)(rule >> ((T<<3)|lcr)) & 1) << c; \
        } \
        { \
        u32 rule = (u32)(rule4 >> 16) & 65535u; \
        u32 td = ti_[s + 1]; int c = td >> 10; int ti = td & 1023u; \
        u32 shift = (shift4 >> 8) & 255u; \
        u32 lcr = (neighborhood >> c) & 7u; \
        u32 T = (sh[cur][k][ti] >> shift) & 1; \
        ng |= ((u32)(rule >> ((T<<3)|lcr)) & 1) << c; \
        } \
        { \
        u32 rule = (u32)(rule4 >> 32) & 65535u; \
        u32 td = ti_[s + 2]; int c = td >> 10; int ti = td & 1023u; \
        u32 shift = (shift4 >> 16) & 255u; \
        u32 lcr = (neighborhood >> c) & 7u; \
        u32 T = (sh[cur][k][ti] >> shift) & 1; \
        ng |= ((u32)(rule >> ((T<<3)|lcr)) & 1) << c; \
        } \
        { \
        u32 rule = (u32)(rule4 >> 48) & 65535u; \
        u32 td = ti_[s + 3]; int c = td >> 10; int ti = td & 1023u; \
        u32 shift = (shift4 >> 24) & 255u; \
        u32 lcr = (neighborhood >> c) & 7u; \
        u32 T = (sh[cur][k][ti] >> shift) & 1; \
        ng |= ((u32)(rule >> ((T<<3)|lcr)) & 1) << c; \
        } \
    } \
    } \
    (v) = ng & wmask; \
}

extern "C" __global__ void ring_mine(
    u64 base, u32* __restrict__ out_fold, u32* __restrict__ out_state,
    const unsigned short* __restrict__ RULE, const unsigned short* __restrict__ TIDX,
    const unsigned char* __restrict__ TSH,  const unsigned char* __restrict__ ADDRBIT,
    const unsigned short* __restrict__ MULA, const unsigned short* __restrict__ MULB,
    const u32* __restrict__ MULC,
    const u32* __restrict__ prevc, u32* __restrict__ banks, int nhashes)
{
    const u32 wmask = (W>=32)?~0u:((1u<<W)-1);
    const int b = threadIdx.x;
    __shared__ u32 sh[2][K][B];        // ping-pong ring register
    __shared__ u32 prodsh[2][K][NDSP]; // 2-deep product pipe (pre-shifted), slot = step parity
    __shared__ u32 foldsh[K];
    const unsigned short* ru  = &RULE[b*W];
    const unsigned short* ti_ = &TIDX[b*W];
    const unsigned char*  ts_ = &TSH[b*W];
    // Original byte prefix remains intact; aligned u16 suffix is [chunk][value][lane].
    const unsigned short* address_lut =
        reinterpret_cast<const unsigned short*>(ADDRBIT + B*ADDRB) + b;

    const u64 cache_lut_policy = cache_priority_lut_policy();

    for (int g0 = blockIdx.x*K; g0 < nhashes; g0 += gridDim.x*K) {
        u32 rd[K];
        for (int k = 0; k < K; k++) {                       // spec.init_heads
            u64 s = (base + (u64)(g0+k)) ^ ECONST;
            for (int i = 0; i <= b; i++) { s^=s<<13; s^=s>>7; s^=s<<17; }
            sh[0][k][b] = ((u32)(s & wmask)) ^ prevc[b];
            rd[k] = 0;
            for (int j = threadIdx.x; j < NDSP; j += B)     // pipe = 0 for steps 0,1
                prodsh[0][k][j] = prodsh[1][k][j] = 0;
        }
        __syncthreads();
        int cur = 0;

        // ---- FILL: same dynamics, read masked to 0, write row = step ----
        for (int st = 0; st < DEPTH; st++) {
            u32 vv[K];
            for (int k = 0; k < K; k++) CA_STEP(cur, k, 0u, vv[k])
            __syncthreads();                                // prodsh[cur] fully read
            for (int k = 0; k < K; k++) {                   // entry-major: coalesced
                u32* bk = &banks[((long long)(blockIdx.x*K+k))*B*DEPTH];
                bk[BANK_INDEX(st, b)] = vv[k];
            }
            for (int k = 0; k < K; k++) DSP_LAYER(cur, k)   // products of state st -> read at st+2
            for (int k = 0; k < K; k++) sh[cur^1][k][(b+1)==B?0:(b+1)] = vv[k];
            __syncthreads();
            cur ^= 1;
        }

        // ---- RUN: data-dependent read-modify-write ----
        for (int st = 0; st < STEPS; st++) {
            u32 vv[K], aa[K];
            for (int k = 0; k < K; k++) {
                u32 v; CA_STEP(cur, k, rd[k], v)
                vv[k] = v;
                u32 a = 0;
                a |= (u32)cache_priority_lut16(&address_lut[(0u + ((v >> 0) & 127u))*B], cache_lut_policy);
                a |= (u32)cache_priority_lut16(&address_lut[(128u + ((v >> 7) & 127u))*B], cache_lut_policy);
                a |= (u32)cache_priority_lut16(&address_lut[(256u + ((v >> 14) & 63u))*B], cache_lut_policy);
                aa[k] = a;
            }
            __syncthreads();                                // prodsh[cur] fully read
            for (int k = 0; k < K; k++) DSP_LAYER(cur, k)   // publish shared state before the exchange
            for (int k = 0; k < K; k++) sh[cur^1][k][(b+1)==B?0:(b+1)] = vv[k];
            __syncthreads();
            for (int k = 0; k < K; k++) {                   // K exchanges issued together (MLP)
                u32* bk = &banks[((long long)(blockIdx.x*K+k))*B*DEPTH];
                // atomicExch fuses the RMW's load and store into ONE memory
                // transaction (returns the old word while writing the new) --
                // the address is touched once per step instead of twice.
                // +5.7% on an RTX 3070 over the separate __ldlu-load + store
                // it replaced (second adversarial pass, 2026-08-05).
                rd[k] = atomicExch(&bk[BANK_INDEX(aa[k], b)], vv[k]);
            }
            cur ^= 1;
        }

        if (b == 0) for (int k = 0; k < K; k++) foldsh[k] = 0;
        __syncthreads();
        int off = (b*W) & 31;                               // spec.difficulty_value
        for (int k = 0; k < K; k++) {
            u32 fin = sh[cur][k][b];
            atomicXor(&foldsh[k], off ? ((fin<<off)|(fin>>(32-off))) : fin);
        }
        __syncthreads();
        if (out_state) for (int k = 0; k < K; k++)      // full final ring state (verify path)
            out_state[((long long)(g0+k))*B + b] = sh[cur][k][b];
        if (b == 0) for (int k = 0; k < K; k++) out_fold[g0+k] = foldsh[k];
        __syncthreads();
    }
}
