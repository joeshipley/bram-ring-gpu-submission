#!/usr/bin/env python3
"""
mine.py — the BRAM-ring on CUDA. The best-effort GPU attack.

ONE BLOCK PER
NONCE, B threads (one per lane), the ring register in shared memory, K nonces
per block for memory-level parallelism, banks entry-major so the fill coalesces.
Kernel structure:

  * B=768 threads/block (one per lane), DEPTH=512, STEPS=2048.
  * THE DSP LAYER, 2-STEP LAG (mirrors the DSP pipeline registers). Each step, NDSP
    products of epoch-chosen lane pairs are computed cooperatively (threads
    0..NDSP-1) into a TWO-DEEP shared-memory pipe; taps read the products of
    the ring state TWO steps back (zero for steps 0,1) — mirroring the DSP48
    operand+output registers that let the FPGA close timing. The pipe is a
    ping-pong indexed by the step parity, which is exactly the ring buffer
    index `cur`. Products are stored pre-shifted by PBITLO so the tapped
    bits fit u32.
  * THE FILL IS THE FUNCTION. The first DEPTH steps run the identical CA+DSP
    dynamics with the RAM read masked to zero and the write address forced to
    the row counter -- no separate xorshift fill.
  * Taps read a POOL: index < B is a ring lane, >= B is a product.

Verifies bit-exact against vectors.txt, then benchmarks the mining rate.
The honest ceiling here is memory bandwidth, not this code: each nonce owns a
1.57 MB bank hit with dependent, uncoalesced random RMW. If you can make this
faster while staying bit-exact, that is the most valuable contribution to the
repo -- the moat claim rests on this number being the best anyone can do.

  python3 mine.py            # verify + benchmark
  python3 mine.py warmup     # just compile the kernels
  python3 mine.py vectors | python3 ../host/check_vectors.py   # the 4-column proof a claimant ships
"""
import sys, os, time, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cupy as cp
import spec
from spec import B, W, DEPTH, ADDR, STEPS, NDSP, PBITLO, EPOCH

assert W <= 32 and B <= 1024, "one thread per lane: B must fit a CUDA block"
RULE, TAP, (MULA, MULB, MULC), ADDR_BITS = spec.epoch_tables(EPOCH)
ECONST = spec.epoch_const(EPOCH)
PREVCHUNK = np.array([sum(((spec.PREV >> ((b*W + c) % 256)) & 1) << c for c in range(W))
                      for b in range(B)], dtype=np.uint32)

# taps -> a flat pool index + shift (pool: [0,B) ring lanes, [B,B+NDSP) products)
TIDX = np.zeros(B*W, dtype=np.uint16)
TSH  = np.zeros(B*W, dtype=np.uint8)
for b in range(B):
    for c in range(W):
        kind, tx, ty = TAP[b*W + c]
        TIDX[b*W + c] = (B + tx) if kind else tx
        TSH[b*W + c]  = (ty - PBITLO) if kind else ty
ADDRBIT = np.array([p for bits in ADDR_BITS for p in bits], dtype=np.uint8)
RULEf   = np.array(RULE, dtype=np.uint16)
MULAf   = np.array(MULA, dtype=np.uint16)
MULBf   = np.array(MULB, dtype=np.uint16)
MULCf   = np.array(MULC, dtype=np.uint32)

MINE_K = int(os.environ.get("MINE_K", "1"))   # nonces per block (MLP lever; K=1 measured best with the fused atomicExch RMW on RTX 3070 -- the exchange already supplies the memory-level parallelism K used to buy)

DEFS = (f"#define B {B}\n#define W {W}\n#define DEPTH {DEPTH}\n#define ADDRB {ADDR}\n"
        f"#define STEPS {STEPS}\n#define NDSP {NDSP}\n#define PBITLO {PBITLO}\n"
        f"#define ECONST {ECONST}ULL\n#define K {MINE_K}\n")

BODY = r"""
typedef unsigned int u32; typedef unsigned long long u64;
#define M18 ((1u<<18)-1)

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
#define CA_STEP(cur, k, rdk, v)                                                  \
{                                                                                \
    u32 g = (sh[cur][k][b] ^ (rdk)) & wmask;                                     \
    u32 ng = 0;                                                                  \
    for (int c = 0; c < W; c++) {                                                \
        int cl = (c==0)?W-1:c-1, cr = (c==W-1)?0:c+1;                            \
        u32 L=(g>>cl)&1, C=(g>>c)&1, R=(g>>cr)&1;                                \
        int ti = ti_[c]; int ts = ts_[c];                                        \
        u32 T = ((ti < B) ? sh[cur][k][ti] : prodsh[cur][k][ti-B]) >> ts & 1;    \
        ng |= ((u32)(ru[c] >> ((T<<3)|(R<<2)|(C<<1)|L)) & 1) << c;               \
    }                                                                            \
    (v) = ng & wmask;                                                            \
}

extern "C" __global__ void __launch_bounds__(B, 2) ring_mine(
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
    const unsigned char*  ab  = &ADDRBIT[b*ADDRB];

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
                u32* bk = &banks[((long long)(g0+k))*B*DEPTH];
                bk[(long long)st*B + b] = vv[k];
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
                for (int q = 0; q < ADDRB; q++) a |= ((v >> ab[q]) & 1) << q;
                aa[k] = a;
            }
            __syncthreads();                                // prodsh[cur] fully read
            for (int k = 0; k < K; k++) {                   // K exchanges issued together (MLP)
                u32* bk = &banks[((long long)(g0+k))*B*DEPTH];
                // atomicExch fuses the RMW's load and store into ONE memory
                // transaction (returns the old word while writing the new) --
                // the address is touched once per step instead of twice.
                // +5.7% on an RTX 3070 over the separate __ldlu-load + store
                // it replaced (second adversarial pass, 2026-08-05).
                rd[k] = atomicExch(&bk[(long long)aa[k]*B + b], vv[k]);
            }
            for (int k = 0; k < K; k++) DSP_LAYER(cur, k)   // int work overlaps the atomics
            for (int k = 0; k < K; k++) sh[cur^1][k][(b+1)==B?0:(b+1)] = vv[k];
            __syncthreads();
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
"""

_mod = cp.RawModule(code=DEFS + BODY, options=("-std=c++14",))
_mine = _mod.get_function("ring_mine")

d_RULE = cp.asarray(RULEf);   d_TIDX = cp.asarray(TIDX);   d_TSH = cp.asarray(TSH)
d_AB   = cp.asarray(ADDRBIT); d_MULA = cp.asarray(MULAf);  d_MULB = cp.asarray(MULBf)
d_MULC = cp.asarray(MULCf);   d_PREV = cp.asarray(PREVCHUNK)


def _launch(base, nhashes, nblocks=None, want_state=False):
    nh = ((nhashes + MINE_K - 1) // MINE_K) * MINE_K
    if nblocks is None:
        nblocks = max(1, min(nh // MINE_K, 64))
    banks = cp.empty(nh * B * DEPTH, dtype=cp.uint32)      # 1.57 MB per nonce
    folds = cp.zeros(nh, dtype=cp.uint32)
    states = cp.zeros(nh * B, dtype=cp.uint32) if want_state else cp.zeros(0, dtype=cp.uint32)
    _mine((nblocks,), (B,),
          (np.uint64(base), folds, states, d_RULE, d_TIDX, d_TSH, d_AB,
           d_MULA, d_MULB, d_MULC, d_PREV, banks, np.int32(nh)))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(folds)[:nhashes], (cp.asnumpy(states).reshape(-1, B)[:nhashes]
                                         if want_state else None)


def run(base, nhashes, nblocks=None):
    """Mine `nhashes` consecutive nonces from `base`; return their 32-bit folds."""
    return _launch(base, nhashes, nblocks)[0]


def run_states(base, nhashes, nblocks=None):
    """Same, but also return each nonce's full final ring state (verify path)."""
    return _launch(base, nhashes, nblocks, want_state=True)


def mine_loop(start, share_target=12.0, nblocks=128):
    """Race entrant: mine forever, emit 'SHARE <nonce> <zeros> <D>' lines
    (cpu/simple.c protocol) with the same self-tuning share difficulty."""
    batch = nblocks * MINE_K * 2
    D, cnt, tsec = 10, 0, time.time()
    print(f"mine.py mining, batch={batch}", file=sys.stderr)
    n = start
    while True:
        folds = run(n, batch, nblocks=nblocks)
        for i, fv in enumerate(folds):
            z = 32 if fv == 0 else 32 - int(fv).bit_length()
            if z >= D:
                print(f"SHARE {n+i:x} {z} {D}", flush=True)
                cnt += 1
        n += batch
        t = time.time(); dt = t - tsec
        if dt >= 1.0:
            rate = cnt / dt
            while rate > share_target*1.5 and D < 31: D += 1; rate /= 2
            while rate < share_target/1.5 and D > 1:  D -= 1; rate *= 2
            cnt, tsec = 0, t


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "warmup":
        run(0, MINE_K); print("kernels compiled"); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "mine":
        mine_loop(int(sys.argv[2], 16) if len(sys.argv) > 2 else 0); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "vectors":        # vectors.txt's 4 columns, from this kernel
        with open(os.path.join(os.path.dirname(__file__), "..", "vectors.txt")) as f:
            nonces = [int(l.split()[0], 16) for l in f if not l.startswith("#") and l.strip()]
        for n in nonces:
            gf, gs = run_states(n, 1)
            print(f"{n:016x} {hashlib.sha256(spec.pack(gs[0])).hexdigest()} {int(gf[0]):08x} "
                  + "".join(f"{int(v):0{(W+3)//4}x}" for v in gs[0][::-1]))
        sys.exit(0)

    dev = cp.cuda.Device()
    name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
    print(f"GPU: {name}   B={B} W={W} DEPTH={DEPTH} STEPS={STEPS} NDSP={NDSP} K={MINE_K}")

    # ---- bit-exact check against the golden vectors ----
    # EVERY vector, and the full ring state (via its digest), not just the
    # 32-bit fold: a fold collision is astronomically unlikely but a fold is
    # only 32 of the 15,360 state bits, and two of the three vectors used to
    # go unchecked entirely because they sit at nonces the batch never reached.
    exp = {}   # nonce -> (digest_hex, fold)
    with open(os.path.join(os.path.dirname(__file__), "..", "vectors.txt")) as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split()
            if len(p) >= 3: exp[int(p[0], 16)] = (p[1], int(p[2], 16))
    bad = 0
    for n, (edig, efold) in sorted(exp.items()):
        gf, gs = run_states(n, 1)                # mine each vector's nonce directly
        gdig = hashlib.sha256(spec.pack(gs[0])).hexdigest()
        if gf[0] != efold:
            print(f"MISMATCH fold   nonce {n:#x}: got {gf[0]:08x} exp {efold:08x}"); bad += 1
        elif gdig != edig:
            print(f"MISMATCH digest nonce {n:#x}: got {gdig} exp {edig}"); bad += 1
    if bad or not exp:
        print(f"BIT-EXACT CHECK FAILED ({bad} mismatches over {len(exp)} vectors)")
        sys.exit(1)
    print(f"mine.py == spec.py: {len(exp)}/{len(exp)} golden vectors bit-exact "
          f"(fold + full-state digest)")

    # ---- benchmark ----
    for nb in (128, 256, 460):        # 460 ~= 4 blocks/SM on a 46-SM 3070;
        nh = nb * MINE_K * 2          # the old sweep stopped at 128 and under-reported
        run(0, MINE_K, nblocks=1)                # warm
        t0 = time.time(); folds = run(1000, nh, nblocks=nb); dt = time.time() - t0
        print(f"  blocks={nb:4d}  nonces={nh:5d}  {nh/dt:8.0f} H/s"
              f"   ({nh*B*DEPTH*4/1e6:.0f} MB of banks resident)")
