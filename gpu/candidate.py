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
assert B % 32 == 0, 'bank tile must divide lane count'
assert DEPTH > 0 and (DEPTH & (DEPTH - 1)) == 0, 'row XOR requires power-of-two depth'
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
"""

_mod = cp.RawModule(code=DEFS + BODY, options=("-std=c++14",))
_mine = _mod.get_function("ring_mine")


def _tap_slot_cost(words):
    # Same-address broadcast counts once. This proxy describes one warp's
    # scalar shared-word instruction, with the standard 32-bank word mapping.
    banks = [0] * 32
    for word in set(words):
        banks[word & 31] += 1
    return max(banks) * 1000 + sum(n * n for n in banks)


def _tap_schedule(rules, taps, lanes, width, pbitlo, mode):
    assert width == 20 and lanes % 32 == 0 and lanes < 1024
    assert mode in ("tap_kind_split", "tap_conflict_scheduled", "tap_grouped_reuse")
    orders = []
    for b in range(lanes):
        products = [c for c in range(width) if taps[b * width + c][0]]
        rings = [c for c in range(width) if not taps[b * width + c][0]]
        assert len(products) == 4 and len(rings) == 16
        if mode == "tap_grouped_reuse":
            products.sort(key=lambda c: (taps[b * width + c][1], c))
            rings.sort(key=lambda c: (taps[b * width + c][1], c))
        orders.append(products + rings)

    def score_all():
        return sum(_tap_slot_cost([taps[(w + i) * width + orders[w + i][s]][1]
                                   for i in range(32)])
                   for w in range(0, lanes, 32) for s in range(width))

    before = score_all()
    swaps = 0
    if mode == "tap_conflict_scheduled":
        for w in range(0, lanes, 32):
            words = [[taps[(w + i) * width + orders[w + i][s]][1]
                      for i in range(32)] for s in range(width)]
            costs = [_tap_slot_cost(col) for col in words]
            # Bounded deterministic coordinate descent. Each accepted swap
            # strictly improves the integer score; no dependence on nonce data.
            for sweep in range(2):
                changed = False
                for i in range(32):
                    for lo, hi in ((0, 4), (4, 20)):
                        best = None
                        best_gain = 0
                        for a in range(lo, hi):
                            for b in range(a + 1, hi):
                                if words[a][i] == words[b][i]:
                                    continue
                                words[a][i], words[b][i] = words[b][i], words[a][i]
                                ca, cb = _tap_slot_cost(words[a]), _tap_slot_cost(words[b])
                                words[a][i], words[b][i] = words[b][i], words[a][i]
                                gain = costs[a] + costs[b] - ca - cb
                                if gain > best_gain:
                                    best, best_gain = (a, b, ca, cb), gain
                        if best is not None:
                            a, b, ca, cb = best
                            words[a][i], words[b][i] = words[b][i], words[a][i]
                            orders[w + i][a], orders[w + i][b] = orders[w + i][b], orders[w + i][a]
                            costs[a], costs[b] = ca, cb
                            swaps += 1
                            changed = True
                if not changed:
                    break
    after = score_all()
    if mode == "tap_conflict_scheduled":
        assert after <= before
    srule, sindex, sshift = [], [], []
    per_lane_loads = []
    for b, order in enumerate(orders):
        assert sorted(order) == list(range(width))
        loads = 0
        prev = None
        for s, c in enumerate(order):
            kind, tx, ty = taps[b * width + c]
            assert bool(kind) == (s < 4)
            assert 0 <= tx < 1024 and 0 <= c < 32
            srule.append(rules[b * width + c])
            sindex.append(tx | (c << 10))
            shift = ty - pbitlo if kind else ty
            assert 0 <= shift < 32
            sshift.append(shift)
            key = (kind, tx)
            if mode != "tap_grouped_reuse" or key != prev:
                loads += 1
            prev = key
        per_lane_loads.append(loads)
    return srule, sindex, sshift, {
        "orders": orders,
        "proxy_cost_before": before,
        "proxy_cost_after": after,
        "accepted_swaps": swaps,
        "per_lane_word_loads": per_lane_loads,
        "proxy_is_not_gpu_timing": True,
    }


def _address_chunk_table(address_bits, lanes, width, address_width):
    # Epoch-only contributions; input chunks partition all20bits exactly.
    assert lanes == 768 and width == 20 and address_width == 9
    assert len(address_bits) == lanes
    for bits in address_bits:
        assert len(bits) == address_width and list(bits) == sorted(set(bits))
        assert all(0 <= bit < width for bit in bits)
    table = []
    for start, count in ((0,7),(7,7),(14,6)):
        for value in range(1 << count):
            for bits in address_bits:
                contribution = 0
                for output_bit, input_bit in enumerate(bits):
                    if start <= input_bit < start+count:
                        contribution |= ((value >> (input_bit-start)) & 1) << output_bit
                table.append(contribution)
    return table

_ADDRESS_LUT = np.asarray(_address_chunk_table(ADDR_BITS, B, W, ADDR), dtype='<u2')
ADDRESS_LUT_LAYOUT = {
    'kind': 'address_variable_chunk_value_lane_u16',
    'chunk_widths': [7,7,6], 'chunk_bit_offsets': [0,7,14],
    'chunk_row_offsets': [0,128,256], 'rows': 320, 'lanes': B,
    'prefix_bytes': int(ADDRBIT.nbytes),
    'prefix_sha256': hashlib.sha256(ADDRBIT.tobytes()).hexdigest(),
    'table_words': int(_ADDRESS_LUT.size),
    'table_sha256': hashlib.sha256(_ADDRESS_LUT.tobytes()).hexdigest(),
    'epoch': EPOCH, 'nonce_independent': True, 'outside_timing': True,
}

_ts_ru, _ts_idx, _ts_sh, TAP_SCHEDULE = _tap_schedule(RULE, TAP, B, W, PBITLO, 'tap_conflict_scheduled')
d_RULE = cp.asarray(np.asarray(_ts_ru, dtype=np.uint16))
d_TIDX = cp.asarray(np.asarray(_ts_idx, dtype=np.uint16))
d_TSH = cp.asarray(np.asarray(_ts_sh, dtype=np.uint8))
d_AB   = cp.asarray(np.concatenate((ADDRBIT, _ADDRESS_LUT.view(np.uint8)))); d_MULA = cp.asarray(MULAf);  d_MULB = cp.asarray(MULBf)
d_MULC = cp.asarray(MULCf);   d_PREV = cp.asarray(PREVCHUNK)


def _launch(base, nhashes, nblocks=None, want_state=False):
    nh = ((nhashes + MINE_K - 1) // MINE_K) * MINE_K
    if nblocks is None:
        nblocks = max(1, min(nh // MINE_K, 64))
    banks = cp.empty(min(nh, nblocks * MINE_K) * B * DEPTH, dtype=cp.uint32)  # one private bank set per active block/slot
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
