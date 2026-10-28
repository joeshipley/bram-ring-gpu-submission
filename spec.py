#!/usr/bin/env python3
"""
spec.py — the BRAM-ring proof-of-work, written to be read.

This file *is* the specification. Everything else in the repo (the C and GPU
miners, the Verilog) is checked bit-for-bit against it.

Shape: B lanes in a ring, each owning one private single-cycle RAM bank that
gets read AND written at data-dependent addresses every step. Logic is cheap
on any chip; hundreds of private single-cycle RAMs are not — which is why
this is fast on an FPGA and slow on a CPU/GPU. Four deliberate choices:

  1. FULL-DEPTH BANKS.  DEPTH=512 uses every row of the FPGA's native
     512x20 block RAM (EFX_RAM10) — the banks cost the FPGA nothing extra,
     while the ~1.5 MB/nonce working set pushes a CPU attack past per-core
     L2 and any GPU cache-resident kernel toward its DRAM-bandwidth ceiling.

  2. DEEP RUNS.  STEPS=2048: one hash = DEPTH-cycle fill + STEPS-cycle run
     = 2563 cycles, so the data-dependent memory phase dominates the hash
     (the fill amortizes to ~20%) and the whole hash paces co-resident
     logic on a production miner die.

  3. A DSP MULTIPLY LAYER.  NDSP 18x19-bit multipliers (the FPGA's otherwise
     idle DSP48 blocks) each combine two epoch-chosen lanes of the ring every
     step; a few cells per lane take their remote tap from the product bits
     instead of a neighbouring lane. Global, multiply-grade diffusion at
     ~zero fabric cost — and it taxes CPUs (2 IMUL ports/core) while costing
     an FPGA nothing. It is NOT claimed to slow GPUs: their multipliers idle
     under the memory bound and absorb this for free. The moat stays a
     memory story.

  4. THE FILL *IS* THE FUNCTION.  Bank row k is filled with the ring's own
     state at step k: the first DEPTH steps run the identical CA+DSP
     dynamics with the RAM read forced to zero and the write address forced
     to the row counter. One datapath serves both phases — which deletes any
     separate fill generator in hardware — and makes the fill un-skippable:
     recomputing bank entry k lazily requires the whole ring's history up
     to step k (B*k work), not one lane's private chain.

Run me:   python3 spec.py       # prints (nonce -> digest); ~15 s per hash
"""
import hashlib
import os

# ---- parameters (sized so one core uses about 60% of a Tz170, the cheap
# ---- reference part; bigger chips replicate cores) ----
B     = 768   # lanes (= RAM banks = RAM10 blocks). B is part of the FUNCTION,
              # so the smallest miner floorplan sets it for the whole network:
              # one core uses ~60% of a Topaz Tz170 (768 of its 1280 RAM10
              # blocks, ~60% of its logic cells -- see the place reports in
              # fpga/tz170/results_*/), leaving ~40% for the rest of a
              # production miner; a Ti375 carries two cores (57%). Must be
              # a multiple of 64 (miner.v PREV rotator: B*W % 256 == 0).
W     = 20    # bits per lane, and per RAM entry (RAM10 native width)
DEPTH = 512   # entries per RAM bank (RAM10 native depth at W=20 -- full block)
ADDR  = 9     # address bits (2**ADDR == DEPTH)
REACH = 8     # a local tap comes from a lane within +-REACH
STEPS = 2048  # steps per hash; DEPTH + STEPS + overhead = 2563 cycles
NDSP  = 128   # DSP multipliers per core (power of two, easy to explain;
              # comfortably inside the Tz170's DSP count)
PTAPS = 4     # cells per lane whose remote tap reads a product bit
PBITLO, PBITHI = 8, 33  # product bits eligible as taps (the well-mixed middle)
EPOCH = os.environ.get("BRAM_EPOCH", "epoch0")  # the seed; another string is another
                                                # epoch (BRAM_EPOCH=<seed> make prove)
# The previous-block hash / job seed. It is XOR-spread into the initial state
# (init_heads) rather than hashed again, so the hot loop has no SHA: SHA-256
# touches only the final state (digest) and the block header on the host.
PREV = int.from_bytes(hashlib.sha256(b"bram-ring:prev-block-0").digest(), "little")


def run(seed, nonce):
    """Evolve the ring and return its final state: a list of B lane values
    (W bits each). This is the object every implementation must reproduce.
    Steps 0..DEPTH-1 are the FILL phase (RAM read forced to zero, write goes
    to row `step`); steps DEPTH..DEPTH+STEPS-1 are the RUN phase (normal
    read-modify-write at the data-dependent address). Same dynamics in both."""
    RULE, TAP, (MULA, MULB, MULC), ADDR_BITS = epoch_tables(seed)
    M18 = (1 << 18) - 1

    head = init_heads(seed, nonce)               # W-bit ring register per lane
    rd   = [0] * B                               # last step's RAM read, per lane
    bank = [[0] * DEPTH for _ in range(B)]       # DEPTH x W bits, per lane
    prodpipe = [[0] * NDSP, [0] * NDSP]          # REGISTERED products, TWO deep: taps
                                                 # read the products of the ring state
                                                 # two steps back (zero for steps 0,1)
                                                 # — DSP operand + output registers

    for step in range(DEPTH + STEPS):
        fill = step < DEPTH                      # FILL: write row `step`, read nothing
        prev = list(head)                        # lane taps read the ring as it was this step
        prod = prodpipe[0]                       # product taps: two steps behind
        new_head = [0] * B
        new_rd   = [0] * B
        for b in range(B):
            # 1. mix in last step's RAM read (zero through FILL and the first
            #    RUN step), then one row of cellular automaton; PTAPS cells per
            #    lane tap the product pool, the rest a nearby lane
            g = [(head[b] ^ rd[b]) >> i & 1 for i in range(W)]
            nxt = [0] * W
            for c in range(W):
                L, C, Rr = g[(c - 1) % W], g[c], g[(c + 1) % W]   # torus neighbours
                kind, tx, tb = TAP[b * W + c]
                T = (prod[tx] if kind else prev[tx]) >> tb & 1    # remote tap
                rule = RULE[b * W + c]                            # this cell's own rule
                nxt[c] = rule >> ((T << 3) | (Rr << 2) | (C << 1) | L) & 1
            v = sum(nxt[i] << i for i in range(W))

            # 2. RAM write: sequential during FILL, read-modify-write at the
            #    data-dependent address during RUN (read the OLD value first)
            if fill:
                bank[b][step] = v
            else:
                a = sum((v >> ADDR_BITS[b][k] & 1) << k for k in range(ADDR))
                new_rd[b]  = bank[b][a]
                bank[b][a] = v

            # 3. shift v to the next lane around the ring
            new_head[(b + 1) % B] = v
        # the DSP layer: NDSP products of epoch-chosen lane pairs, REGISTERED
        # TWO deep — computed from this step's ring, consumed two steps later.
        # Second operand forced odd (19 bits) + an epoch constant, so no product
        # is ever stuck at zero. One DSP48 each (operand registers + multiplier
        # + post-adder + output register); the 2-step skew is what lets the
        # globally-routed operands and the DSP close timing.
        prodpipe = [prodpipe[1],
                    [(prev[MULA[j]] & M18) * (((prev[MULB[j]] & M18) << 1) | 1) + MULC[j]
                     for j in range(NDSP)]]
        head, rd = new_head, new_rd
    return head


def digest(seed, nonce):
    """The canonical 256-bit output: SHA-256 of the final ring state."""
    return hashlib.sha256(pack(run(seed, nonce))).hexdigest()


def difficulty_value(heads):
    """32-bit XOR fold of the bit-packed ring state (B*W bits)."""
    bits = 0
    for b in range(B):
        bits |= heads[b] << (b * W)
    v = 0
    for _ in range((B * W + 31) // 32):
        v ^= bits & 0xFFFFFFFF
        bits >>= 32
    return v


# ---- helpers ----

def pack(heads):
    """Concatenate lane values little-endian into a byte string."""
    nbytes = (W + 7) // 8
    return b"".join(int(h).to_bytes(nbytes, "little") for h in heads)


def epoch_const(seed):
    """Per-epoch 64-bit constant tying the nonce-seeding to the epoch."""
    return int.from_bytes(hashlib.shake_256(seed.encode()).digest(8), "little")


def init_heads(seed, nonce):
    """Seed the ring: xorshift64 carries the nonce, XOR spread-PREV commits the
    block. Lane b takes the low W bits of the b-th xorshift output, XORed with
    a W-bit window of PREV starting at bit (b*W) mod 256."""
    MASK = (1 << 64) - 1
    s = (nonce ^ epoch_const(seed)) & MASK
    mask = (1 << W) - 1
    heads = []
    for b in range(B):
        s ^= (s << 13) & MASK; s ^= s >> 7; s ^= (s << 17) & MASK
        pv = sum(((PREV >> ((b * W + c) % 256)) & 1) << c for c in range(W))
        heads.append((s ^ pv) & mask)
    return heads


def epoch_tables(seed):
    """Derive the whole circuit from the epoch seed via SHAKE-256:
       RULE[b*W+c]      : the 16-bit truth table for cell (lane b, col c)
       TAP [b*W+c]      : (kind, x, y) -- kind 0: bit y of lane x (|x-b|<=REACH);
                          kind 1: bit y of product x (y in [PBITLO, PBITHI])
       MULA/MULB/MULC[j]: DSP j's two operand lanes and 32-bit added constant
       ADDR_BITS[b]     : which ADDR bits of a lane's value form its RAM address
    A new epoch = a new bitstream."""
    rules = approved_rules()
    palette = [rules[(i * len(rules)) // 256] for i in range(256)]
    xof = hashlib.shake_256(seed.encode()).digest(
        B * W * 6 + B * 16 + NDSP * 8 + B * ADDR * 3 + 8192)
    pos = 0
    def draw(n=1):
        nonlocal pos; v = int.from_bytes(xof[pos:pos + n], "little"); pos += n; return v

    MULA, MULB, MULC = [], [], []
    for _ in range(NDSP):
        MULA.append(draw(2) % B); MULB.append(draw(2) % B); MULC.append(draw(4))

    RULE, TAP = [], []
    for b in range(B):
        picks = []                       # the PTAPS cells of this lane that tap products
        while len(picks) < PTAPS:
            p = draw() % W
            if p not in picks: picks.append(p)
        for c in range(W):
            RULE.append(palette[draw() & 0xFF])
            if c in picks:
                TAP.append((1, draw(2) % NDSP, PBITLO + draw() % (PBITHI - PBITLO + 1)))
            else:
                off = draw() % (2 * REACH + 1) - REACH
                TAP.append((0, (b + off) % B, draw() % W))

    ADDR_BITS = []
    for b in range(B):
        picks = []
        while len(picks) < ADDR:
            p = draw() % W
            if p not in picks: picks.append(p)
        # Sorted ASCENDING, deliberately. Which ADDR of the W bits form the
        # address is epoch-secret; the ORDER they are packed in is not -- a
        # permutation of address bits is a bijection on {0..DEPTH-1}, i.e. a
        # pure relabelling of bank slots, with zero cryptographic content.
        # Unsorted, a CPU must gather the bits one at a time (measured: 28% of
        # verify time). Sorted, the whole address is one PEXT instruction. The
        # FPGA is unaffected (the address is a wire permutation either way) and
        # so is the GPU (it is bandwidth-bound, not ALU-bound). Pure verifier
        # tax removed, moat untouched.
        ADDR_BITS.append(sorted(picks))
    return RULE, TAP, (MULA, MULB, MULC), ADDR_BITS


def approved_rules():
    """The 4-input truth tables allowed as cell rules (balanced,
    nonlinearity 4, all-input-dependent, degree 3, SAC; exactly 1056)."""
    def nonlinearity(tt):
        best = 16
        for coeffs in range(32):
            c = [(coeffs >> i) & 1 for i in range(5)]
            dist = sum(((tt >> x) & 1) !=
                       (c[0] ^ sum(c[1 + i] & ((x >> i) & 1) for i in range(4)) & 1)
                       for x in range(16))
            best = min(best, dist)
        return best
    def degree(tt):
        a = [(tt >> i) & 1 for i in range(16)]
        for i in range(4):
            for j in range(16):
                if j & (1 << i): a[j] ^= a[j ^ (1 << i)]
        return max((bin(i).count("1") for i in range(16) if a[i]), default=0)
    def depends_on_all(tt):
        return all(any(((tt >> x) & 1) != ((tt >> (x ^ (1 << v))) & 1) for x in range(16))
                   for v in range(4))
    def sac(tt):
        return all(sum(((tt >> x) & 1) != ((tt >> (x ^ (1 << v))) & 1) for x in range(16)) == 8
                   for v in range(4))
    return [tt for tt in range(1 << 16)
            if bin(tt).count("1") == 8 and nonlinearity(tt) == 4
            and depends_on_all(tt) and degree(tt) == 3 and sac(tt)]


if __name__ == "__main__":
    print(f"BRAM-ring: B={B} lanes, W={W} bits, DEPTH={DEPTH}, {STEPS} steps, "
          f"{NDSP} DSP mults, epoch '{EPOCH}'")
    for nonce in (0, 1, 0xdeadbeef):
        print(f"  nonce {nonce:#010x} -> {digest(EPOCH, nonce)}")
