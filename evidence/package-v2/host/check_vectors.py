#!/usr/bin/env python3
"""
check_vectors.py -- does YOUR implementation reproduce vectors.txt, every column?

Have your program print one line per golden vector in vectors.txt's own format
(hex; the raw state is lane B-1..0, (W+3)//4 hex digits per lane):

    <nonce> <sha256 digest> <32-bit fold> <raw ring state>

then pipe it through this script (exit 0 = bit-exact on every column of every
vector; anything else exits 1 and names the first difference):

    your_program | python3 host/check_vectors.py
    python3 host/check_vectors.py yours.txt            # or from a file
    python3 host/check_vectors.py yours.txt other.txt  # against another vectors file
                                                       # (e.g. one made with BRAM_EPOCH=...)

Columns are compared as integers, so case and leading zeros do not matter, but
every golden vector must be present and all four columns must match. This is
the bar BOUNTY.md rule 1 sets; `make prove` holds the shipped implementations
to the same one.
"""
import sys, os

def load(path_or_lines):
    rows = {}
    for ln in path_or_lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        f = ln.split()
        if len(f) != 4:
            sys.exit(f"FAIL: expected 4 columns (nonce digest fold state), got {len(f)}: {ln[:60]}...")
        try:
            rows[int(f[0], 16)] = tuple(int(x, 16) for x in f[1:])
        except ValueError:
            sys.exit(f"FAIL: non-hex field in: {ln[:60]}...")
    return rows

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    golden_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "vectors.txt")
    yours = load(open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin)
    golden = load(open(golden_path))
    if not golden: sys.exit(f"FAIL: no vectors in {golden_path}")
    names = ("digest", "fold", "state")
    for nonce, exp in sorted(golden.items()):
        if nonce not in yours:
            sys.exit(f"FAIL: vector nonce {nonce:#x} missing from your output ({len(yours)} lines seen)")
        got = yours[nonce]
        for n, g, e in zip(names, got, exp):
            if g != e:
                sys.exit(f"FAIL: nonce {nonce:#x} {n} differs\n  got {g:x}\n  exp {e:x}")
    print(f"PASS: {len(golden)}/{len(golden)} vectors bit-exact on every column (digest, fold, state)")

if __name__ == "__main__":
    main()
