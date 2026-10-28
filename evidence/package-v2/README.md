# BRAM-ring: explicit geometry and optional CPU packing

This is conditional submission preparation. It contains the unchanged RULE64
mining source, the untouched original stock implementation, and an optional
portable CPU serializer. No result or prize qualification is claimed here.
Standalone GPU execution, final source selection, and organizer review remain
separate. The default comparison still uses 460 blocks and NumPy packing;
23 blocks and native packing must be selected explicitly.

Use Linux, Python 3.12, an idle RTX 3070, a compatible NVIDIA driver and CUDA/NVRTC
12.8, plus a C compiler with AddressSanitizer and UndefinedBehaviorSanitizer.
The CPU library is built on that machine. No precompiled library is shipped.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B bench.py --check-package
python -B cpu_check.py
```

For the optional native CPU packer:

```sh
python -B packing/build.py --output build/native-pack
python -B packing/cpu_check.py --build build/native-pack --output build/native-pack/cpu-proof.json
```

The CPU proof checks all 20-bit values, arbitrary uint32 values, misaligned
addresses, output guards, exact-size sanitizer allocations, and 24 bundled
independent golden vectors. Its library/source/compiler identity is recorded.
Rebuild and recheck when changing machines or files. Source files and the
manifest must remain unchanged; put builds and results in new subdirectories.

## Explicit full-work comparisons

After the corresponding experiment is selected and reviewed, run the proposed
combination against untouched stock:

```sh
BRAM_EPOCH=epoch0 MINE_K=1 python -B bench.py --candidate-blocks 23 --candidate-packer native --pack-build build/native-pack --pack-proof build/native-pack/cpu-proof.json --output results/epoch0-23-native
python -B check_result.py results/epoch0-23-native --candidate-blocks 23 --candidate-packer native
```

The optional 23-block NumPy control uses the same mining source:

```sh
BRAM_EPOCH=epoch0 MINE_K=1 python -B bench.py --candidate-blocks 23 --candidate-packer numpy --output results/epoch0-23-numpy
python -B check_result.py results/epoch0-23-numpy --candidate-blocks 23 --candidate-packer numpy
```

Stock always uses 460 blocks and NumPy packing. Every command records its actual
block counts and packer. These are separate stock comparisons; subtracting their
percentages is not a directly paired native-versus-NumPy comparison. The campaign
runs that direct control separately. `gpu/candidate.py` retains its inherited
launch defaults; the explicit benchmark options choose this experiment's
geometry without modifying the kernel or silently overriding its launcher.

Each run retains all 20 alternating matched pairs, each with 1,840 identical
nonces, 768 threads per block, and K=1. The timer includes allocation/clearing,
nonce initialization, all 512 FILL and 2,048 RUN steps, synchronization, full
state transfer, canonical packing and SHA-256 for every nonce. Only compilation,
immutable epoch metadata, native-library setup and verification occur before
scoring. The aggregate checksum is outside timing, as in the frozen benchmark.
The native packer changes serialization only; it performs no hashing or CUDA.

The timer is derived from the frozen Round25 function by exactly four source
edits: an explicit block-count parameter, two launch arguments, and the returned
actual block count. The inverse is checked and the derived function is recorded.
Native packing substitutes only the serializer in that function's globals.
Neither the mining source nor its launch function is monkey-patched.

Before scoring, both implementations pass eight official four-column vectors,
51 additional full-state differential nonces covering reused slots and inactive
blocks, and full 1,840-nonce canonical-versus-selected packing checks. Every
scored checksum must match. Native source/CUBIN captures, metadata readbacks,
actual resources, and (when used) the CPU library/build/proof are saved alongside
raw samples. The CPU result checker checks their identities and both gain
formulas. It supplies no confidence interval and cannot replace campaign
full-bank checks, GPU sanitizers or organizer validation.

## Independent epochs

The upstream specification, generator, vectors, checker and license are preserved
byte-for-byte. Generate a separate CPU reference without editing packaged files:

```sh
mkdir -p results/fresh-reference/cpu results/fresh-reference/fpga
cp spec.py generate.py results/fresh-reference/
(cd results/fresh-reference && BRAM_EPOCH=your-independent-seed python generate.py)
BRAM_EPOCH=your-independent-seed MINE_K=1 python -B bench.py --verify-only --candidate-blocks 23 --vectors results/fresh-reference/vectors.txt --output results/fresh-verify
```

For a fresh full benchmark, remove `--verify-only` and select the intended packer
with the same arguments as above. Supply the same reference to
`check_result.py --vectors ...`. A non-epoch0 run without explicit reference
vectors is rejected. The stock/candidate inherited command-line rates are
fold-only diagnostics and must not be used as this full-SHA result.

Preserve the complete result directory and package manifest. Report actual
geometry, packing choice, both gain metrics, and all samples. This package does
not contact a provider, change GPU settings, publish a claim, or submit anything.
