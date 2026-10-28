# BRAM-ring RTX 3070 GPU speed submission

A reproducible candidate for Peter Rizun's **0.01 BTC BRAM-ring GPU speed award**. The selected implementation combines the RULE64 kernel, an explicit **23-block launch**, and a portable native CPU serializer. The untouched upstream comparator uses **460 blocks and NumPy packing**.

The measured standalone package achieved **+13.627261%** in a same-session comparison on a rented RTX 3070, winning **20 of 20 matched pairs**. Every sample processed 1,840 nonces, all 512 FILL rounds, all 2,048 RUN steps, and every nonce's SHA-256 digest. This is local evidence; organizer verification and the applicable champion determine the award.

| Historical standalone epoch0 result | Mean hashes/s |
| --- | ---: |
| Upstream stock, 460 blocks, NumPy | 1,476.523792 |
| RULE64, 23 blocks, native packing | 1,677.733547 |

All 40 observations are retained. The ratio of mean rates gives +13.627261%; the total-elapsed estimator gives +13.627236%. The main experiment's separate three-epoch confirmations measured +13.654793%, +13.656695%, and +13.585876%. Historical gains have not been added together.

**Evidence scope:** the GPU results were produced by the immutable 20-pair package in [`evidence/package-v2/`](evidence/package-v2/). This public submission broadens the command-line interface to accept five or twenty pairs. Its CUDA sources, timed functions, timer derivation, and native serializer are byte-identical to that measured package. The new five-pair interface has passed CPU control-flow and rejection tests; it has not been represented as a new GPU run or an organizer result. Historical results must be checked with their original package, not relabeled as executions of this commit.

The upstream project is [nexapow/bram-ring](https://git.nodelabs.ca/nexapow/bram-ring), based here on commit [`abe3f552fef39b1d7fc52327b2ca5e97134b57b3`](https://git.nodelabs.ca/nexapow/bram-ring/-/commit/abe3f552fef39b1d7fc52327b2ca5e97134b57b3). The [bounty rules](https://git.nodelabs.ca/nexapow/bram-ring/-/blob/main/BOUNTY.md) and [original announcement](https://x.com/PeterRizun/status/2107245548311060888) govern judging. Publication of this repository is not itself the required announcement-thread claim.

## Reproduce on an RTX 3070

Use Linux, Python 3.12, a compatible NVIDIA driver, CUDA/NVRTC 12.x, and a C compiler with AddressSanitizer and UndefinedBehaviorSanitizer. Recorded development versions were Python 3.12.3, driver 580.159.03, CUDA runtime 12.9, NVRTC 12.8, GCC 13.3, CuPy 14.2.0, and NumPy 2.5.4. The host CPU was an i7-8700K; the organizer's reference host is an i7-12700KF. These rental measurements make no hardware ownership or performance-per-dollar claim.

From a clean clone:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B bench.py --check-package
python -B cpu_check.py
python -B packing/build.py --output build/native-pack
python -B packing/cpu_check.py --build build/native-pack --output build/native-pack/cpu-proof.json
```

The native library must be built and proved on the judging machine. Archived shared libraries are evidence only. The CPU proof checks all 20-bit values, arbitrary uint32 inputs, misaligned addresses, guards, exact-size sanitizer allocations, and independent golden vectors. No root access, provider credentials, driver modifications, or clock changes are needed by this repository.

Run the organizer's initial five samples per implementation:

```sh
BRAM_EPOCH=epoch0 MINE_K=1 python -B bench.py \
  --pairs 5 --nonces 1840 --nonce-base 5600000 \
  --candidate-blocks 23 --candidate-packer native \
  --pack-build build/native-pack --pack-proof build/native-pack/cpu-proof.json \
  --output results/epoch0-five

python -B check_result.py results/epoch0-five \
  --pairs 5 --nonces 1840 --nonce-base 5600000 \
  --candidate-blocks 23 --candidate-packer native
```

For twenty samples per implementation, use a new output directory and `--pairs 20` in both commands. The runner records five or twenty matched pairs, alternating stock/candidate order. With five pairs there are three pairs in one order and two in the other. No sample is discarded. Both runs use the arithmetic mean of rates; the checker also reports the ratio based on total elapsed time.

**Pass the candidate flags explicitly.** Compatibility defaults remain 460 blocks and NumPy. The original `gpu/candidate.py` command-line rate display is a fold-only diagnostic and is not the full-SHA submission benchmark. Use `bench.py` above. Run on an otherwise idle GPU without concurrent benchmarks or transfers. Output directories must be new.

Only epoch-table generation, kernel compilation, native-library setup, and correctness checks occur before timing. Each timed sample includes allocation/clearing, nonce initialization, FILL, RUN, synchronization, state transfer, canonical serialization, and SHA-256 for every nonce. An aggregate audit checksum is calculated after timing. Each paired full-output checksum must match.

Before scoring, both implementations pass eight official four-column vectors, 51 additional full-state scratch-reuse/inactive-block comparisons, and a full 1,840-nonce comparison of canonical and selected serialization. Actual compiler captures, metadata readbacks, resource attributes, native build identity, environment, and raw observations are saved. Do not use `python -O` or `PYTHONOPTIMIZE`.

## Check another epoch

`spec.py`, `generate.py`, and committed `vectors.txt` are preserved byte-for-byte. Create reference data in a scratch directory so the frozen definition files remain unchanged:

```sh
mkdir -p results/fresh-reference/cpu results/fresh-reference/fpga
cp spec.py generate.py results/fresh-reference/
(cd results/fresh-reference && BRAM_EPOCH='organizer-seed' python generate.py)

BRAM_EPOCH='organizer-seed' MINE_K=1 python -B bench.py \
  --pairs 5 --verify-only --candidate-blocks 23 --candidate-packer native \
  --pack-build build/native-pack --pack-proof build/native-pack/cpu-proof.json \
  --vectors results/fresh-reference/vectors.txt --output results/fresh-verify
```

Replace `organizer-seed` with the organizer's exact seed. A non-epoch0 run requires an independent reference vector file. The verification log includes both stock and candidate outputs checked by `host/check_vectors.py`, including nonce, digest, fold, and full ring state. `--verify-only` produces no timing claim. Speed judging uses epoch0; the fresh-seed response is due within 48 hours of the organizer's seed reply under the current rules.

## Inspect the existing evidence without a GPU

With NumPy installed:

```sh
python -B evidence/check_evidence.py
python -B evidence/package-v2/check_result.py \
  evidence/standalone-epoch0-full-v2/benchmark \
  --candidate-blocks 23 --candidate-packer native
```

[`evidence/`](evidence/) retains the exact historical standalone package, raw twenty-pair result, held-out correctness run, native captures, and additional direct comparisons. Its `FILES.json` binds the selected artifacts; `ORIGIN.json` records their origin. Raw measurements and original receipts are unchanged. `MANIFEST.json` binds the current runtime source and documentation; the Git commit freezes the whole submission. `PROVENANCE.json` explains the source bridge and [`validation/`](validation/) records checks of the publication interface.

## What changed from upstream

The candidate retains the complete earlier optimization chain:

- Reuse bank scratch by active block, with 32-lane bank tiles and an invertible row XOR layout.
- Replace repeated address-bit gathering with 480 KiB of seed-derived lookup tables, using cache-priority hints for immutable table reads.
- Schedule product and ring taps separately and load RULE metadata four uint16 entries at a time, with corresponding packed shift loads.
- Publish shared state before the bank exchange while preserving the specified dependency ordering.
- Use the measured 23-block launch and a byte-exact native CPU serializer for the full-digest path.

Precomputation depends only on the supplied epoch seed and is excluded from timing as permitted by the rules. It does not hard-code epoch0. Every nonce still performs the full specified computation.

The 23-block choice is implementation-dependent: stock became 0.84–1.17% slower at 23 blocks in separate controls. RULE64 at 23 blocks with the original NumPy packer still beat stock at 460 blocks by 13.04–13.20%. The native packer alone contributed about 0.42–0.45%. These controls are not summed to produce the headline result. No DRAM-bound fraction, physical SM placement, or organizer qualification is inferred from block-count scaling.

## Licence and attribution

MIT. The original Peter Rizun copyright and permission notice are preserved in [`LICENSE`](LICENSE); see [`NOTICE.txt`](NOTICE.txt). The unchanged definition files and upstream comparator remain identifiable by their published hashes. This repository makes no claim to the separate 0.1 BTC hardware-efficiency award and asserts no organizer endorsement or prize decision.
