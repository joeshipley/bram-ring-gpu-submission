# Historical measured evidence

These are unmodified results from the frozen **package-v2** and the preceding
direct comparison campaign. They test unchanged RULE64 at **23 blocks with
native CPU packing**, against untouched stock at **460 blocks with NumPy
packing**. Every scored observation includes 1,840 nonces and all final SHA-256
digests. These records do not describe a GPU run of the new five-run interface.

| Measurement | Mean-rate gain | Paired wins |
|---|---:|---:|
| Standalone package-v2, epoch0 | +13.627261% | 20/20 |
| Earlier direct comparison, epoch0 | +13.654793% | 20/20 |
| Earlier direct comparison, fresh epoch | +13.656695% | 20/20 |
| Earlier direct comparison, holdout epoch | +13.585876% | 20/20 |

The standalone mean rates are 1677.733547 H/s for the candidate and
1476.523792 H/s for stock. Its independently recomputed descriptive 95% paired
bootstrap interval is [+13.589128%, +13.667490%]. The total-elapsed gain is
+13.627236%. Every observation is retained; historical gains are not added.

The host was an RTX 3070 with an Intel i7-8700K, Linux kernel 6.8.0-138,
Python 3.12.3, NumPy 2.5.4, CuPy 14.2.0, NVIDIA driver 580.159.03, CUDA runtime
12.9, NVRTC 12.8 and GCC 13.3. The organizer's reference host differs, and these
results are not organizer verification or an award claim.

## CPU-only replay

From the repository root, using Python 3.12 or later:

```sh
python3 -B evidence/check_evidence.py
```

The helper checks the exact public file inventory and historical source hashes,
closed wrapper/controller receipts and all saved raw samples. It independently
recomputes the four paired comparisons, both estimators, order/half splits and
descriptive intervals. It also replays the original held-out CPU evidence checker
with only its package path relocated in memory, and invokes the original package
result checker. It does not execute a GPU kernel or load the saved native library.
It prints its report and does not change files.

To replay only the historical standalone result checker:

```sh
python3 -B evidence/package-v2/check_result.py \
  evidence/standalone-epoch0-full-v2/benchmark \
  --candidate-blocks 23 --candidate-packer native
```

Use the checker under `evidence/package-v2/` for these historical results.
The current submission harness has different source hashes because it adds the
five/twenty-run judging interface. Historical records must not be relabelled as
results from that new harness. The frozen original source snapshot and its
original tarball are both included.

## Contents and provenance

- `standalone-epoch0-full-v2/`: all 40 raw observations, full-column vectors,
  original environment/summary, actual CUDA source and complete/returned CUBINs,
  metadata readbacks, native packing build/proof, child logs and wrapper receipts.
- `standalone-holdout-v2/`: unscored independent-seed correctness, both sets of
  eight full-column vectors, the reference vectors, recorded 51-nonce full-state
  differential coverage, native captures and its closed wrapper evidence.
- `geometry-{epoch0,fresh,holdout}-stock_native/` and matching `*-analysis.json`:
  the three earlier direct twenty-pair comparisons, complete and unmodified.
- `controller-*`, `execution/` and `standalone-cpu-v2/`: exact closed job records,
  their plans, the historical execution wrapper and actual-host CPU preparation.
- `package-v2/`: all 27 original package payload files plus their manifest.
- `historical-package-v2.tar.gz`: the exact original archive, SHA-256
  `f4337743330c71806b65b2b83dd59bec51952ee2c6f8431436a7970c59db0a57`.
- `ORIGIN.json`: each copied file's original logical path and SHA-256, the source
  full-archive identity, and the distinction between historical and new interfaces.
- `FILES.json`: hashes for every public evidence file except itself.

Every copied result was checked against the verified final source archive:
SHA-256 `b89deeb06d6188d77afcc50fd9bbc3f896911ac12b0b86e956c0154f93ab0ba4`,
193,181,018 bytes and 10,134 files. This focused bundle omits that large archive,
unrelated campaign runs and provider/account control material. Campaign admission
records retain references to omitted precursor validation files; this bundle does
not claim to include the entire campaign dependency tree.

Recorded `/root/...` paths, GPU identifiers and machine/version details remain
unchanged to preserve hashes. They identify the historical execution environment;
the public CPU replay uses relative evidence paths. The saved Linux shared library
is a hashed historical artifact, not a portable binary to install or execute.

Both eight-vector outputs can be independently checked here. The 51 additional
full-state comparisons are established by the pinned successful runner and its
recorded assertion coverage; their individual states were not saved for CPU
recomputation. The unscored holdout supplies no timing claim. The new five-run
interface has CPU tests only; no five-run GPU result is represented here.
