Publication checks

package-hashes.json: current MANIFEST.json and its 33 runtime/documentation entries passed.
cpu-checks.json: canonical vector consistency, preserved timed-function source, explicit-block timer checks, and the new five/twenty-pair CPU interface controls passed.
historical-evidence-check.json: the original package-v2 checker and held-out checker reproduced the unchanged archived GPU evidence. All 40 standalone observations and the three additional 40-observation comparisons were retained.

These checks ran without a GPU. Synthetic fixtures in the interface tests check command admission, ordering, count, labels and checksum rejection; they are not performance measurements. The current CLI revision has not been claimed as a new GPU execution. Raw historical evidence and its original checker remain under evidence/.

Commands from repository root:
  python -B bench.py --check-package
  python -B cpu_check.py
  python -B evidence/check_evidence.py
