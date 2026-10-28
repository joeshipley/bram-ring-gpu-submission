Standalone-v2 execution closure

Package-v2 and its archive are unchanged. execute_standalone.py is an external
orchestration/checking wrapper; run it from /root/bram35 with that rental's
.venv/bin/python. It uses the previously extracted external package at
/root/bram35-standalone-v2/bram-round35 and the already CPU-proved Linux build.
It neither builds a new library nor changes a production default.

Run the two one-job controller lists separately and in this order:
  submission-prep/round35/jobs-standalone-holdout-verify.json
  submission-prep/round35/jobs-standalone-epoch0-full.json

The first produces round35/results/standalone-holdout-v2. It runs verify-only
with the fixed Round35 held-out seed and independently generated vectors, then
CPU-checks both eight-row/four-column outputs, 51 differential full-state
nonces (including 23-block reuse/inactive coverage), actual NVRTC captures,
full/returned executable identity, all metadata readbacks and pack-library
identity. It emits no measurement or summary and makes no timing claim. The
package's verify-only vector serialization remains canonical NumPy; prior
packing correctness evidence and the pinned native CPU proof remain distinct.

The second requires the first receipt and its entire file inventory to pass.
It produces round35/results/standalone-epoch0-full-v2, running exactly 20
alternating full-work pairs of 1,840 nonces starting at 5,600,000. Stock uses
460 blocks/NumPy and the unchanged RULE64 candidate uses 23 blocks/native CPU
packing. The packaged check_result.py then verifies all 40 observations,
full-output checks, actual geometry, packing, captures and arithmetic.

Both stages hold /tmp/bram-round30-gpu.lock and retain command argv, bounded
timeouts, combined child stdout/stderr logs, step exits and hashes. Process
groups are terminated on timeout/interruption. A passing completion.json
binds all closed result files plus before/after package, archive, manifest,
all prepared payload/build files, wrapper and plan identities. A failure gets
failure.json instead and must not be accepted as a completed stage.

The parent may fetch other CLOSED evidence during the unscored holdout stage.
The parent MUST finish all other GPU jobs and transfers before starting the
scored full stage, and must not fetch/upload while that stage runs. The flock
prevents cooperating GPU jobs from overlapping; it cannot prevent independent
network/CPU activity. The wrapper itself performs no network access.

Upload execution wrapper, plan and two job lists; also preserve the CPU control
report and this note in the final archive inventory. PREPARATION-v2.json and
the archive already exist on the rental. All result artifacts are under
round35/results, so the final independent full archive includes them.

CPU-only local self-test:
  python3 -B submission-prep/round35/test_execution_wrapper.py
It checks identity failures and prerequisite-receipt controls with temporary
fixtures. It never imports CuPy or invokes the execution entrypoint.
