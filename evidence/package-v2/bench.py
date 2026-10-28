#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare a candidate against a named frozen baseline, including full outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_package():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        if path.resolve().parent != ROOT and ROOT not in path.resolve().parents:
            raise ValueError('Manifest path escapes package')
        if sha256(path) != expected:
            raise ValueError('Package hash mismatch: ' + relative)
    return {'files_verified': len(manifest['files']),
            'manifest_sha256': sha256(ROOT / 'MANIFEST.json')}


def load_kernel(label, filename):
    path = Path(filename).resolve()
    spec = importlib.util.spec_from_file_location('handoff_' + label, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PILOT_VARIANT = label
    module.PILOT_SOURCE_SHA256 = sha256(path)
    return module


def packed_states(module, states):
    import numpy as np
    assert module.W == 20 and states.dtype == np.uint32
    little = states.astype('<u4', copy=False)
    return np.ascontiguousarray(
        little.view(np.uint8).reshape(len(states), module.B, 4)[:, :, :3])


def measure(module, base, count, mode):
    start = time.perf_counter()
    if mode == 'fold':
        folds = module.run(base, count, nblocks=460)
        digests = []
    else:
        folds, states = module.run_states(base, count, nblocks=460)
        if mode == 'canonical':
            digests = [hashlib.sha256(module.spec.pack(s)).digest() for s in states]
        else:
            packed = packed_states(module, states)
            digests = [hashlib.sha256(s).digest() for s in packed]
    seconds = time.perf_counter() - start
    # The per-nonce SHA-256s above are timed. This aggregate audit checksum is not.
    checksum = hashlib.sha256(folds.tobytes() + b''.join(digests)).hexdigest()
    return {'seconds': seconds, 'hashes_per_second': count / seconds,
            'checksum': checksum, 'base': base, 'nonces': count, 'blocks': 460}


def verify(module, vectors, output):
    nonces = []
    for line in vectors.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError('Expected four columns in reference vectors')
            nonces.append(int(fields[0], 16))
    if len(nonces) != 8 or len(set(nonces)) != 8:
        raise ValueError('Expected eight distinct reference vector nonces')
    rows = []
    for nonce in nonces:
        folds, states = module.run_states(nonce, 1, nblocks=1)
        canonical = module.spec.pack(states[0])
        assert packed_states(module, states)[0].tobytes() == canonical
        rows.append(f'{nonce:016x} {hashlib.sha256(canonical).hexdigest()} '
                    f'{int(folds[0]):08x} ' + ''.join(f'{int(v):05x}' for v in states[0][::-1]))
    path = output / (module.PILOT_VARIANT + '-vectors.txt')
    path.write_text('\n'.join(rows) + '\n')
    checked = subprocess.run([sys.executable, str(ROOT / 'host/check_vectors.py'),
                              str(path), str(vectors)], check=True,
                             capture_output=True, text=True)
    return {'type': 'correctness', 'variant': module.PILOT_VARIANT,
            'epoch': module.EPOCH, 'result': checked.stdout.strip(),
            'vectors_sha256': sha256(vectors), 'output_sha256': sha256(path)}


def command_result(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'argv': argv, 'error': str(error)}


def main():
    if not __debug__:
        raise RuntimeError('Run without Python -O or PYTHONOPTIMIZE; correctness assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT/'gpu/candidate.py')
    parser.add_argument('--baseline', type=Path, default=ROOT/'gpu/mine.py')
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--nonces', type=int, default=1840)
    parser.add_argument('--nonce-base', type=lambda s: int(s, 0), default=5600000)
    parser.add_argument('--mode', choices=('full',), default='full',
                        help='full includes state transfer/packing/SHA-256; fold is separately labeled')
    parser.add_argument('--candidate-blocks',type=int,choices=(23,460),default=460)
    parser.add_argument('--candidate-packer',choices=('numpy','native'),default='numpy')
    parser.add_argument('--pack-build',type=Path)
    parser.add_argument('--pack-proof',type=Path)
    parser.add_argument('--vectors', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--check-package', action='store_true', help='hash check only; no CUDA imports')
    args = parser.parse_args()
    package = check_package()
    if args.check_package:
        print(json.dumps({'status': 'package_hashes_passed', **package}, indent=2))
        return
    if args.candidate.resolve() != ROOT/'gpu/candidate.py' or args.baseline.resolve() != ROOT/'gpu/mine.py':
        parser.error('Only the two manifest-bound packaged sources may be compared')
    if args.pairs != 20 or args.nonces != 1840:
        parser.error('This package uses exactly20 full pairs of1840 nonces')
    if args.candidate_packer=='numpy' and (args.pack_build is not None or args.pack_proof is not None):
        parser.error('Native CPU build/proof arguments require native packing')
    if args.nonce_base < 0 or args.nonce_base + args.pairs * args.nonces > 2**64:
        parser.error('Timed nonce range must fit unsigned 64-bit values')
    if os.environ.get('MINE_K', '1') != '1':
        parser.error('This frozen validation package requires MINE_K=1')
    os.environ['MINE_K'] = '1'
    epoch = os.environ.get('BRAM_EPOCH', 'epoch0')
    if epoch != 'epoch0' and args.vectors is None:
        parser.error('A non-epoch0 seed requires --vectors from an independent CPU reference')
    vectors = (args.vectors or ROOT / 'vectors.txt').resolve()
    if not vectors.is_file():
        parser.error('Reference vector file does not exist')
    default_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    output = (args.output or ROOT / 'results' / default_name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from package_runtime import derived_timer,native_packer
    packing_object=packing_binding=None
    if args.candidate_packer=='native':
        packing_object,packing_binding=native_packer(args.pack_build,args.pack_proof,output)
    packers={'stock':'numpy','candidate':args.candidate_packer}
    blocks={'stock':460,'candidate':args.candidate_blocks}
    timers={};timer_derivations={}
    for role in ('stock','candidate'):
        timers[role],timer_derivations[role]=derived_timer(sys.modules[__name__],packing_object if role=='candidate' else None)
    capture = None
    completed = False
    with (output / 'measurements.jsonl').open('w') as log:
        def emit(value):
            line = json.dumps(value, sort_keys=True, default=str)
            log.write(line + '\n')
            log.flush()
            print(line, flush=True)

        try:
            from capture_actual import begin_benchmark_capture
            capture = begin_benchmark_capture(output / 'compiled')
            stock = load_kernel('baseline', args.baseline)
            candidate = load_kernel(args.candidate.stem, args.candidate)
            capture.register_modules({'stock': stock, 'candidate': candidate})
            if hasattr(candidate, 'TAP_SCHEDULE'):
                (output / 'tap-schedule.json').write_text(json.dumps(candidate.TAP_SCHEDULE, indent=2) + '\n')
            if hasattr(candidate, 'SHARED_LAYOUT'):
                (output / 'shared-layout.json').write_text(json.dumps({
                    'diagnostics': candidate.SHARED_LAYOUT,
                    'logical_to_physical': candidate._shared_permutation,
                    'outside_timing': True}, indent=2) + '\n')
            if hasattr(stock, 'SHARED_LAYOUT'):
                (output / 'baseline-shared-layout.json').write_text(json.dumps({
                    'diagnostics': stock.SHARED_LAYOUT,
                    'logical_to_physical': stock._shared_permutation,
                    'outside_timing': True}, indent=2) + '\n')
            import cupy as cp
            import numpy as np
            for prefix, module in (('', candidate), ('baseline-', stock)):
                if hasattr(module, 'INIT_JUMP_LAYOUT'):
                    previous = cp.asnumpy(module.d_PREV)
                    table = np.asarray(module._INIT_JUMP, dtype=np.uint32).reshape(-1)
                    assert np.array_equal(previous[:module.B], module.PREVCHUNK)
                    assert np.array_equal(previous[module.B:], table)
                    (output / (prefix + 'init-jump.json')).write_text(json.dumps({
                        'diagnostics': module.INIT_JUMP_LAYOUT,
                        'table_sha256': hashlib.sha256(table.astype('<u4').tobytes()).hexdigest(),
                        'device_previous_words': int(previous.size),
                        'prefix_equal': True, 'suffix_equal': True,
                        'outside_timing': True}, indent=2) + '\n')
                if hasattr(module, 'ADDRESS_LUT_LAYOUT'):
                    address_bytes = cp.asnumpy(module.d_AB).view(np.uint8).reshape(-1)
                    address_prefix = np.asarray(module.ADDRBIT, dtype=np.uint8).reshape(-1)
                    address_table = np.ascontiguousarray(module._ADDRESS_LUT).reshape(-1)
                    table_bytes = address_table.view(np.uint8)
                    if hasattr(module, 'ADDRESS_PAYLOAD_HOST'):
                        expected_payload = np.asarray(module.ADDRESS_PAYLOAD_HOST, dtype=np.uint8).reshape(-1)
                        assert np.array_equal(address_bytes, expected_payload), 'Custom address payload mismatch'
                        table_bytes = expected_payload[address_prefix.size:]
                    assert np.array_equal(address_bytes[:address_prefix.size], address_prefix)
                    assert np.array_equal(address_bytes[address_prefix.size:], table_bytes)
                    (output / (prefix + 'address-lut.json')).write_text(json.dumps({
                        'diagnostics': module.ADDRESS_LUT_LAYOUT,
                        'device_total_bytes': int(address_bytes.size),
                        'prefix_sha256': hashlib.sha256(address_prefix.tobytes()).hexdigest(),
                        'table_sha256': hashlib.sha256(address_table.tobytes()).hexdigest(),
                        'suffix_sha256': hashlib.sha256(table_bytes.tobytes()).hexdigest(),
                        'payload_sha256': hashlib.sha256(address_bytes.tobytes()).hexdigest(),
                        'table_dtype': str(address_table.dtype),
                        'prefix_equal': True, 'suffix_equal': True,
                        'outside_timing': True}, indent=2) + '\n')
            assert stock.EPOCH == candidate.EPOCH == epoch
            assert stock.MINE_K in (1,2) and candidate.MINE_K in (1,2)
            if not args.verify_only:
                assert args.nonces % stock.MINE_K == args.nonces % candidate.MINE_K == 0, 'Timed work must not pad nonce count'
            properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            keys = ('name', 'major', 'minor', 'multiProcessorCount', 'totalGlobalMem',
                    'clockRate', 'memoryClockRate', 'memoryBusWidth', 'l2CacheSize',
                    'sharedMemPerBlock', 'sharedMemPerBlockOptin', 'regsPerBlock',
                    'regsPerMultiprocessor','maxThreadsPerMultiProcessor','reservedSharedMemPerBlock','sharedMemPerMultiprocessor')
            device = {key: properties.get(key) for key in keys}
            device = {key: value.decode() if isinstance(value, bytes) else value
                      for key, value in device.items()}
            cpuinfo = Path('/proc/cpuinfo')
            cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                              if line.startswith('model name')), '') if cpuinfo.exists() else platform.processor()
            environment = {'type': 'environment', 'utc': datetime.now(timezone.utc).isoformat(),
                           'platform': platform.platform(), 'python': platform.python_version(),
                           'cpu': cpu_model, 'logical_cpus': os.cpu_count(),
                           'numpy': np.__version__, 'cupy': cp.__version__,
                           'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
                           'cuda_driver': cp.cuda.runtime.driverGetVersion(),
                           'device': device, 'epoch': epoch, 'MINE_K': {'stock':stock.MINE_K,'candidate':candidate.MINE_K},
                           'mode': args.mode, 'blocks': blocks, 'packers':packers, 'packing_binding':packing_binding, 'timer_derivations':timer_derivations, 'baseline_path': str(args.baseline), 'candidate_path': str(args.candidate), 'baseline_label': str(args.baseline), 'nonces': args.nonces,
                           'pairs': args.pairs, 'nonce_base': args.nonce_base,
                           'command': sys.argv, 'package': package,
                           'harness_source_sha256': sha256(Path(__file__)),
                           'kernel_attributes': {'stock': dict(stock._mine.attributes),
                                                 'candidate': dict(candidate._mine.attributes)},
                           'nvidia_smi': command_result(['nvidia-smi', '-q'])}
            (output / 'environment.json').write_text(json.dumps(environment, indent=2, default=str) + '\n')
            emit(environment)
            from metadata_capture import dump_metadata
            emit(dump_metadata(stock, 'stock', output))
            emit(dump_metadata(candidate, 'candidate', output))
            assert candidate._mine.attributes['local_size_bytes']==0, 'Candidate spills; timing gate rejected'
            assert candidate._mine.attributes['max_threads_per_block']>=768, 'Candidate cannot launch 768 threads'
            assert candidate._mine.attributes['num_regs']<=80, 'Candidate exceeds sm86 register budget'
            if True:
                alignment = {'RULE':int(candidate.d_RULE.data.ptr)%8, 'TIDX':int(candidate.d_TIDX.data.ptr)%8, 'TSH':int(candidate.d_TSH.data.ptr)%4}
                assert all(v == 0 for v in alignment.values()), 'Wide metadata base misaligned'
                emit({'type':'metadata_alignment','remainders':alignment,'all_aligned':True})
            emit(verify(stock, vectors, output))
            emit(verify(candidate, vectors, output))
            reuse_cases=((1234567,8,2),(0x8123456789000000,17,3),(193,25,args.candidate_blocks),(0,1,args.candidate_blocks+1))
            for base, count, launch_blocks in reuse_cases:
                af, ast = stock.run_states(base, count, nblocks=launch_blocks)
                bf, bst = candidate.run_states(base, count, nblocks=launch_blocks)
                assert np.array_equal(af, bf) and np.array_equal(ast, bst), 'Scratch-reuse mismatch'
            emit({'type':'scratch_reuse','differential_nonces':51,'launch_cases':[list(x) for x in reuse_cases],'all_equal':True})
            if not args.verify_only:
                # Canonical packing equality is established for BOTH implementations
                # outside timing samples, using the complete intended work size.
                for role,module in (('stock',stock),('candidate',candidate)):
                    canonical = timers[role](module,9000,args.nonces,'canonical',blocks[role])
                    fast = timers[role](module,9000,args.nonces,'full',blocks[role])
                    assert canonical['checksum'] == fast['checksum'], 'Serialization mismatch'
                for role,module in (('stock',stock),('candidate',candidate)):
                    timers[role](module,1000,args.nonces,args.mode,blocks[role])
                rates = {'stock': [], 'candidate': []}
                for trial in range(args.pairs):
                    order = [('stock', stock), ('candidate', candidate)]
                    if trial % 2:
                        order.reverse()
                    checksums = []
                    for label, module in order:
                        if packing_object is not None:packing_object.assert_unchanged()
                        row = timers[label](module,args.nonce_base+trial*args.nonces,args.nonces,args.mode,blocks[label])
                        rates[label].append(row['hashes_per_second'])
                        checksums.append(row['checksum'])
                        emit({'type': 'measurement', 'trial': trial, 'which': label,
                              'mode': args.mode, 'packer':packers[label], **row})
                    assert checksums[0] == checksums[1], 'Timed pair output mismatch'
                means = {key: statistics.mean(value) for key, value in rates.items()}
                summary = {'type': 'summary', 'mode': args.mode, 'epoch': epoch,
                           'pairs': args.pairs, 'nonces': args.nonces, 'blocks': blocks, 'packers':packers, 'packing_binding':packing_binding,
                           'mean_rates': means, 'rates': rates,
                           'baseline_source_sha256': stock.PILOT_SOURCE_SHA256,
                           'candidate_source_sha256': candidate.PILOT_SOURCE_SHA256,
                           'baseline_label': stock.PILOT_VARIANT,
                           'candidate_label': candidate.PILOT_VARIANT,
                           'speedup_percent': 100 * (means['candidate'] / means['stock'] - 1),
                           'timing_includes_digest': args.mode == 'full',
                           'all_paired_checksums_equal': True}
                (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                emit(summary)
                emit({'type': 'post_run_device_state',
                      'nvidia_smi': command_result(['nvidia-smi', '-q'])})
                completed = True
        except BaseException as error:
            emit({'type': 'failure', 'error': f'{type(error).__name__}: {error}'})
            raise
        finally:
            check_package()
            if packing_object is not None:packing_object.assert_unchanged()
            if capture is not None:
                report = capture.finish(timing_completed=completed, disassemble=False)
                emit({'type': 'compiled_provenance', 'status': report['status'],
                      'timing_completed': completed,
                      'report': str(output / 'compiled/benchmark-capture.json')})
                if report['status'] != 'ok' and sys.exc_info()[0] is None:
                    raise RuntimeError('Compiled-binary capture incomplete; inspect provenance report')
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
