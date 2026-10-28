#!/usr/bin/env python3
"""Standalone CPU package checks; does not execute the CUDA candidate."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bench
import spec


def function_source_sha(source, name):
    # AST locations identify the unchanged text; AST serialization is deliberately
    # excluded because its fields differ between supported Python versions.
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1 and not nodes[0].decorator_list
    segment = ast.get_source_segment(source, nodes[0])
    assert isinstance(segment, str)
    return hashlib.sha256(segment.encode('utf-8')).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Use normal Python, without -O')
    package = bench.check_package()
    provenance = json.loads((ROOT/'PROVENANCE.json').read_text())
    for name, record in provenance['untouched_upstream_files'].items():
        assert bench.sha256(ROOT/name) == record['sha256'], 'Upstream file changed: '+name
    assert bench.sha256(ROOT/'gpu/candidate.py') == provenance['candidate']['sha256']
    source = (ROOT/'bench.py').read_bytes().decode('utf-8')
    for name, expected in provenance['benchmark']['preserved_function_source_sha256'].items():
        assert function_source_sha(source, name) == expected, 'Benchmark function changed: '+name
    for name in json.loads((ROOT/'MANIFEST.json').read_text())['files']:
        if name.endswith('.py'):
            compile((ROOT/name).read_text(), name, 'exec')
    count = 0
    for line in (ROOT/'vectors.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        nonce, digest, fold, raw = line.split()
        assert len(raw) == spec.B*((spec.W+3)//4)
        state = int(raw, 16)
        heads = [(state >> (spec.W*b)) & ((1 << spec.W)-1) for b in range(spec.B)]
        assert hashlib.sha256(spec.pack(heads)).hexdigest() == digest
        assert spec.difficulty_value(heads) == int(fold, 16)
        count += 1
    assert count == 8
    checked = subprocess.run([sys.executable, '-B', str(ROOT/'host/check_vectors.py'),
                              str(ROOT/'vectors.txt'), str(ROOT/'vectors.txt')], check=True,
                             capture_output=True, text=True)
    import numpy as np
    from types import SimpleNamespace
    from package_runtime import derived_timer
    calls=[]
    def run_states(base,count,nblocks):
        calls.append([base,count,nblocks]);return np.arange(count,dtype=np.uint32),((np.arange(count*768,dtype=np.uint32).reshape(count,768)+base)&0xfffff)
    module=SimpleNamespace(B=768,W=20,run_states=run_states)
    timer,derivation=derived_timer(bench)
    a=bench.measure(module,193,25,'full');b=timer(module,193,25,'full',460);d=timer(module,193,25,'full',23)
    assert a['checksum']==b['checksum']==d['checksum'] and calls==[[193,25,460],[193,25,460],[193,25,23]]
    assert (b['blocks'],d['blocks'])==(460,23) and derivation['inverse_source_equal'] is True
    judging = subprocess.run([sys.executable, '-B', str(ROOT/'tests/test_judging_interface.py')],
                             check=True, capture_output=True, text=True)
    judging_report = json.loads(judging.stdout)
    assert judging_report['status']=='passed_CPU_judging_interface_tests' and judging_report['GPU_executed'] is False
    assert 'cupy' not in sys.modules
    print(json.dumps({'status':'passed_CPU_package_checks','package':package,
                      'official_vector_state_digest_fold_consistency_rows':count,
                      'official_checker':checked.stdout.strip(),
                      'preserved_timing_function_source_hashes':provenance['benchmark']['preserved_function_source_sha256'],
                      'geometry_timer_actual_blocks_checked':calls,'geometry_timer_inverse':derivation,
                      'judging_interface':judging_report,'CUDA_candidate_executed':False,'GPU_validation':'pending'}, indent=2))


if __name__ == '__main__':
    main()
