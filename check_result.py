#!/usr/bin/env python3
"""Bounded CPU validation of one completed standalone full-work benchmark."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bench
import spec
from capture_actual import elf_code_identity


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)


def correctness_roles(records):
    assert len(records)==2 and {r['variant'] for r in records}=={'baseline','candidate'}, 'Both distinct implementation vector records are required'
    return {r['variant']:r for r in records}


def metadata_path(directory, relative):
    relative=Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts, 'Unsafe metadata path'
    directory=directory.resolve()
    metadata=directory/'metadata'
    assert metadata.is_dir() and not metadata.is_symlink(), 'Metadata folder must be a real directory within the result'
    path=(directory/relative).resolve()
    assert path.is_relative_to(metadata) and path.is_file(), 'Metadata path escapes the readback folder'
    return path


def packing_path(directory, relative):
    relative=Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts, 'Unsafe packing path'
    directory=directory.resolve();folder=directory/'packing'
    assert folder.is_dir() and not folder.is_symlink(), 'Packing folder must be a real directory within the result'
    path=(directory/relative).resolve()
    assert path.parent==folder and path.is_file(), 'Packing path escapes the evidence folder'
    return path


def completion_row(rows,capture):
    completed=[r for r in rows if r.get('type')=='compiled_provenance']
    assert len(completed)==1 and rows[-1]==completed[0], 'Final compiler-provenance completion row is missing or duplicated'
    record=completed[0]
    assert record['status']==capture['status']=='ok' and record['timing_completed'] is capture['timing_completed'] is True, 'Runner and capture completion differ'
    assert Path(record['report']).parts[-2:]==('compiled','benchmark-capture.json')


def cuda_source(path, epoch):
    """Reconstruct only the two literal CUDA definitions, without CuPy imports."""
    tree=ast.parse(path.read_text())
    values={name:getattr(spec,name) for name in ('B','W','DEPTH','ADDR','STEPS','NDSP','PBITLO')}
    values.update(ECONST=spec.epoch_const(epoch),MINE_K=1)
    def value(name):
        nodes=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        assert len(nodes)==1
        return nodes[0]
    defs=value('DEFS')
    assert all(isinstance(n,(ast.JoinedStr,ast.FormattedValue,ast.Constant,ast.Name,ast.Load)) for n in ast.walk(defs))
    return eval(compile(ast.Expression(defs),'<DEFS>','eval'),{'__builtins__':{}},values)+ast.literal_eval(value('BODY'))


def check(directory, pairs, nonces, base, vectors, candidate_blocks=460, candidate_packer="numpy"):
    if not __debug__:
        raise RuntimeError('Use normal Python, without -O')
    package = bench.check_package()
    assert candidate_blocks in (23,460) and candidate_packer in ('numpy','native')
    assert pairs in (5,20) and nonces==1840
    blocks={'stock':460,'candidate':candidate_blocks};packers={'stock':'numpy','candidate':candidate_packer}
    env, summary = read(directory/'environment.json'), read(directory/'summary.json')
    rows = [json.loads(x) for x in (directory/'measurements.jsonl').read_text().splitlines() if x.strip()]
    assert not any(r.get('type') == 'failure' for r in rows), 'Benchmark failed'
    assert [r for r in rows if r.get('type') == 'environment'] == [env]
    assert [r for r in rows if r.get('type') == 'summary'] == [summary]
    assert env['package'] == package and env['harness_source_sha256'] == bench.sha256(ROOT/'bench.py')
    assert env['device']['name'] in ('NVIDIA GeForce RTX 3070', 'GeForce RTX 3070')
    assert (env['device']['major'], env['device']['minor'], env['device']['multiProcessorCount']) == (8, 6, 46)
    assert (env['pairs'],env['nonces'],env['blocks'],env['nonce_base'],env['mode']) == (pairs,nonces,blocks,base,'full')
    assert env['MINE_K'] == {'stock':1,'candidate':1}
    assert summary['timing_includes_digest'] is True and summary['all_paired_checksums_equal'] is True
    assert (summary['pairs'],summary['nonces'],summary['blocks'],summary['epoch'],summary['mode']) == (pairs,nonces,blocks,env['epoch'],'full')
    assert env['packers']==summary['packers']==packers and env['packing_binding']==summary['packing_binding']
    from package_runtime import derived_timer
    for role in ('stock','candidate'):
        _,expected_timer=derived_timer(bench,object() if packers[role]=='native' else None)
        assert env['timer_derivations'][role]==expected_timer
    if candidate_packer=='native':
        binding=env['packing_binding'];proof_path=packing_path(directory,'packing/cpu-proof.json');build_path=packing_path(directory,'packing/build.json')
        proof=read(proof_path);build=read(build_path)
        assert binding['kind']=='native_cpu_pack' and binding['mining_source_changed'] is False and binding['every_nonce_SHA_retained'] is True
        assert bench.sha256(build_path)==binding['build_record_sha256']==proof['build_record_sha256']
        assert bench.sha256(proof_path)==binding['cpu_proof_sha256']
        library=packing_path(directory,binding['library'])
        assert bench.sha256(library)==binding['library_sha256']==proof['library_sha256']==build['library_sha256']
        assert library.name==build['library'] and build['status']=='built_not_benchmarked'
        assert proof['status']=='passed_cpu_only_not_gpu_validated' and proof['sanitizer']['status']=='passed' and proof['sanitizer']['exit_code']==0
        assert proof['sources']==binding['sources'] and set(proof['sources'])=={'pack.c','adapter.py','build.py','sanitize.c','cpu_check.py'}
        for name,h in proof['sources'].items():assert bench.sha256(ROOT/'packing'/name)==h
        assert build['source_sha256']==proof['sources']['pack.c'] and build['builder_sha256']==proof['sources']['build.py']
        assert proof['python']==env['python'] and proof['numpy']==env['numpy']
        assert proof['machine']==build['machine']==binding['machine'] and build['pointer_bits']==binding['pointer_bits']
        assert len(proof['golden_vectors'])==3
        for v in proof['golden_vectors']:
            path=(ROOT/v['path']).resolve();assert path.is_relative_to(ROOT) and bench.sha256(path)==v['sha256'] and v['vectors']==8 and v['every_digest_equal'] is True
    else:assert env['packing_binding'] is None
    source_sha = {'stock':bench.sha256(ROOT/'gpu/mine.py'),'candidate':bench.sha256(ROOT/'gpu/candidate.py')}
    assert summary['baseline_source_sha256'] == source_sha['stock'] and summary['candidate_source_sha256'] == source_sha['candidate']
    measured = [r for r in rows if r.get('type') == 'measurement']
    assert len(measured) == 2*pairs
    rates = {'stock':[],'candidate':[]}; seconds = {'stock':[],'candidate':[]}; wins = 0
    for i in range(pairs):
        pair = measured[2*i:2*i+2]
        assert [r['which'] for r in pair] == (['stock','candidate'] if i%2 == 0 else ['candidate','stock'])
        assert pair[0]['checksum'] == pair[1]['checksum']
        for row in pair:
            assert (row['trial'],row['nonces'],row['blocks'],row['base'],row['mode']) == (i,nonces,blocks[row['which']],base+i*nonces,'full')
            assert row['packer']==packers[row['which']]
            assert math.isfinite(row['seconds']) and row['seconds'] > 0
            assert close(row['hashes_per_second'], nonces/row['seconds'])
            assert len(row['checksum']) == 64 and int(row['checksum'],16) >= 0
            rates[row['which']].append(row['hashes_per_second']);seconds[row['which']].append(row['seconds'])
        wins += rates['candidate'][-1] > rates['stock'][-1]
    assert rates == summary['rates']
    means = {r:statistics.mean(v) for r,v in rates.items()}
    assert all(close(means[r],summary['mean_rates'][r]) for r in rates)
    gain = 100*(means['candidate']/means['stock']-1)
    assert close(gain,summary['speedup_percent'])
    assert [r for r in rows if r.get('type') == 'scratch_reuse'] == [
        {'type':'scratch_reuse','differential_nonces':51,'launch_cases':[[1234567,8,2],[0x8123456789000000,17,3],[193,25,candidate_blocks],[0,1,candidate_blocks+1]],'all_equal':True}]
    correct = [r for r in rows if r.get('type') == 'correctness']
    for record in correctness_roles(correct).values():
        path = directory/(record['variant']+'-vectors.txt')
        assert record['epoch'] == env['epoch'] and record['vectors_sha256'] == bench.sha256(vectors)
        assert record['output_sha256'] == bench.sha256(path)
        subprocess.run([sys.executable,'-B',str(ROOT/'host/check_vectors.py'),str(path),str(vectors)],
                       check=True,capture_output=True,text=True)
    capture = read(directory/'compiled/benchmark-capture.json')
    completion_row(rows,capture)
    assert capture['status'] == 'ok' and capture['timing_completed'] is True
    assert capture['capture_hook_restored_before_timing'] is True and capture['compile_options_changed'] is False
    assert capture['cache_repaired'] is False
    for role,binding in capture['modules'].items():
        assert role in source_sha and binding['candidate_source_sha256'] == source_sha[role]
        assert binding['epoch'] == env['epoch'] and binding['MINE_K'] == 1
        assert binding['cupy_kernel_attributes'] == env['kernel_attributes'][role]
        assert len(binding['matching_captures']) == 1
        expected=cuda_source(ROOT/('gpu/mine.py' if role=='stock' else 'gpu/candidate.py'),env['epoch'])
        assert hashlib.sha256(expected.encode()).hexdigest()==binding['defs_body_sha256']
        index=binding['matching_captures'][0]['index']
        assert len([r for r in capture['captures'] if r['index']==index])==1
        assert expected in (directory/'compiled'/f'capture-{index:02d}'/'actual-nvrtc-source.cu').read_text()
    assert set(capture['modules']) == {'stock','candidate'}
    for record in capture['captures']:
        assert record['status']=='ok' and record['compile_return_object_unchanged'] is True
        folder = directory/'compiled'/f'capture-{record["index"]:02d}'
        full,returned = (folder/'nvrtc-complete.cubin').read_bytes(),(folder/'cupy-returned.cubin').read_bytes()
        assert hashlib.sha256(full).hexdigest() == record['nvrtc_complete']['sha256']
        assert hashlib.sha256(returned).hexdigest() == record['cupy_returned']['sha256']
        assert bench.sha256(folder/'actual-nvrtc-source.cu') == record['source_sha256']
        identity = elf_code_identity(full,returned)
        assert identity['all_identical'] and identity['target_section_present']
        assert identity == record['relationship']['executable_sections']
    metadata = read(directory/'metadata-readbacks.json')
    assert metadata['epoch'] == env['epoch'] and metadata['outside_timing'] is True
    assert set(metadata['roles']) == {'stock','candidate'}
    for role,record in metadata['roles'].items():
        assert record['source_sha256'] == source_sha[role]
        assert set(record['tables'])=={'d_PREV','d_RULE','d_TIDX','d_TSH','d_AB','d_MULA','d_MULB','d_MULC'}
        for name,table in record['tables'].items():
            raw = metadata_path(directory,table['raw_file']).read_bytes()
            assert len(raw) == table['nbytes'] and hashlib.sha256(raw).hexdigest() == table['sha256']
    assert metadata['roles']['candidate']['tables']['d_RULE']['device_ptr']%8 == 0
    assert metadata['roles']['candidate']['tables']['d_RULE']['nbytes']==30720
    attrs = env['kernel_attributes']['candidate']
    assert attrs['local_size_bytes'] == 0 and attrs['max_threads_per_block'] >= 768 and attrs['binary_version'] == 86
    assert 'cupy' not in sys.modules
    return {'status':'passed_standalone_package_result_check','epoch':env['epoch'],
            'blocks':blocks,'packers':packers,'pairs':pairs,'nonces_per_sample':nonces,'measurements_retained':len(measured),'no_samples_dropped':True,
            'mean_rate_gain_percent':gain,'total_elapsed_gain_percent':100*(sum(seconds['stock'])/sum(seconds['candidate'])-1),
            'paired_wins':wins,'package':package,'candidate_source_sha256':source_sha['candidate'],
            'baseline_source_sha256':source_sha['stock'],'summary_sha256':bench.sha256(directory/'summary.json'),
            'confidence_interval':None,'organizer_validation':False,
            'scope':'Bounded result identity, vectors, metadata, raw pairs and arithmetic; no full-bank or sanitizer replay.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--pairs',type=int,choices=(5,20),default=20);p.add_argument('--nonces',type=int,default=1840)
    p.add_argument('--nonce-base',type=lambda x:int(x,0),default=5600000);p.add_argument('--vectors',type=Path,default=ROOT/'vectors.txt')
    p.add_argument('--candidate-blocks',type=int,choices=(23,460),default=460);p.add_argument('--candidate-packer',choices=('numpy','native'),default='numpy')
    a=p.parse_args();print(json.dumps(check(a.directory.resolve(),a.pairs,a.nonces,a.nonce_base,a.vectors.resolve(),a.candidate_blocks,a.candidate_packer),indent=2))
