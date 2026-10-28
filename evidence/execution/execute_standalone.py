#!/usr/bin/env python3
"""Run one closed standalone-v2 stage; never rent, transfer, or modify the package."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/root/bram35')
PACKAGE = Path('/root/bram35-standalone-v2/bram-round35')
HERE = Path(__file__).resolve().parent
PLAN = HERE/'EXECUTION-PLAN.json'
PYTHON = ROOT/'.venv/bin/python'
BLOCKS = {'stock': 460, 'candidate': 23}
PACKERS = {'stock': 'numpy', 'candidate': 'native'}
CHILD = None
sys.dont_write_bytecode = True

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

def safe_file(base, relative):
    p = Path(relative)
    need(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative file path')
    q = base/p
    need(q.is_file() and not q.is_symlink() and q.resolve() == q.absolute(), 'File missing or symlinked: '+str(q))
    return q

def checked_files(base, files):
    need(bool(files), 'Empty file inventory')
    actual = {name: sha(safe_file(base, name)) for name in files}
    need(actual == files, 'Pinned files changed')
    return actual

def identities(plan):
    need(plan['package'] == str(PACKAGE), 'Wrong package path')
    need(sha(HERE/'PREPARATION-v2.json') == plan['preparation_sha256'], 'Preparation differs')
    need(sha(ROOT/plan['archive']) == plan['archive_sha256'], 'Archive differs')
    need(sha(PACKAGE/'MANIFEST.json') == plan['manifest_sha256'], 'Package manifest differs')
    manifest = read(PACKAGE/'MANIFEST.json')
    payloads = checked_files(PACKAGE, manifest['files'])
    need(len(payloads) == 27, 'Unexpected payload count')
    need(payloads['gpu/candidate.py'] == plan['candidate_sha256'] and payloads['gpu/mine.py'] == plan['stock_sha256'], 'Source identity differs')
    prepared_path = ROOT/plan['cpu_prepared_path']
    need(sha(prepared_path) == plan['cpu_prepared_sha256'], 'Actual-host CPU preparation differs')
    prepared = read(prepared_path)
    need(prepared['status'] == 'passed_CPU_only_no_GPU' and prepared['package'] == str(PACKAGE), 'CPU setup did not pass')
    need(prepared['archive_sha256'] == plan['archive_sha256'] and prepared['manifest_sha256'] == plan['manifest_sha256'], 'CPU setup package differs')
    need(len(prepared['steps']) == 4 and all(type(x['exit_code']) is int and x['exit_code'] == 0 for x in prepared['steps']), 'CPU setup steps incomplete')
    prepared_files = checked_files(PACKAGE, prepared['files'])
    checked_files(PACKAGE, plan['build_files'])
    need(all(prepared_files.get(k) == h for k, h in plan['build_files'].items()), 'Build not in CPU preparation')
    build = read(PACKAGE/'build/native-pack/build.json')
    proof = read(PACKAGE/'build/native-pack/cpu-proof.json')
    need(build['status'] == 'built_not_benchmarked' and proof['status'] == 'passed_cpu_only_not_gpu_validated', 'Build/proof status differs')
    need(proof['sanitizer']['status'] == 'passed' and proof['sanitizer']['exit_code'] == 0, 'CPU sanitizer failed')
    need(proof['library_sha256'] == build['library_sha256'] == plan['build_files']['build/native-pack/libbram_pack.so'], 'Native library differs')
    need(proof['build_record_sha256'] == plan['build_files']['build/native-pack/build.json'], 'Proof build binding differs')
    need(sha(ROOT/plan['holdout_vectors']) == plan['holdout_vectors_sha256'], 'Held-out vectors differ')
    need((ROOT/plan['holdout_seed_file']).read_text().strip() == plan['holdout_seed'], 'Held-out seed differs')
    return {'archive_sha256': plan['archive_sha256'], 'preparation_sha256': plan['preparation_sha256'],
            'manifest_sha256': plan['manifest_sha256'], 'prepared_sha256': plan['cpu_prepared_sha256'],
            'package_payloads': payloads, 'all_prepared_files': prepared_files,
            'wrapper_sha256': sha(__file__), 'execution_plan_sha256': sha(PLAN)}

def terminate_child():
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            CHILD.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait(timeout=5)

def interrupted(signum, frame):
    terminate_child()
    raise RuntimeError('Interrupted by signal '+str(signum))

def step(out, rows, name, argv, timeout, env):
    global CHILD
    log = out/(name+'.log')
    start = time.time()
    row = {'name': name, 'argv': [str(x) for x in argv], 'cwd': str(PACKAGE), 'timeout_seconds': timeout,
           'started_unix': start, 'BRAM_EPOCH': env['BRAM_EPOCH'], 'log': log.name}
    try:
        with log.open('wb') as stream:
            CHILD = subprocess.Popen(row['argv'], cwd=PACKAGE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            row['pid'] = CHILD.pid
            try:
                row['exit_code'] = CHILD.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                row['timed_out'] = True
                terminate_child()
                row['exit_code'] = CHILD.returncode
                raise RuntimeError(name+' timed out')
    finally:
        terminate_child()
        CHILD = None
        row['elapsed_seconds'] = time.time()-start
        if log.exists(): row['log_sha256'] = sha(log)
        rows.append(row)
        save(out/'steps.json', rows)
    need(row['exit_code'] == 0, name+' failed: '+str(row['exit_code']))
    return log

def load_checker():
    sys.path.insert(0, str(PACKAGE))
    spec = importlib.util.spec_from_file_location('standalone_result_checker', PACKAGE/'check_result.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def prior_holdout(plan):
    folder = ROOT/plan['stages']['holdout-verify']['output']
    need(not (folder/'failure.json').exists(), 'Held-out failure marker present')
    completion = read(folder/'completion.json')
    need(completion['status'] == 'completed' and completion['stage'] == 'holdout-verify' and completion['exit_code'] == 0, 'Held-out stage is not complete')
    need(completion['scored_stage'] is False and completion['timing_claim'] is False and completion['all_step_exits_zero'] is True, 'Held-out receipt scope differs')
    need(completion['execution_plan_sha256'] == sha(PLAN) and completion['wrapper_sha256'] == sha(__file__), 'Held-out wrapper/plan differs')
    need(completion['archive_sha256'] == plan['archive_sha256'] and completion['manifest_sha256'] == plan['manifest_sha256'], 'Held-out package differs')
    checked_files(folder, completion['files'])
    need(completion['result_check_sha256'] == sha(folder/'result-check.json'), 'Held-out result check differs')
    checked = read(folder/'result-check.json')
    need(checked['status'] == 'passed_standalone_holdout_verify_only_check' and checked['epoch'] == plan['holdout_seed'] and checked['blocks'] == BLOCKS and checked['packers'] == PACKERS, 'Held-out check failed or differs')
    need(checked['timing_claim'] is False and checked['timed_measurements'] == 0 and checked['differential_full_state_nonces'] == 51, 'Held-out validation incomplete')
    return {'path':str(folder/'completion.json'), 'sha256':sha(folder/'completion.json'), 'result_check_sha256':sha(folder/'result-check.json')}

def verify_holdout(directory, vectors, plan, run_vector_checker):
    """CPU-only audit of verify-only output. No summary or timing result is synthesized."""
    checker = load_checker()
    rows = [json.loads(x) for x in (directory/'measurements.jsonl').read_text().splitlines() if x.strip()]
    env = read(directory/'environment.json')
    need([r for r in rows if r.get('type') == 'environment'] == [env], 'Environment row differs')
    need(not any(r.get('type') in ('failure', 'measurement', 'summary') for r in rows) and not (directory/'summary.json').exists(), 'Verify-only contains timing/failure records')
    need(env['epoch'] == plan['holdout_seed'] and env['blocks'] == BLOCKS and env['packers'] == PACKERS, 'Verify-only scope differs')
    need(env['pairs'] == 20 and env['nonces'] == 1840 and env['nonce_base'] == 5600000 and env['mode'] == 'full', 'Requested full scope differs')
    need(env['MINE_K'] == {'stock': 1, 'candidate': 1}, 'Kernel K differs')
    need(env['device']['name'] in ('NVIDIA GeForce RTX 3070', 'GeForce RTX 3070') and (env['device']['major'], env['device']['minor'], env['device']['multiProcessorCount']) == (8, 6, 46), 'Device differs')
    need(env['package'] == {'files_verified': 27, 'manifest_sha256': plan['manifest_sha256']} and env['harness_source_sha256'] == sha(PACKAGE/'bench.py'), 'Harness/package differs')
    need([r for r in rows if r.get('type') == 'scratch_reuse'] == [{'type':'scratch_reuse', 'differential_nonces':51, 'launch_cases':[[1234567,8,2],[0x8123456789000000,17,3],[193,25,23],[0,1,24]], 'all_equal':True}], '51 full-state scratch comparisons missing')
    correctness = checker.correctness_roles([r for r in rows if r.get('type') == 'correctness'])
    for role, row in correctness.items():
        path = directory/(role+'-vectors.txt')
        need(row['epoch'] == env['epoch'] and row['vectors_sha256'] == sha(vectors) == plan['holdout_vectors_sha256'] and row['output_sha256'] == sha(path), 'Vector record binding differs')
        need(len(path.read_text().splitlines()) == 8 and all(len(x.split()) == 4 for x in path.read_text().splitlines()), 'Eight full-column vectors missing')
        run_vector_checker(role, path, vectors)
    capture = read(directory/'compiled/benchmark-capture.json')
    completion = [r for r in rows if r.get('type') == 'compiled_provenance']
    need(len(completion) == 1 and completion[0] == rows[-1] and completion[0]['status'] == capture['status'] == 'ok', 'Actual capture did not close')
    need(completion[0]['timing_completed'] is False and capture['timing_completed'] is False, 'Verify-only mislabeled timed')
    need(capture['capture_hook_restored_before_timing'] is True and capture['compile_options_changed'] is False and capture['cache_repaired'] is False, 'Compilation intervention differs')
    source_sha = {'stock': plan['stock_sha256'], 'candidate': plan['candidate_sha256']}
    need(set(capture['modules']) == set(source_sha), 'Actual capture roles differ')
    cubins = {}
    for role, binding in capture['modules'].items():
        need(binding['candidate_source_sha256'] == source_sha[role] and binding['epoch'] == env['epoch'] and binding['MINE_K'] == 1, 'Capture source/epoch differs')
        need(binding['cupy_kernel_attributes'] == env['kernel_attributes'][role] and len(binding['matching_captures']) == 1, 'Capture/attributes binding differs')
        expected = checker.cuda_source(PACKAGE/('gpu/mine.py' if role == 'stock' else 'gpu/candidate.py'), env['epoch'])
        need(hashlib.sha256(expected.encode()).hexdigest() == binding['defs_body_sha256'], 'Actual CUDA reconstruction differs')
        index = binding['matching_captures'][0]['index']
        need(len([r for r in capture['captures'] if r['index'] == index]) == 1, 'Nonunique native capture')
        folder = directory/'compiled'/f'capture-{index:02d}'
        need(expected in (folder/'actual-nvrtc-source.cu').read_text(), 'Actual NVRTC source differs')
        cubins[role] = sha(folder/'nvrtc-complete.cubin')
    for row in capture['captures']:
        folder = directory/'compiled'/f'capture-{row["index"]:02d}'
        need(row['status'] == 'ok' and row['compile_return_object_unchanged'] is True, 'NVRTC capture failed')
        full = (folder/'nvrtc-complete.cubin').read_bytes(); returned = (folder/'cupy-returned.cubin').read_bytes()
        need(hashlib.sha256(full).hexdigest() == row['nvrtc_complete']['sha256'] and hashlib.sha256(returned).hexdigest() == row['cupy_returned']['sha256'] and sha(folder/'actual-nvrtc-source.cu') == row['source_sha256'], 'Capture bytes differ')
        identity = checker.elf_code_identity(full, returned)
        need(identity['all_identical'] and identity['target_section_present'] and identity == row['relationship']['executable_sections'], 'Full/returned executable bridge differs')
    metadata = read(directory/'metadata-readbacks.json')
    need(metadata['epoch'] == env['epoch'] and metadata['outside_timing'] is True and set(metadata['roles']) == set(source_sha), 'Metadata readback scope differs')
    for role, row in metadata['roles'].items():
        need(row['source_sha256'] == source_sha[role] and set(row['tables']) == {'d_PREV','d_RULE','d_TIDX','d_TSH','d_AB','d_MULA','d_MULB','d_MULC'}, 'Metadata source/table coverage differs')
        for table in row['tables'].values():
            path = checker.metadata_path(directory, table['raw_file'])
            need(path.stat().st_size == table['nbytes'] and sha(path) == table['sha256'], 'Metadata bytes differ')
    need(metadata['roles']['candidate']['tables']['d_RULE']['device_ptr'] % 8 == 0 and metadata['roles']['candidate']['tables']['d_RULE']['nbytes'] == 30720, 'RULE64 alignment/size differs')
    attrs = env['kernel_attributes']['candidate']
    need(attrs['local_size_bytes'] == 0 and attrs['max_threads_per_block'] >= 768 and attrs['binary_version'] == 86, 'Candidate native resources differ')
    packing = env['packing_binding']
    for relative, field in [('build.json','build_record_sha256'), ('cpu-proof.json','cpu_proof_sha256'), ('libbram_pack.so','library_sha256')]:
        need(sha(checker.packing_path(directory,'packing/'+relative)) == packing[field] == plan['build_files']['build/native-pack/'+relative], 'Actual-host pack evidence differs')
    need(packing['kind'] == 'native_cpu_pack' and packing['mining_source_changed'] is False and packing['every_nonce_SHA_retained'] is True, 'Packer provenance differs')
    need('cupy' not in sys.modules, 'CPU evidence checker imported CUDA')
    return {'status':'passed_standalone_holdout_verify_only_check', 'epoch':env['epoch'], 'blocks':BLOCKS, 'packers':PACKERS,
            'vector_roles':2, 'full_column_vectors_per_role':8, 'differential_full_state_nonces':51,
            'complete_cubin_sha256':cubins, 'metadata_sha256':sha(directory/'metadata-readbacks.json'),
            'capture_sha256':sha(directory/'compiled/benchmark-capture.json'), 'vectors_sha256':sha(vectors),
            'timing_claim':False, 'timed_measurements':0,
            'scope':'Pack library/proof identity checked. Verify-only vector serialization is canonical NumPy; no native-packing performance claim.'}

def main():
    need(__debug__ and not os.environ.get('PYTHONOPTIMIZE'), 'Assertions must be enabled')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('holdout-verify','epoch0-full'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    need(Path.cwd() == ROOT and HERE == ROOT/'submission-prep/round35', 'Run only from the pinned experiment root')
    plan = read(PLAN); stage_plan = plan['stages'][args.stage]
    need(args.output == Path(stage_plan['output']), 'Output differs from fixed execution plan')
    out = ROOT/args.output
    need(not out.exists() and out.parent.resolve() == ROOT/'round35/results', 'Output exists or escapes results')
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    env = dict(os.environ); env.update(BRAM_EPOCH=plan['holdout_seed'] if args.stage == 'holdout-verify' else 'epoch0', MINE_K='1', PYTHONDONTWRITEBYTECODE='1')
    with open('/tmp/bram-round30-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(); steps = []; before = None
        try:
            before = identities(plan)
            save(out/'before.json', before)
            if args.stage == 'epoch0-full': save(out/'heldout-prerequisite.json', prior_holdout(plan))
            save(out/'request.json', {'stage':args.stage, 'argv':sys.argv, 'plan':plan, 'plan_sha256':sha(PLAN), 'wrapper_sha256':sha(__file__),
                                     'scored_stage':args.stage == 'epoch0-full', 'concurrent_transfers_allowed':args.stage == 'holdout-verify'})
            benchdir = out/'benchmark'
            argv = [PYTHON,'-B',PACKAGE/'bench.py','--pairs','20','--nonces','1840','--nonce-base','5600000',
                    '--mode','full','--candidate-blocks','23','--candidate-packer','native',
                    '--pack-build',PACKAGE/'build/native-pack','--pack-proof',PACKAGE/'build/native-pack/cpu-proof.json','--output',benchdir]
            if args.stage == 'holdout-verify':
                inputs = out/'inputs'; inputs.mkdir()
                vectors = inputs/'vectors.txt'; shutil.copy2(ROOT/plan['holdout_vectors'], vectors)
                need(sha(vectors) == plan['holdout_vectors_sha256'], 'Copied vectors differ')
                argv += ['--vectors',vectors,'--verify-only']
            step(out,steps,'benchmark',argv,stage_plan['benchmark_timeout_seconds'],env)
            if args.stage == 'holdout-verify':
                def check_vectors(role,path,reference):
                    step(out,steps,'check-vectors-'+role,[PYTHON,'-B',PACKAGE/'host/check_vectors.py',path,reference],60,env)
                checked = verify_holdout(benchdir,vectors,plan,check_vectors)
            else:
                log = step(out,steps,'check-result',[PYTHON,'-B',PACKAGE/'check_result.py',benchdir,'--pairs','20','--nonces','1840',
                           '--nonce-base','5600000','--candidate-blocks','23','--candidate-packer','native'],180,env)
                checked = read(log)
                need(checked['status'] == 'passed_standalone_package_result_check' and checked['epoch'] == 'epoch0' and checked['blocks'] == BLOCKS and checked['packers'] == PACKERS, 'Scored result scope differs')
                need(checked['pairs'] == 20 and checked['nonces_per_sample'] == 1840 and checked['measurements_retained'] == 40 and checked['no_samples_dropped'] is True, 'Scored observations incomplete')
                need(checked['package'] == {'files_verified':27,'manifest_sha256':plan['manifest_sha256']} and checked['candidate_source_sha256'] == plan['candidate_sha256'] and checked['baseline_source_sha256'] == plan['stock_sha256'], 'Scored source/package binding differs')
            save(out/'result-check.json', checked)
            after = identities(plan); need(after == before, 'Inputs changed during standalone stage')
            save(out/'after.json',after)
            files = {str(x.relative_to(out)):sha(x) for x in sorted(out.rglob('*')) if x.is_file()}
            save(out/'completion.json', {'status':'completed', 'stage':args.stage, 'exit_code':0, 'steps':len(steps),
                 'all_step_exits_zero':all(x['exit_code'] == 0 for x in steps), 'package_unchanged':True,
                 'scored_stage':args.stage == 'epoch0-full', 'timing_claim':args.stage == 'epoch0-full',
                 'result_check_sha256':sha(out/'result-check.json'), 'files':files,
                 'manifest_sha256':plan['manifest_sha256'], 'archive_sha256':plan['archive_sha256'],
                 'execution_plan_sha256':sha(PLAN), 'wrapper_sha256':sha(__file__), 'finished_unix':time.time()})
            print(json.dumps({'status':'completed','stage':args.stage,'output':str(out),'result_check_sha256':sha(out/'result-check.json')}), flush=True)
        except BaseException as error:
            terminate_child()
            try:
                after = identities(plan); save(out/'after-failure.json',after)
            except BaseException as after_error:
                save(out/'after-failure.json',{'identity_check_error':str(after_error)})
            save(out/'failure.json',{'status':'failed','stage':args.stage,'error':str(error),'steps':steps,'before_recorded':before is not None})
            raise

if __name__ == '__main__':
    main()
