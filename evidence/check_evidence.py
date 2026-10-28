#!/usr/bin/env python3
"""Check published historical evidence using CPU/file operations only.

Does not import CuPy, run a miner, load the captured native library, contact a
provider, or rewrite any evidence. Python 3.12+ is recommended.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def safe(base,name):
    relative=Path(name);need(not relative.is_absolute() and '..' not in relative.parts,'Unsafe relative evidence path')
    path=base/relative;need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base.resolve()),'Missing or escaping evidence file')
    return path


def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values,q):
    values=sorted(values);at=(len(values)-1)*q;low=int(at);weight=at-low
    return values[low]*(1-weight)+values[min(low+1,len(values)-1)]*weight


def paired_stats(rows,role_key,cycle_key,candidate,baseline,base):
    need(len(rows)==40,'Expected all40 historical observations');a=[];b=[];first=[]
    for i in range(20):
        pair=rows[2*i:2*i+2];expected=[baseline,candidate] if i%2==0 else [candidate,baseline]
        need([r[role_key] for r in pair]==expected,'Actual paired order differs')
        values={}
        for row in pair:
            need(row[cycle_key]==i and row['nonces']==1840 and row['base']==base+1840*i and row['mode']=='full' and row['blocks']==(23 if row[role_key]==candidate else 460),'Historical work or nonce range differs')
            need(math.isfinite(row['seconds']) and row['seconds']>0 and math.isfinite(row['hashes_per_second']) and math.isclose(row['hashes_per_second'],1840/row['seconds'],rel_tol=1e-12),'Invalid elapsed time or rate')
            need(len(row['checksum'])==64 and all(c in '0123456789abcdef' for c in row['checksum']),'Malformed output checksum')
            values[row[role_key]]=row['seconds']
        need(pair[0]['checksum']==pair[1]['checksum'],'Paired full outputs differ')
        a.append(values[candidate]);b.append(values[baseline]);first.append(expected[0]==candidate)
    def gain(aa,bb):return 100*(statistics.mean(1840/x for x in aa)/statistics.mean(1840/x for x in bb)-1)
    rng=random.Random(350035);rates=[];elapsed=[]
    for _ in range(20000):
        ix=[rng.randrange(20) for _ in range(20)];aa=[a[i] for i in ix];bb=[b[i] for i in ix]
        rates.append(gain(aa,bb));elapsed.append(100*(sum(bb)/sum(aa)-1))
    return {'mean_rate_gain_percent':gain(a,b),'total_elapsed_gain_percent':100*(sum(b)/sum(a)-1),
            'mean_candidate_rate':statistics.mean(1840/x for x in a),'mean_baseline_rate':statistics.mean(1840/x for x in b),
            'paired_wins':sum(x<y for x,y in zip(a,b)),'paired_median_gain_percent':statistics.median(100*(y/x-1) for x,y in zip(a,b)),
            'candidate_first_gain_percent':gain([a[i] for i in range(20) if first[i]],[b[i] for i in range(20) if first[i]]),
            'baseline_first_gain_percent':gain([a[i] for i in range(20) if not first[i]],[b[i] for i in range(20) if not first[i]]),
            'first_half_gain_percent':gain(a[:10],b[:10]),'second_half_gain_percent':gain(a[10:],b[10:]),
            'descriptive_mean_rate_95_interval':[percentile(rates,q) for q in (.025,.975)],
            'descriptive_total_elapsed_95_interval':[percentile(elapsed,q) for q in (.025,.975)],'bootstrap_seed':350035,'bootstrap_resamples':20000,'pairs':20,'all_observations_retained':True}


def vectors(path):
    result={}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        nonce,digest,fold,state=line.split();n=int(nonce,16)
        need(n not in result and len(state)==768*5,'Vector count/columns differ')
        words=[int(state[i:i+5],16) for i in range(0,len(state),5)][::-1]
        need(hashlib.sha256(b''.join(v.to_bytes(3,'little') for v in words)).hexdigest()==digest,'Canonical vector digest differs')
        result[n]=(digest,int(fold,16),tuple(words))
    need(len(result)==8,'Expected eight distinct vectors');return result


def closed_case(name,kind):
    folder=ROOT/name;done=read(folder/('completion.json' if kind=='standalone' else 'geometry-completed.json'))
    need(done['status']=='completed' and not (folder/'failure.json').exists(),'Case did not close')
    for path,h in done['files'].items():need(sha(safe(folder,path))==h,'Closed evidence hash differs')
    expected=set(done['files'])|{('completion.json' if kind=='standalone' else 'geometry-completed.json')}
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    need(actual==expected,'Closed case inventory differs');return folder,done


def main():
    manifest=read(ROOT/'FILES.json');origin=read(ROOT/'ORIGIN.json')
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    need(actual==set(manifest)|{'FILES.json'},'Public evidence inventory differs')
    for name,h in manifest.items():need(sha(safe(ROOT,name))==h,'Public evidence file changed: '+name)
    for name,row in origin['copied_files'].items():need(manifest[name]==row['sha256'],'Historical source hash differs')
    package=ROOT/'package-v2';need(sha(package/'MANIFEST.json')==origin['historical_package_manifest_sha256'],'Historical package identity differs')
    for name,h in read(package/'MANIFEST.json')['files'].items():need(sha(package/name)==h,'Historical package payload differs')
    need(sha(ROOT/'historical-package-v2.tar.gz')=='f4337743330c71806b65b2b83dd59bec51952ee2c6f8431436a7970c59db0a57','Historical package archive differs')
    for name in ('spec.py','generate.py','vectors.txt','gpu/candidate.py','gpu/mine.py'):
        need(sha(ROOT.parent/name)==sha(package/name),'Current function/mining source differs from historical source')
    for controller,plan in [('controller-standalone-holdout','jobs-standalone-holdout-verify.json'),('controller-standalone-full','jobs-standalone-epoch0-full.json'),('controller-geometry-direct','jobs-geometry-direct.json')]:
        folder=ROOT/controller;jobs=jsonl(folder/'jobs.jsonl');expected=read(ROOT/'execution'/plan);done=read(folder/'completed.json')
        need(done['status']=='completed' and done['jobs']==len(jobs)==len(expected) and all(a['name']==b['name'] and a['argv']==b['argv'] and a['exit_code']==0 for a,b in zip(jobs,expected)),'Exact historical controller plan/exit differs')
    hold,hc=closed_case('standalone-holdout-v2','standalone');full,fc=closed_case('standalone-epoch0-full-v2','standalone')
    for folder,done in [(hold,hc),(full,fc)]:
        need(done['exit_code']==0 and done['all_step_exits_zero'] and done['package_unchanged'] and read(folder/'before.json')==read(folder/'after.json'),'Historical wrapper did not close unchanged')
        need(done['result_check_sha256']==sha(folder/'result-check.json'),'Wrapper result receipt differs')
        steps=read(folder/'steps.json');need(len(steps)==done['steps'] and all(s['exit_code']==0 and sha(folder/s['log'])==s['log_sha256'] for s in steps),'Actual wrapper child execution differs')
    prerequisite=read(full/'heldout-prerequisite.json');need(prerequisite['sha256']==sha(hold/'completion.json') and prerequisite['result_check_sha256']==sha(hold/'result-check.json'),'Full stage holdout prerequisite differs')
    hold_rows=jsonl(hold/'benchmark/measurements.jsonl');need(not any(r.get('type') in ('measurement','summary','failure') for r in hold_rows),'Unscored holdout contains timing/failure')
    expected_vectors=vectors(hold/'inputs/vectors.txt')
    for role in ('baseline','candidate'):need(vectors(hold/'benchmark'/(role+'-vectors.txt'))==expected_vectors,'Held-out full-column vectors differ')
    need([r for r in hold_rows if r.get('type')=='scratch_reuse']==[{'type':'scratch_reuse','differential_nonces':51,'launch_cases':[[1234567,8,2],[0x8123456789000000,17,3],[193,25,23],[0,1,24]],'all_equal':True}],'Recorded differential coverage differs')
    wrapper_spec=importlib.util.spec_from_file_location('historical_evidence_wrapper',ROOT/'execution/execute_standalone.py')
    wrapper=importlib.util.module_from_spec(wrapper_spec);wrapper_spec.loader.exec_module(wrapper)
    wrapper.PACKAGE=package  # Only relocate its read-only CPU checker; never call main.
    def check_vectors(role,path,reference):
        subprocess.run([sys.executable,'-B',str(package/'host/check_vectors.py'),str(path),str(reference)],check=True,capture_output=True,text=True,timeout=60)
    holdout_replay=wrapper.verify_holdout(hold/'benchmark',hold/'inputs/vectors.txt',read(ROOT/'execution/EXECUTION-PLAN.json'),check_vectors)
    need(holdout_replay==read(hold/'result-check.json'),'Historical holdout CPU checker replay differs')
    # The original checker reconstructs CUDA and executable-section identities,
    # validates metadata/build/proof bytes and vectors; it never executes CUDA.
    replay=subprocess.run([sys.executable,'-B',str(package/'check_result.py'),str(full/'benchmark'),'--candidate-blocks','23','--candidate-packer','native'],check=True,capture_output=True,text=True,timeout=60)
    saved=read(full/'result-check.json');need(json.loads(replay.stdout)==saved==read(full/'check-result.log'),'Immutable historical checker replay differs')
    measured=[r for r in jsonl(full/'benchmark/measurements.jsonl') if r.get('type')=='measurement'];stats=paired_stats(measured,'which','trial','candidate','stock',5600000)
    need(math.isclose(stats['mean_rate_gain_percent'],saved['mean_rate_gain_percent'],abs_tol=1e-10) and math.isclose(stats['total_elapsed_gain_percent'],saved['total_elapsed_gain_percent'],abs_tol=1e-10) and stats['paired_wins']==saved['paired_wins'],'Independent standalone arithmetic differs')
    direct={}
    for epoch in ('epoch0','fresh','holdout'):
        name='geometry-'+epoch+'-stock_native';folder,done=closed_case(name,'geometry');rows=jsonl(folder/'geometry-measurements.jsonl');analysis=read(ROOT/(name+'-analysis.json'))
        need(analysis['all_raw_observations']==rows and analysis['all_samples_retained'],'Direct raw rows differ from published analysis')
        value=paired_stats(rows,'arm','cycle','winner23_native','stock460_numpy',5740000);saved=analysis['comparisons']['winner23_native_vs_stock460_numpy']
        need(math.isclose(value['mean_rate_gain_percent'],saved['ratio_of_mean_rates_gain_percent'],abs_tol=1e-10) and math.isclose(value['total_elapsed_gain_percent'],saved['ratio_of_total_elapsed_gain_percent'],abs_tol=1e-10) and value['paired_wins']==saved['paired_wins'],'Independent direct arithmetic differs');direct[epoch]=value
    need('cupy' not in sys.modules,'CPU replay unexpectedly imported CUDA')
    print(json.dumps({'status':'passed_CPU_only_public_evidence_replay','historical_package_checker_matches':True,'historical_holdout_checker_matches':True,'historical_standalone':stats,'historical_direct_comparisons':direct,'holdout_vectors_per_role':8,'recorded_holdout_differential_nonces':51,'differential_states_saved_for_CPU_replay':False,'new_five_run_interface_GPU_tested':False,'organizer_validation':False},indent=2,allow_nan=False))


if __name__=='__main__':main()
