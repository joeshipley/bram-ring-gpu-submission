#!/usr/bin/env python3
"""Exercise actual judging-interface source fragments with CPU-only fixtures.

These tests do not import or execute either CUDA candidate. Synthetic timing
rows test orchestration and rejection logic only; they are not GPU evidence.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import bench
import check_result


def function(tree, name):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    return nodes[0]


def execute(nodes, namespace):
    # Execute the actual source statements, never a separate imitation of them.
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                 '<actual judging-interface source>', 'exec'), namespace)
    return namespace


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


BENCH_TREE = ast.parse((ROOT/'bench.py').read_text())
CHECK_TREE = ast.parse((ROOT/'check_result.py').read_text())
MAIN = function(BENCH_TREE, 'main')
CHECK = function(CHECK_TREE, 'check')


def parse_benchmark(argv):
    first = next(i for i,n in enumerate(MAIN.body) if assigned(n, 'parser'))
    last = next(i for i,n in enumerate(MAIN.body) if assigned(n, 'args'))
    guards = [n for n in MAIN.body if isinstance(n, ast.If)
              and any(isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
                      and x.value.id == 'args' and x.attr == 'pairs' for x in ast.walk(n.test))]
    # The count-scope guard and unsigned nonce-range guard are the only such ifs.
    scope = [n for n in guards if any(isinstance(x,ast.Constant) and x.value == 1840 for x in ast.walk(n.test))]
    assert len(scope) == 1
    namespace = {'argparse':argparse,'Path':Path,'ROOT':ROOT,'__doc__':bench.__doc__}
    old = sys.argv
    try:
        sys.argv = ['bench.py', *argv]
        execute(MAIN.body[first:last+1] + scope, namespace)
        return namespace['args']
    finally:
        sys.argv = old


def run_pair_loop(pairs, blocks, packer):
    loops = [n for n in ast.walk(MAIN) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'trial']
    assert len(loops) == 1
    rows=[];calls=[]
    stock=object();candidate=object()
    def timer(role):
        def measured(module, base, count, mode, actual_blocks):
            assert module is (stock if role == 'stock' else candidate)
            calls.append((role,base,count,mode,actual_blocks))
            return {'seconds':1.0,'hashes_per_second':float(count),'checksum':f'{base:064x}',
                    'base':base,'nonces':count,'blocks':actual_blocks}
        return measured
    checks=[]
    packing_object = SimpleNamespace(assert_unchanged=lambda: checks.append(True)) if packer == 'native' else None
    namespace={'args':SimpleNamespace(pairs=pairs,nonces=1840,nonce_base=5600000,mode='full'),
               'stock':stock,'candidate':candidate,'timers':{r:timer(r) for r in ('stock','candidate')},
               'blocks':{'stock':460,'candidate':blocks},'packers':{'stock':'numpy','candidate':packer},
               'packing_object':packing_object,'rates':{'stock':[],'candidate':[]},'emit':rows.append}
    execute(loops,namespace)
    assert len(rows) == len(calls) == 2*pairs
    for i in range(pairs):
        pair=rows[2*i:2*i+2]
        assert [r['which'] for r in pair] == (['stock','candidate'] if i%2 == 0 else ['candidate','stock'])
        for row in pair:
            assert (row['trial'],row['base'],row['nonces'],row['mode']) == (i,5600000+i*1840,1840,'full')
            assert row['blocks'] == (460 if row['which'] == 'stock' else blocks)
            assert row['packer'] == ('numpy' if row['which'] == 'stock' else packer)
    assert len(checks) == (2*pairs if packer == 'native' else 0)
    return rows


def validate_rows(rows, pairs, blocks, packer, nonces=1840):
    # Keep the scope assertion plus the complete actual measurement-count,
    # pair-order, checksum, nonce, elapsed/rate and summary-arithmetic section.
    guards=[n for n in CHECK.body if isinstance(n,ast.Assert)
            and any(isinstance(x,ast.Name) and x.id=='pairs' for x in ast.walk(n.test))
            and any(isinstance(x,ast.Constant) and x.value==1840 for x in ast.walk(n.test))]
    assert len(guards) == 1
    start=next(i for i,n in enumerate(CHECK.body) if assigned(n,'measured'))
    stop=next(i for i,n in enumerate(CHECK.body) if isinstance(n,ast.Assert)
              and any(isinstance(x,ast.Constant) and x.value=='scratch_reuse' for x in ast.walk(n)))
    rates={r:[x['hashes_per_second'] for x in rows if x.get('which')==r] for r in ('stock','candidate')}
    summary={'rates':rates,'mean_rates':{r:statistics.mean(v) if v else 0 for r,v in rates.items()},'speedup_percent':0.0}
    namespace={'rows':rows,'pairs':pairs,'nonces':nonces,'base':5600000,
               'blocks':{'stock':460,'candidate':blocks},'packers':{'stock':'numpy','candidate':packer},
               'summary':summary,'statistics':statistics,'math':check_result.math,'close':check_result.close}
    return execute(guards + CHECK.body[start:stop],namespace)


def main():
    if not __debug__:
        raise RuntimeError('Run without Python -O')
    rejected=[];accepted=[]
    def reject(label, thunk):
        with contextlib.redirect_stderr(io.StringIO()):
            try: thunk()
            except (AssertionError,SystemExit): rejected.append(label);return
        raise AssertionError('Invalid fixture accepted: '+label)
    assert parse_benchmark([]).pairs == 20
    for pairs in (5,20):
        for blocks in (23,460):
            for packer in ('numpy','native'):
                parsed=parse_benchmark(['--pairs',str(pairs),'--candidate-blocks',str(blocks),'--candidate-packer',packer])
                assert (parsed.pairs,parsed.nonces,parsed.candidate_blocks,parsed.candidate_packer) == (pairs,1840,blocks,packer)
                rows=run_pair_loop(pairs,blocks,packer)
                checked=validate_rows(rows,pairs,blocks,packer)
                assert len(checked['measured'])==2*pairs and checked['wins']==0
                accepted.append({'pairs':pairs,'blocks':blocks,'packer':packer,'synthetic_rows':len(rows)})
                reject(f'missing_row_{pairs}_{blocks}_{packer}',lambda:validate_rows(rows[:-1],pairs,blocks,packer))
                reject(f'extra_row_{pairs}_{blocks}_{packer}',lambda:validate_rows(rows+[rows[-1]],pairs,blocks,packer))
        rows=run_pair_loop(pairs,23,'native')
        wrong=copy.deepcopy(rows);wrong[0],wrong[1]=wrong[1],wrong[0]
        reject(f'wrong_order_{pairs}',lambda:validate_rows(wrong,pairs,23,'native'))
        wrong=copy.deepcopy(rows);wrong[1]['checksum']='f'*64
        reject(f'wrong_checksum_{pairs}',lambda:validate_rows(wrong,pairs,23,'native'))
        wrong=copy.deepcopy(rows);wrong[1]['blocks']=460
        reject(f'wrong_blocks_{pairs}',lambda:validate_rows(wrong,pairs,23,'native'))
        wrong=copy.deepcopy(rows);wrong[1]['packer']='numpy'
        reject(f'wrong_packer_{pairs}',lambda:validate_rows(wrong,pairs,23,'native'))
        reject(f'wrong_expected_pair_count_{pairs}',lambda:validate_rows(rows,20 if pairs==5 else 5,23,'native'))
    for pairs in (-1,0,1,4,6,19,21):
        reject('CLI_pairs_'+str(pairs),lambda pairs=pairs:parse_benchmark(['--pairs',str(pairs)]))
        reject('checker_pairs_'+str(pairs),lambda pairs=pairs:validate_rows(run_pair_loop(5,23,'native'),pairs,23,'native'))
    for count in (0,1839,1841):
        reject('CLI_nonces_'+str(count),lambda count=count:parse_benchmark(['--nonces',str(count)]))
        reject('checker_nonces_'+str(count),lambda count=count:validate_rows(run_pair_loop(5,23,'native'),5,23,'native',count))
    # The checker CLI also exposes exactly the documented choices/default.
    cli = [n for n in CHECK_TREE.body if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='__main__' for x in ast.walk(n.test))]
    assert len(cli)==1
    pairs_args=[n for n in ast.walk(cli[0]) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--pairs']
    assert len(pairs_args)==1
    assert {k.arg:ast.literal_eval(k.value) for k in pairs_args[0].keywords if k.arg in ('choices','default')} == {'choices':(5,20),'default':20}
    assert 'cupy' not in sys.modules
    print(json.dumps({'status':'passed_CPU_judging_interface_tests','accepted_scopes':accepted,
                      'rejected_controls':rejected,'rejected_count':len(rejected),
                      'defaults':{'pairs':20,'nonces':1840},'GPU_executed':False,
                      'scope':'Actual source orchestration and measurement-validation fragments with explicit synthetic CPU fixtures; no GPU correctness or performance claim.'},indent=2))


if __name__ == '__main__':
    main()
