#!/usr/bin/env python3
"""Explicit geometry derivation and optional proven native CPU serialization."""
import hashlib,inspect,json,platform,shutil,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def need(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(data):return hashlib.sha256(data).hexdigest()

def derived_timer(bench,packer=None):
    original=inspect.getsource(bench.measure)
    need(original.count('def measure(module, base, count, mode):')==1 and original.count('nblocks=460')==2 and original.count("'blocks': 460")==1,'Frozen timer edit sites differ')
    changed=original.replace('def measure(module, base, count, mode):','def measure(module, base, count, mode, blocks):').replace('nblocks=460','nblocks=blocks').replace("'blocks': 460","'blocks': blocks")
    inverse=changed.replace('def measure(module, base, count, mode, blocks):','def measure(module, base, count, mode):').replace('nblocks=blocks','nblocks=460').replace("'blocks': blocks","'blocks': 460")
    need(inverse==original,'Timer inverse differs')
    namespace=dict(bench.measure.__globals__)
    if packer is not None:namespace['packed_states']=packer
    exec(compile(changed,str(ROOT/'derived-full-timer.py'),'exec'),namespace)
    return namespace['measure'],{'original_function_sha256':digest(original.encode()),'derived_function_sha256':digest(changed.encode()),'inverse_source_equal':True,'changes':['explicit blocks parameter','two nblocks arguments','returned actual blocks field'],'packing_changed':packer is not None,'mining_launch_not_monkeypatched':True,'source':changed}

def native_packer(build_directory,proof_path,output):
    need(build_directory is not None and proof_path is not None,'Native packing requires --pack-build and --pack-proof')
    from packing.adapter import NativePacker
    pack=NativePacker(build_directory);proof_path=Path(proof_path).resolve();proof=json.loads(proof_path.read_text())
    need(proof['status']=='passed_cpu_only_not_gpu_validated' and proof['build_record_sha256']==sha(pack.directory/'build.json') and proof['library_sha256']==pack.record['library_sha256'],'CPU packing proof/build differs')
    expected=('pack.c','adapter.py','build.py','sanitize.c','cpu_check.py')
    need(set(proof['sources'])==set(expected),'CPU proof source coverage differs')
    for name,h in proof['sources'].items():need(sha(ROOT/'packing'/name)==h,'CPU-proved packing source differs')
    need(proof['sanitizer']['status']=='passed' and proof['sanitizer']['exit_code']==0,'CPU sanitizer proof missing')
    need(proof['machine']==platform.machine() and proof['python']==platform.python_version(),'CPU proof host/runtime differs')
    import numpy as np
    need(proof['numpy']==np.__version__,'CPU proof NumPy version differs')
    need(len(proof['golden_vectors'])==3 and all(x['vectors']==8 and x['every_digest_equal'] is True for x in proof['golden_vectors']),'Golden CPU vectors missing')
    for row in proof['golden_vectors']:
        path=(ROOT/row['path']).resolve();need(path.is_relative_to(ROOT) and sha(path)==row['sha256'],'Bundled CPU vectors differ')
    folder=output/'packing';folder.mkdir(exist_ok=False)
    shutil.copy2(pack.directory/'build.json',folder/'build.json');shutil.copy2(proof_path,folder/'cpu-proof.json');shutil.copy2(pack.library,folder/pack.library.name)
    record={'kind':'native_cpu_pack','library':str(Path('packing')/pack.library.name),'library_sha256':pack.record['library_sha256'],'build_record_sha256':sha(folder/'build.json'),'cpu_proof_sha256':sha(folder/'cpu-proof.json'),'sources':proof['sources'],'machine':pack.record['machine'],'pointer_bits':pack.record['pointer_bits'],'mining_source_changed':False,'every_nonce_SHA_retained':True}
    (folder/'binding.json').write_text(json.dumps(record,indent=2)+'\n');pack.assert_unchanged()
    return pack,record
