#!/usr/bin/env python3
"""Meaningful CPU-only byte, contract, exact-size and sanitizer checks."""
import argparse,ctypes,hashlib,json,platform,shutil,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from adapter import NativePacker,sha,HERE
ROOT=HERE.parent

def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def reference(a):
    return np.ascontiguousarray(a.astype('<u4',copy=False).view(np.uint8).reshape(*a.shape,4)[...,:3])
def checks(build_directory,output):
    pack=NativePacker(build_directory);module=SimpleNamespace(B=768,W=20);cases=[];negative=[]
    # All canonical20-bit values, independent expected byte positions.
    values=np.arange(1<<20,dtype=np.uint32);raw=np.empty(values.size*3,dtype=np.uint8)
    need(pack._call(values.ctypes.data,values.size,raw.ctypes.data,raw.nbytes)==0,'Exhaustive call rejected')
    expected=np.stack(((values&255).astype(np.uint8),((values>>8)&255).astype(np.uint8),((values>>16)&255).astype(np.uint8)),axis=1).reshape(-1)
    need(np.array_equal(raw,expected),'Exhaustive20-bit packing differs')
    cases.append({'name':'all20bit_values','words':values.size,'output_sha256':hashlib.sha256(raw).hexdigest()})
    rng=np.random.default_rng(350035)
    for count in (0,1,2,3,5,17,1840):
        for label,a in [('random_u32',rng.integers(0,2**32,size=(count,768),dtype=np.uint32)),
                        ('all_ones',np.full((count,768),0xffffffff,dtype=np.uint32))]:
            before=a.copy();actual=pack(module,a);expected=reference(a)
            need(np.array_equal(actual,expected) and np.array_equal(a,before),'Bulk packing/source differs')
            need(actual.flags.owndata and actual.flags.c_contiguous and actual.nbytes==count*2304,'Output ownership/size differs')
            cases.append({'name':label,'nonces':count,'output_sha256':hashlib.sha256(actual).hexdigest()})
    basis=np.zeros((1,768),dtype=np.uint32)
    for bit in range(32):
        for position in range(4):
            basis.fill(0);basis[0,position::4]=np.uint32(1<<bit)
            need(np.array_equal(pack(module,basis),reference(basis)),'Four-word bit placement differs')
    cases.append({'name':'all32bits_each4word_position','cases':128})
    vectors=[]
    for relative in ('packing/vectors/epoch0.txt','packing/vectors/round30.txt','packing/vectors/round34.txt'):
        path=ROOT/relative
        rows=[s.split() for s in path.read_text().splitlines() if s.strip() and not s.startswith('#')]
        need(len(rows)==8,'Expected eight independently generated CPU vectors')
        for nonce,digest,fold,state in rows:
            need(len(state)==768*5,'Golden state shape differs')
            a=np.array([int(state[i:i+5],16) for i in range(0,len(state),5)][::-1],dtype=np.uint32).reshape(1,768)
            packed=pack(module,a)
            independent=b''.join(int(v).to_bytes(4,'little')[:3] for v in a[0])
            need(packed.tobytes()==independent and hashlib.sha256(packed[0]).hexdigest()==digest,'Golden canonical digest differs')
        vectors.append({'path':relative,'sha256':sha(path),'vectors':8,'every_digest_equal':True})
    def reject(name,fn):
        try:fn()
        except (ValueError,OverflowError,RuntimeError):negative.append(name)
        else:raise RuntimeError('Negative accepted: '+name)
    a=np.zeros((2,768),dtype=np.uint32)
    reject('wrong_width',lambda:pack(SimpleNamespace(B=768,W=21),a))
    reject('wrong_lane_count',lambda:pack(SimpleNamespace(B=767,W=20),a))
    reject('wrong_dtype',lambda:pack(module,a.astype(np.uint64)))
    reject('opposite_endian',lambda:pack(module,a.astype('>u4' if sys.byteorder=='little' else '<u4')))
    reject('strided',lambda:pack(module,np.zeros((2,1536),dtype=np.uint32)[:,::2]))
    reject('wrong_shape',lambda:pack(module,a.reshape(-1)))
    reject('wrong_columns',lambda:pack(module,a[:,:767]))
    reject('non_array',lambda:pack(module,a.tolist()))
    # Pointer guards and deliberately misaligned source/output. Final-byte ASan
    # protection is exercised separately by exact-size C allocations below.
    for words in (0,1,2,3,4,5,7,31,768,769):
        for offset in range(8):
            input_buffer=ctypes.create_string_buffer(bytes([0xa5])*(words*4+32))
            output_buffer=ctypes.create_string_buffer(bytes([0x5a])*(words*3+32))
            address=ctypes.addressof(input_buffer)+offset+8;dest=ctypes.addressof(output_buffer)+offset+8
            expected=bytearray()
            for i in range(words):
                value=(i*2654435761)&0xffffffff
                ctypes.memmove(address+4*i,value.to_bytes(4,sys.byteorder),4);expected.extend(value.to_bytes(4,'little')[:3])
            prior=input_buffer.raw
            need(pack._call(address,words,dest,3*words)==0,'Misaligned call failed')
            actual=output_buffer.raw
            need(input_buffer.raw==prior and actual[offset+8:offset+8+3*words]==expected,'Unaligned byte mapping failed')
            need(actual[:offset+8]==bytes([0x5a])*(offset+8) and actual[offset+8+3*words:-1]==bytes([0x5a])*(24-offset),'Output guard overwritten')
    compiler=shutil.which('clang') or shutil.which('cc');need(compiler is not None,'Compiler required for CPU sanitizers')
    with tempfile.TemporaryDirectory(prefix='bram-pack-sanitizer-') as temporary:
        executable=Path(temporary)/'sanitize'
        command=[compiler,'-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fno-omit-frame-pointer','-fsanitize=address,undefined',str(HERE/'pack.c'),str(HERE/'sanitize.c'),'-o',str(executable)]
        compilation=subprocess.run(command,capture_output=True,text=True,timeout=60)
        need(compilation.returncode==0,'CPU sanitizer compilation failed: '+compilation.stderr)
        run=subprocess.run([str(executable)],capture_output=True,text=True,timeout=60)
        need(run.returncode==0 and 'checks passed' in run.stdout and not run.stderr,'CPU memory/UB sanitizer failed: '+run.stderr)
    pack.assert_unchanged()
    proof={'status':'passed_cpu_only_not_gpu_validated','sources':{p.name:sha(p) for p in (HERE/'pack.c',HERE/'adapter.py',HERE/'build.py',HERE/'sanitize.c',Path(__file__))},
        'build_record_sha256':sha(Path(build_directory)/'build.json'),'library_sha256':pack.record['library_sha256'],
        'python':platform.python_version(),'numpy':np.__version__,'machine':platform.machine(),
        'positive_cases':cases,'golden_vectors':vectors,'python_contract_rejections':negative,
        'unaligned_guard_cases':80,'bit_placement_cases':128,
        'sanitizer':{'status':'passed','tools':['address','undefined'],'command':command,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr}}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    return proof
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=checks(a.build,a.output);print(json.dumps({'status':result['status'],'golden_vectors':24,'contract_rejections':len(result['python_contract_rejections']),'output':str(a.output)}))
