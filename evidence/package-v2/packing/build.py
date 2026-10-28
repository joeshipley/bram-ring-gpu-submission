#!/usr/bin/env python3
"""Build one local CPU library, recording compiler/flags/source/binary identity."""
import argparse, hashlib, json, os, platform, shutil, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(directory, compiler='cc'):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    cc=shutil.which(compiler)
    if not cc:raise RuntimeError('C compiler unavailable')
    version=subprocess.run([cc,'--version'],capture_output=True,text=True,check=True,timeout=20)
    library=directory/('libbram_pack.dylib' if platform.system()=='Darwin' else 'libbram_pack.so')
    argv=[cc,'-std=c11','-O3','-Wall','-Wextra','-Werror','-fPIC',
          '-dynamiclib' if platform.system()=='Darwin' else '-shared',str(HERE/'pack.c'),'-o',str(library)]
    process=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    (directory/'build.stdout').write_text(process.stdout);(directory/'build.stderr').write_text(process.stderr)
    if process.returncode:raise RuntimeError('Native library compilation failed')
    record={'status':'built_not_benchmarked','source_sha256':sha(HERE/'pack.c'),
        'builder_sha256':sha(__file__),'library':library.name,'library_sha256':sha(library),
        'command':argv,'compiler':str(Path(cc).resolve()),'compiler_version':version.stdout,
        'platform':platform.platform(),'machine':platform.machine(),'pointer_bits':__import__('struct').calcsize('P')*8,
        'no_march_native':True,'source':str(HERE/'pack.c')}
    (directory/'build.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    return record
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--cc',default='cc');a=p.parse_args()
    print(json.dumps(build(a.output,a.cc),indent=2))
