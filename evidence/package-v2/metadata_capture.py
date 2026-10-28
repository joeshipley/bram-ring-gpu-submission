#!/usr/bin/env python3
"""Untimed readback of all eight existing metadata allocations; no GPU mutation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

TABLES = ('d_PREV','d_RULE','d_TIDX','d_TSH','d_AB','d_MULA','d_MULB','d_MULC')


def dump_metadata(module, role, output_dir):
    """Return and save one role record, plus the combined metadata-readbacks.json.

    Call once per role after compilation and before warmup/timing. Existing raw
    files/role entries are refused. This copies arrays to CPU without changing
    allocations, kernel arguments or CUDA source.
    """
    if role not in ('stock','candidate'):
        raise ValueError('Metadata role must be stock or candidate')
    import cupy as cp
    import numpy as np
    output = Path(output_dir)
    folder = output/'metadata';folder.mkdir(parents=True,exist_ok=True)
    manifest_path = output/'metadata-readbacks.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['epoch']!=module.EPOCH or role in manifest['roles']:
            raise RuntimeError('Metadata role already captured or epoch differs')
    else:
        manifest = {'schema_version':1,'epoch':module.EPOCH,'outside_timing':True,
                    'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'roles':{}}
    record = {'source_sha256':module.PILOT_SOURCE_SHA256,'module_object_id':id(module),'tables':{}}
    for name in TABLES:
        device = getattr(module,name)
        pointer = int(device.data.ptr)
        host = np.ascontiguousarray(cp.asnumpy(device))
        raw = host.tobytes(order='C')
        if int(device.data.ptr)!=pointer or len(raw)!=int(device.nbytes):
            raise RuntimeError('Metadata allocation changed during readback')
        filename = f'metadata/{role}-{name}.bin'
        with (output/filename).open('xb') as stream:
            stream.write(raw)
        record['tables'][name] = {'dtype':str(host.dtype),'dtype_str':host.dtype.str,
            'shape':list(host.shape),'nbytes':len(raw),'device_ptr':pointer,
            'sha256':hashlib.sha256(raw).hexdigest(),'raw_file':filename}
    manifest['roles'][role] = record
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return {'type':'metadata_readback','role':role,'epoch':module.EPOCH,
            'source_sha256':record['source_sha256'],'outside_timing':True,
            'table_sha256':{name:row['sha256'] for name,row in record['tables'].items()},
            'allocation_pointers':{name:row['device_ptr'] for name,row in record['tables'].items()},
            'helper_sha256':manifest['helper_sha256']}
