#!/usr/bin/env python3
"""Strict bulk native pack adapter; NumPy owns exact-sized output storage."""
import ctypes, hashlib, json, platform, struct
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NativePacker:
    def __init__(self,build_directory):
        self.directory=Path(build_directory).resolve()
        self.record=json.loads((self.directory/'build.json').read_text())
        r=self.record
        if r['status']!='built_not_benchmarked' or r['source_sha256']!=sha(HERE/'pack.c') or r['builder_sha256']!=sha(HERE/'build.py'):
            raise ValueError('Stale native build/source')
        self.library=(self.directory/r['library']).resolve()
        if self.library.parent!=self.directory or sha(self.library)!=r['library_sha256']:
            raise ValueError('Native library path/hash differs')
        if r['machine']!=platform.machine() or r['pointer_bits']!=struct.calcsize('P')*8:
            raise ValueError('Library architecture differs')
        self._lib=ctypes.CDLL(str(self.library))
        self._call=self._lib.bram_pack_u32_le24
        self._call.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
        self._call.restype=ctypes.c_int
    def __call__(self,module,states):
        if module.B!=768 or module.W!=20:raise ValueError('Exact768-lane20-bit layout required')
        if not isinstance(states,np.ndarray) or states.dtype!=np.dtype(np.uint32) or not states.dtype.isnative:
            raise ValueError('Native-endian uint32 ndarray required')
        if states.ndim!=2 or states.shape[1]!=768 or not states.flags.c_contiguous:
            raise ValueError('Contiguous N-by768 array required')
        words=int(states.size)
        if words>min(np.iinfo(np.intp).max//3,ctypes.c_size_t(-1).value//4):raise OverflowError('Packing size overflow')
        output=np.empty((len(states),768,3),dtype=np.uint8)
        status=self._call(states.ctypes.data,words,output.ctypes.data,output.nbytes)
        if status:raise RuntimeError('Native pack contract rejected: '+str(status))
        return output
    def assert_unchanged(self):
        if sha(self.library)!=self.record['library_sha256'] or sha(HERE/'pack.c')!=self.record['source_sha256']:
            raise RuntimeError('Native packing implementation changed')
