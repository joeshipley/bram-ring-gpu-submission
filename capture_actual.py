#!/usr/bin/env python3
"""Bounded, read-only capture of the exact live CuPy/NVRTC compilation.

No CUDA options or compiler return values are changed. A diagnostic failure is
recorded and does not authorize a cache repair or a different compiler backend.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TARGET = "ring_mine"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def byte_record(data):
    return {"bytes": len(data), "sha256": digest(data),
            "elf_magic": data.startswith(b"\x7fELF")}


def elf_code_identity(full, returned):
    """Compare every executable section, using the complete ELF's descriptors.

    This is an identity check, not a repair of the shorter CuPy return value.
    Refuse other ELF formats rather than guessing their layout.
    """
    if len(full) < 64 or full[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected complete little-endian ELF64")
    shoff = struct.unpack_from("<Q", full, 40)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", full, 58)
    if shentsize != 64 or not shnum or shstrndx >= shnum:
        raise ValueError("Unsupported ELF section table")
    if shoff + shentsize * shnum > len(full):
        raise ValueError("Complete NVRTC output has a truncated section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", full, shoff + i * 64)
                for i in range(shnum)]
    strings = sections[shstrndx]
    start, size = strings[4], strings[5]
    if start + size > len(full):
        raise ValueError("ELF string table exceeds complete output")
    names = full[start:start + size]
    result = []
    for section in sections:
        nameoff, _, flags, _, start, size, *_ = section
        if not flags & 4:  # SHF_EXECINSTR
            continue
        if start + size > len(full) or nameoff >= len(names):
            raise ValueError("Executable section exceeds complete output")
        end = names.find(b"\0", nameoff)
        if end < 0:
            raise ValueError("Unterminated ELF section name")
        actual = returned[start:start + size]
        expected = full[start:start + size]
        result.append({"name": names[nameoff:end].decode("utf-8", "replace"),
                       "offset": start, "bytes": size,
                       "complete_sha256": digest(expected),
                       "returned_sha256": digest(actual),
                       "identical": actual == expected})
    if not result:
        raise ValueError("No executable sections in complete output")
    return {"sections": result, "all_identical": all(r["identical"] for r in result),
            "target_section_present": any(TARGET in r["name"] for r in result)}


def same_loaded_nvrtc():
    """Open only the already mapped NVRTC library, never a second installation."""
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if re.fullmatch(r"libnvrtc\.so(?:\.[0-9]+)*", Path(path).name):
            mapped.add(str(Path(path).resolve()))
    if len(mapped) != 1:
        raise RuntimeError(f"Expected one mapped libnvrtc.so; found {sorted(mapped)}")
    path = next(iter(mapped))
    if not hasattr(os, "RTLD_NOLOAD"):
        raise RuntimeError("RTLD_NOLOAD unavailable; refusing another library load")
    lib = ctypes.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    lib.nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
    lib.nvrtcVersion.restype = ctypes.c_int
    lib.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetCUBINSize.restype = ctypes.c_int
    lib.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.nvrtcGetCUBIN.restype = ctypes.c_int
    major, minor = ctypes.c_int(), ctypes.c_int()
    status = lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    if status:
        raise RuntimeError(f"nvrtcVersion status {status}")
    return lib, {"path": path, "sha256": file_digest(path),
                 "version": [major.value, minor.value], "already_mapped_only": True}


def capture_full(program, expected_version):
    lib, metadata = same_loaded_nvrtc()
    if metadata["version"] != list(expected_version):
        raise RuntimeError("Mapped library and CuPy NVRTC versions disagree")
    size = ctypes.c_size_t()
    ptr = ctypes.c_void_p(int(program.ptr))
    status = lib.nvrtcGetCUBINSize(ptr, ctypes.byref(size))
    if status:
        raise RuntimeError(f"nvrtcGetCUBINSize status {status}")
    if not 64 <= size.value <= 256 * 1024 * 1024:
        raise RuntimeError(f"Unexpected CUBIN size {size.value}")
    buffer = ctypes.create_string_buffer(size.value)
    status = lib.nvrtcGetCUBIN(ptr, buffer)
    if status:
        raise RuntimeError(f"nvrtcGetCUBIN status {status}")
    return buffer.raw, metadata


def install_capture(compiler, directory, records):
    original = compiler._NVRTCProgram.compile

    def wrapped(program, *args, **kwargs):
        # This exact return object reaches CuPy unchanged, including a short
        # binary if that is what its wrapper returns. Original errors propagate.
        result = original(program, *args, **kwargs)
        if TARGET not in program.src:
            return result
        record = {"index": len(records), "status": "started",
                  "method": program.method, "program_name": program.name,
                  "options": list(args[0] if args else kwargs.get("options", ())),
                  "compile_return_object_unchanged": True}
        records.append(record)
        target = directory / f"capture-{record['index']:02d}"
        try:
            target.mkdir()
            source = program.src.encode("utf-8")
            (target / "actual-nvrtc-source.cu").write_bytes(source)
            record["source_sha256"] = digest(source)
            if program.method != "cubin":
                raise RuntimeError("NVRTC did not generate CUBIN; no backend substitution")
            returned = result[0]
            if not isinstance(returned, bytes):
                raise TypeError("Unexpected CuPy compiler result type")
            (target / "cupy-returned.cubin").write_bytes(returned)
            record["cupy_returned"] = byte_record(returned)
            full, library = capture_full(program, compiler.nvrtc.getVersion())
            (target / "nvrtc-complete.cubin").write_bytes(full)
            record["nvrtc_complete"] = byte_record(full)
            record["nvrtc_library"] = library
            record["relationship"] = {
                "identical": full == returned,
                "returned_is_complete_minus_final_byte": full[:-1] == returned,
                "complete_final_byte_hex": full[-1:].hex(),
                "size_difference": len(full) - len(returned),
                "executable_sections": elf_code_identity(full, returned)}
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        finally:
            # A capture I/O error must not replace a successful compilation.
            try:
                write_json(target / "capture.json", record)
            except Exception as exc:
                record["record_write_error"] = f"{type(exc).__name__}: {exc}"
        return result

    compiler._NVRTCProgram.compile = wrapped
    return original


def run_command(argv, deadline, maximum=20):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {"command": argv, "status": "skipped_deadline", "returncode": None,
                "stdout": "", "stderr": ""}
    started = time.monotonic()
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=min(maximum, remaining))
        result = {"command": argv, "status": "ok" if p.returncode == 0 else "failed",
                  "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        result = {"command": argv, "status": "timeout", "returncode": None,
                  "stdout": text(exc.stdout), "stderr": text(exc.stderr)}
    except OSError as exc:
        result = {"command": argv, "status": "unavailable", "returncode": None,
                  "stdout": "", "stderr": str(exc)}
    result["seconds"] = time.monotonic() - started
    return result


def find_tools(name):
    candidates = [os.environ.get(name.upper()), shutil.which(name),
                  f"/usr/local/cuda/bin/{name}"]
    candidates += [str(p) for p in Path(sys.prefix).glob(
        f"lib/python*/site-packages/nvidia/*/bin/{name}")]
    found = []
    for path in candidates:
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            path = str(Path(path).resolve())
            if path not in found:
                found.append(path)
    return found[:2]


def disassemble(directory, records, deadline):
    attempts = []
    tools = [("cuobjdump", p) for p in find_tools("cuobjdump")]
    tools += [("nvdisasm", p) for p in find_tools("nvdisasm")]
    if not tools:
        return {"status": "unavailable", "reason": "No disassembler found", "attempts": []}
    good = [r for r in records if r["status"] == "ok"]
    for record in good:
        target = directory / f"capture-{record['index']:02d}"
        binary = target / "nvrtc-complete.cubin"
        if file_digest(binary) != record["nvrtc_complete"]["sha256"]:
            raise RuntimeError("Complete CUBIN changed after capture")
        succeeded = False
        for kind, path in tools:
            stem = f"{kind}-{len(attempts):02d}"
            version = run_command([path, "--version"], deadline, 5)
            command = [path, "--dump-sass", str(binary)] if kind == "cuobjdump" else [path, str(binary)]
            outcome = run_command(command, deadline)
            write_json(target / f"{stem}.json", {"version": version, **outcome})
            (target / f"{stem}.stdout.txt").write_text(outcome["stdout"])
            (target / f"{stem}.stderr.txt").write_text(outcome["stderr"])
            valid = outcome["status"] == "ok" and TARGET in outcome["stdout"]
            attempts.append({"capture_index": record["index"], "tool": kind,
                             "path": path, "status": outcome["status"],
                             "target_present": TARGET in outcome["stdout"],
                             "evidence": str((target / f"{stem}.json").relative_to(directory))})
            if valid:
                (target / "kernel.sass").write_text(outcome["stdout"])
                succeeded = True
                break
        record["disassembly_success"] = succeeded
        write_json(target / "capture.json", record)
    return {"status": "ok" if good and all(r.get("disassembly_success") for r in good) else "failed",
            "attempts": attempts}


def cache_inventory(cache, records):
    result = []
    for path in sorted(cache.glob("*.cubin")):
        data = path.read_bytes()
        payload = data[40:]
        result.append({"file": path.name, **byte_record(payload),
                       "sha1_prefix_valid": data[:40] == hashlib.sha1(payload).hexdigest().encode(),
                       "matching_captures": [r["index"] for r in records
                           if r.get("cupy_returned", {}).get("sha256") == digest(payload)]})
    for record in records:
        record["matching_valid_cache_entries"] = [r["file"] for r in result
            if r["sha1_prefix_valid"] and record["index"] in r["matching_captures"]]
    return result


class BenchmarkCapture:
    """Optional observation of benchmark compilation, outside timed regions.

    Construct before the first CuPy import, register the actual module objects
    before timing, and finish in finally. Registering removes the hook so no
    diagnostic capture work runs during the timing loop.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache = self.directory / "cache"
        if self.cache.exists():
            raise RuntimeError("Benchmark capture requires a fresh cache directory")
        if "cupy" in sys.modules:
            raise RuntimeError("Begin benchmark capture before importing CuPy")
        self.cache.mkdir()
        os.environ["CUPY_CACHE_DIR"] = str(self.cache)
        os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "1"
        self.original = self.compiler = None
        self.finished = False
        self.report = {"status": "started", "captures": [], "modules": {},
                       "compile_options_changed": False, "cache_repaired": False,
                       "capture_hook_restored_before_timing": False,
                       "timing_completed": False, "hardware_counters_requested": False,
                       "purpose": "Observe compilation of the module objects passed to the benchmark",
                       "environment": {k: v for k, v in os.environ.items() if k.startswith("CUPY_")}}
        try:
            import cupy as cp
            from cupy.cuda import compiler
            self.compiler = compiler
            self.report["versions"] = {"cupy": cp.__version__, "nvrtc": compiler.nvrtc.getVersion(),
                                       "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                                       "cuda_driver": cp.cuda.runtime.driverGetVersion()}
            self.report["device"] = {"id": cp.cuda.Device().id,
                                     "compute_capability": cp.cuda.Device().compute_capability}
            self.original = install_capture(compiler, self.directory, self.report["captures"])
        except Exception as exc:
            self.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.finish(timing_completed=False)
            raise

    def restore(self):
        if self.original is not None:
            self.compiler._NVRTCProgram.compile = self.original
            self.original = None

    def register_modules(self, modules):
        try:
            for label, module in modules.items():
                attrs = dict(module._mine.attributes)  # force the same lazy compile now
                source = module.DEFS + module.BODY
                matches = []
                for record in self.report["captures"]:
                    path = self.directory / f"capture-{record['index']:02d}" / "actual-nvrtc-source.cu"
                    if path.exists():
                        compiled = path.read_text()
                        if source in compiled:
                            matches.append({"index": record["index"],
                                            "source_relationship": "identical" if source == compiled else "contains_complete_module_source"})
                self.report["modules"][label] = {
                    "variant": getattr(module, "PILOT_VARIANT", label),
                    "epoch": module.EPOCH, "MINE_K": module.MINE_K,
                    "candidate_source_sha256": getattr(module, "PILOT_SOURCE_SHA256", None),
                    "defs_body_sha256": digest(source.encode()),
                    "cupy_kernel_attributes": attrs, "matching_captures": matches}
        finally:
            self.restore()
            self.report["capture_hook_restored_before_timing"] = True
            write_json(self.directory / "benchmark-capture.json", self.report)

    def finish(self, *, timing_completed, disassemble=False):
        self.restore()
        if self.finished:
            return self.report
        self.finished = True
        self.report["timing_completed"] = bool(timing_completed)
        try:
            records = self.report["captures"]
            self.report["cache_entries"] = cache_inventory(self.cache, records)
            good = bool(records) and all(r["status"] == "ok" and r.get("matching_valid_cache_entries")
                and r["relationship"]["executable_sections"]["all_identical"]
                and r["relationship"]["executable_sections"]["target_section_present"] for r in records)
            modules_matched = bool(self.report["modules"]) and all(
                module["matching_captures"] for module in self.report["modules"].values())
            self.report["status"] = "ok" if good and modules_matched else "partial"
            for record in records:
                write_json(self.directory / f"capture-{record['index']:02d}" / "capture.json", record)
            if disassemble:
                self.report["disassembly"] = globals()["disassemble"](
                    self.directory, records, time.monotonic() + 40)
        except Exception as exc:
            self.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(self.directory / "benchmark-capture.json", self.report)
        return self.report


def begin_benchmark_capture(directory):
    return BenchmarkCapture(directory)

