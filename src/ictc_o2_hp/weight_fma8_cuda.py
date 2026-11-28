"""Offline-built SM80 SIMT weight kernel, launched on the current Torch stream."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import threading
import torch

_modules = {}
_lock = threading.Lock()
_driver = None


def _check(result, operation):
    if result:
        raise RuntimeError(f"CUDA driver {operation} failed with error {result}")


def _function():
    global _driver
    with _lock:
        if _driver is None:
            d = C.CDLL("libcuda.so.1")
            signatures = {
                "cuInit": [C.c_uint],
                "cuCtxGetCurrent": [C.POINTER(C.c_void_p)],
                "cuModuleLoadData": [C.POINTER(C.c_void_p), C.c_void_p],
                "cuModuleGetFunction": [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                "cuLaunchKernel": [C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
                    C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p,
                    C.POINTER(C.c_void_p), C.c_void_p],
            }
            for name, args in signatures.items():
                getattr(d, name).argtypes = args
                getattr(d, name).restype = C.c_int
            _check(d.cuInit(0), "cuInit")
            _driver = d
        context = C.c_void_p()
        _check(_driver.cuCtxGetCurrent(C.byref(context)), "cuCtxGetCurrent")
        if context.value is None:
            torch.cuda.synchronize()
            _check(_driver.cuCtxGetCurrent(C.byref(context)), "cuCtxGetCurrent")
        if context.value is None:
            raise RuntimeError("A current Torch CUDA context is required")
        if context.value not in _modules:
            folder = Path(__file__).parent / "kernels"
            meta = json.loads((folder / "weight_fma8_sm80.json").read_text())
            binary = (folder / "weight_fma8_sm80.cubin").read_bytes()
            source = (folder / "weight_fma8.cu").read_bytes()
            if hashlib.sha256(binary).hexdigest() != meta["cubin_sha256"] or hashlib.sha256(source).hexdigest() != meta["source_sha256"]:
                raise RuntimeError("Weight FMA8 source/binary seal mismatch")
            if meta["tile"] != [32, 64] or meta["warps"] != 4:
                raise RuntimeError("Unsupported weight FMA8 launch plan")
            blob = C.create_string_buffer(binary)
            module = C.c_void_p(); function = C.c_void_p()
            _check(_driver.cuModuleLoadData(C.byref(module), blob), "cuModuleLoadData")
            _check(_driver.cuModuleGetFunction(C.byref(function), module, b"weight_fma8"), "cuModuleGetFunction")
            _modules[context.value] = (module, function)
        return _driver, _modules[context.value][1]


def weight_partials(left, right, partials):
    """Return False for layouts/devices outside the sealed fast path."""
    if tuple(left.shape) != (256, 10504) or tuple(right.shape) != (10504, 1024) or tuple(partials.shape) != (4, 256, 1024):
        raise ValueError("Weight FMA8 expects the fixed 256x1024, K=10504 plan")
    if left.stride() != (1, 256) or not right.is_contiguous() or right.data_ptr() % 16:
        return False
    if any(v.dtype != torch.float32 or v.device != left.device for v in (left, right, partials)):
        raise ValueError("Weight FMA8 requires same-device CUDA FP32 tensors")
    if torch.cuda.get_device_capability(left.device) != (8, 0):
        return False
    with torch.cuda.device(left.device):
        driver, function = _function()
        stream = torch.cuda.current_stream(left.device)
        pointers = [C.c_uint64(v.data_ptr()) for v in (left, right, partials)]
        arguments = (C.c_void_p * 3)(*[C.addressof(v) for v in pointers])
        _check(driver.cuLaunchKernel(function, 8, 16, 4, 128, 1, 1, 0,
            C.c_void_p(stream.cuda_stream), arguments, None), "cuLaunchKernel")
        left.record_stream(stream)
        right.record_stream(stream)
    return True
