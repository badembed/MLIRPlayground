from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

import argparse
import ctypes
import numpy as np

from utils import memref


def create_kernel(ctx: ir.Context) -> ir.Module:
    with ctx:
        module = ir.Module.parse(
            r"""
    // Compute element-wise addition.
    func.func @add(%a: memref<4xf32>, %b: memref<4xf32>, %out: memref<4xf32>) {
        linalg.add ins(%a, %b : memref<4xf32>, memref<4xf32>)
                   outs(%out : memref<4xf32>)
        return
    }
"""
        )
    return module


class GenericMemRefF32(ctypes.Structure):
    _fields_ = [
        ("data",    ctypes.POINTER(ctypes.c_float)),
        ("offset",  ctypes.c_int64),
        ("sizes",   ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("rank",    ctypes.c_int64),
    ]


class GenericMemRefWrapper:
    def __init__(self, descriptor):
        sizes = list(descriptor.shape)
        strides = list(descriptor.strides)
        rank = len(sizes)

        if rank <= 0:
            raise ValueError("GenericMemRefWrapper: rank must be > 0")

        SizesArray   = ctypes.c_int64 * rank
        StridesArray = ctypes.c_int64 * rank

        self._sizes_buf   = SizesArray(*sizes)
        self._strides_buf = StridesArray(*strides)

        data_ptr = ctypes.cast(descriptor.aligned,
                               ctypes.POINTER(ctypes.c_float))

        gm = GenericMemRefF32()
        gm.data    = data_ptr
        gm.offset  = ctypes.c_int64(descriptor.offset)
        gm.sizes   = ctypes.cast(self._sizes_buf,
                                 ctypes.POINTER(ctypes.c_int64))
        gm.strides = ctypes.cast(self._strides_buf,
                                 ctypes.POINTER(ctypes.c_int64))
        gm.rank    = ctypes.c_int64(rank)

        self.gm = gm

    def ptr(self):
        return ctypes.byref(self.gm)


def load_simulator_lib():
    lib_path = "/home/alex/sources/ML/mlirPython2/playground/prebuilt_kernels/liblinalg_add.so"
    lib = ctypes.CDLL(lib_path)

    lib.sim_linalg_add_f32.argtypes = [
        ctypes.POINTER(GenericMemRefF32),
        ctypes.POINTER(GenericMemRefF32),
        ctypes.POINTER(GenericMemRefF32),
    ]
    lib.sim_linalg_add_f32.restype = None

    return lib


def run_lib_linalg_add(a_memref, b_memref, out_memref):
    a_gm = GenericMemRefWrapper(a_memref)
    b_gm = GenericMemRefWrapper(b_memref)
    out_gm = GenericMemRefWrapper(out_memref)

    lib = load_simulator_lib()
    lib.sim_linalg_add_f32(a_gm.ptr(), b_gm.ptr(), out_gm.ptr())

    return


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    # CPU buffers
    a = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    b = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    out = np.ascontiguousarray(np.zeros(4, dtype=np.float32))

    a_memref = get_ranked_memref_descriptor(a)
    b_memref = get_ranked_memref_descriptor(b)
    out_memref = get_ranked_memref_descriptor(out)

    run_lib_linalg_add(a_memref, b_memref, out_memref)

    print("out =", out)
    return


if __name__ == "__main__":
    main()
