from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager

from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
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


def create_gpu_pipeline(ctx: ir.Context) -> PassManager:
    with ctx:
        pm = PassManager("builtin.module")
        pm.add("func.func(llvm-request-c-wrappers)")
        pm.add("convert-linalg-to-parallel-loops")
        pm.add("func.func(gpu-map-parallel-loops)")
        pm.add("func.func(convert-parallel-loops-to-gpu)")
        pm.add("gpu-kernel-outlining")
        pm.add("gpu-lower-to-nvvm-pipeline")

        pm.add("reconcile-unrealized-casts")
        pm.add("canonicalize")
        pm.add("cse")
        pm.add("gpu-to-llvm")
    return pm


def create_jit_pipeline(ctx: ir.Context) -> PassManager:
    """
    Lower linalg to LLVM dialect.
    """
    with ctx:
        pm = PassManager("builtin.module")

        pm.add("func.func(llvm-request-c-wrappers)")
        pm.add("convert-linalg-to-loops")
        pm.add("convert-scf-to-cf")
        pm.add("convert-to-llvm")
        pm.add("reconcile-unrealized-casts")
        pm.add("cse")
        pm.add("canonicalize")
    return pm

def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)
    print(kernel)

    pm = create_jit_pipeline(ctx)
    pm.run(kernel.operation)

    print(kernel)

    mlir_libs = []
    eng = ExecutionEngine(kernel, shared_libs=mlir_libs)
    eng.initialize()
    add_func = eng.lookup("add")

    a = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    b = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    out = np.ascontiguousarray(np.zeros(4, dtype=np.float32))

    a_memref = get_ranked_memref_descriptor(a)
    b_memref = get_ranked_memref_descriptor(b)
    out_memref = get_ranked_memref_descriptor(out)

    a_ptr = ctypes.pointer(ctypes.pointer(a_memref))
    b_ptr = ctypes.pointer(ctypes.pointer(b_memref))
    out_ptr = ctypes.pointer(ctypes.pointer(out_memref))

    args = memref.get_packed_arg([a_ptr, b_ptr, out_ptr])
    add_func(args)
    print(out)

if __name__ == "__main__":
    main()
